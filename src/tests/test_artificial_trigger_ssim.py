import ast
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from skimage.metrics import structural_similarity
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artificial_trigger_metrics import ArtificialTriggerSSIM


def load_functions(path, names, scope):
    nodes = [node for node in ast.parse(path.read_bytes()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    namespace = dict(scope)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


import math
utilities = load_functions(ROOT/'utils.py', {
    'validate_artificial_trigger_strength', 'apply_white_and_paste_K', 'apply_K_with_blend'},
    dict(torch=torch, math=math))
evaluation = load_functions(ROOT/'Test.py', {'Artificial_evaluate_asr', 'compute_g',
                                           'evaluate_artificial_trigger_ssim'},
                           dict(torch=torch, nn=torch.nn, DataLoader=torch.utils.data.DataLoader,
                                utils=SimpleNamespace(**utilities),
                                ArtificialTriggerSSIM=ArtificialTriggerSSIM))


class ToyModel(torch.nn.Module):
    def forward(self, x):
        value = x[:, :, -1, -1].mean(1)
        return torch.stack((-value, value), dim=1)


def fixture():
    images = torch.linspace(-1, .5, 5 * 28 * 28).reshape(5, 1, 28, 28)
    images[:, 0, -1, -1] = torch.tensor([.2, .8, -.2, .9, -.5])
    labels = torch.tensor([0, 0, 1, 0, 0])
    args = SimpleNamespace(dataset='FashionMNIST', local_bs=3, device='cpu', back_target=1,
                           artificial_trigger_strength=.5, S=1)
    return images, torch.utils.data.TensorDataset(images, labels), args


class ArtificialTriggerSSIMTests(unittest.TestCase):
    def test_full_image_sample_mean_and_real_inverse_normalization(self):
        for dataset, channels in [('FashionMNIST', 1), ('CIFAR10', 3), ('GTSRB', 3)]:
            with self.subTest(dataset=dataset):
                metric = ArtificialTriggerSSIM(dataset)
                before = torch.linspace(-1, 1, channels * 28 * 28).reshape(channels, 28, 28)
                unchanged = before.clone()
                after = before.clone()
                after[:, -8:, -8:] = 4.  # Do not clip this injected value.
                metric.update(before, unchanged)
                metric.update(before, after)
                mean, std = metric._NORMALIZATION[dataset]
                mean = np.asarray(mean)[:, None, None]
                std = np.asarray(std)[:, None, None]
                original = before.numpy().astype(np.float64) * std + mean
                modified = after.numpy().astype(np.float64) * std + mean
                score = structural_similarity(original, modified, channel_axis=0, data_range=1.,
                                              gaussian_weights=True, sigma=1.5,
                                              use_sample_covariance=False)
                self.assertEqual(metric.count, 2)
                self.assertAlmostEqual(metric.mean, (1 + score) / 2, places=12)
                clipped = structural_similarity(original, modified.clip(0, 1), channel_axis=0,
                                                data_range=1., gaussian_weights=True, sigma=1.5,
                                                use_sample_covariance=False)
                self.assertNotAlmostEqual(score, clipped, places=6)
                torch.testing.assert_close(before, unchanged, rtol=0, atol=0)

    def test_actual_evaluator_only_counts_eligible_samples_and_preserves_asr_and_rng(self):
        images, data, args = fixture()
        original = images.clone()
        kernel = torch.ones(1, 1)
        for strength in (0., .5, 1.):
            with self.subTest(strength=strength):
                args.artificial_trigger_strength = strength
                rng = torch.get_rng_state()
                numpy_rng = np.random.get_state()
                baseline = evaluation['Artificial_evaluate_asr'](ToyModel(), data, kernel, 1, .5, args)
                after_baseline = torch.get_rng_state()
                metric = evaluation['evaluate_artificial_trigger_ssim'](data, kernel, 1, .5, args)
                self.assertTrue(torch.equal(torch.get_rng_state(), after_baseline))
                current_numpy = np.random.get_state()
                np.testing.assert_array_equal(numpy_rng[1], current_numpy[1])
                self.assertEqual(numpy_rng[2:], current_numpy[2:])
                torch.set_rng_state(rng)
                result = evaluation['Artificial_evaluate_asr'](ToyModel(), data, kernel, 1, .5, args)
                self.assertEqual(result, baseline)
                self.assertEqual(metric.count, 2)  # Original examples 0 and 4, all strengths.
                expected = ArtificialTriggerSSIM(args.dataset)
                for index in (0, 4):
                    injected = utilities['apply_K_with_blend'](images[index], kernel, 1,
                                                               alpha=strength, args=args)
                    expected.update(images[index], injected)
                self.assertEqual(metric.mean, expected.mean)
                if strength == 0:
                    self.assertEqual(metric.mean, 1.)
        torch.testing.assert_close(images, original, rtol=0, atol=0)

    def test_empty_and_non_image_reports_are_na(self):
        metric = ArtificialTriggerSSIM('CIFAR10')
        self.assertIsNone(metric.mean)
        self.assertIn('mean_ssim: N/A | samples: 0', metric.summary(.5))
        for dataset in ('AGNews', 'DBPedia', 'LOAN', 'SpeechCommands'):
            metric = ArtificialTriggerSSIM(dataset)
            metric.update(torch.ones(3), torch.zeros(3))
            self.assertIsNone(metric.mean)
            self.assertIn('image datasets only', metric.summary(0))

    def test_empty_cohort_and_non_image_measurement(self):
        _, data, args = fixture()
        metric = evaluation['evaluate_artificial_trigger_ssim'](data, torch.ones(1, 1), 1, -10, args)
        self.assertEqual(metric.count, 0)
        self.assertIsNone(metric.mean)
        for dataset in ('AGNews', 'DBPedia', 'LOAN', 'SpeechCommands'):
            args.dataset = dataset
            # Unsupported data must return N/A without even reading samples.
            metric = evaluation['evaluate_artificial_trigger_ssim'](None, None, 1, .5, args)
            self.assertIsNone(metric.mean)
            self.assertIn('image datasets only', metric.summary(.5))

    def test_main_reports_once_before_training_and_never_inside_rounds(self):
        images, data, args = fixture()
        args.epochs, args.attack, args.attack_type, args.aggregation = 3, 1, 'Following_Nature', 'mean'
        tree = ast.parse((ROOT/'main.py').read_bytes())
        fl = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'FL')
        loop = next(node for node in fl.body if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'epoch')
        call_name = 'evaluate_artificial_trigger_ssim'
        report = next(node for node in fl.body if isinstance(node, ast.If)
                      and any(isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                              and child.func.id == call_name for child in ast.walk(node)))
        create_clients = next(node for node in fl.body if isinstance(node, ast.Assign)
                              and any(isinstance(target, ast.Name) and target.id == 'clients'
                                      for target in node.targets))
        self.assertLess(report.lineno, create_clients.lineno)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id == call_name for node in ast.walk(loop)))
        clients = []
        for cid in (3, 7):
            clients.append(SimpleNamespace(id=cid, model=ToyModel(), local_train=lambda *a, **k: None,
                                           local_predict_logits=lambda *a, **k: torch.zeros(5, 2),
                                           local_distill=lambda *a, **k: None))
        meters, requests = [], []

        def measure(*a, **kw):
            result = evaluation['evaluate_artificial_trigger_ssim'](*a, **kw)
            meters.append(result)
            return result

        def evaluate(*a, **kw):
            self.assertFalse(kw)
            requests.append(True)
            return evaluation['Artificial_evaluate_asr'](*a, **kw)

        scope = dict(args=args, clients=clients, torch=torch, contextlib=contextlib, io=io,
                     evaluate_artificial_trigger_ssim=measure, Artificial_evaluate_asr=evaluate,
                     evaluate_asr=lambda *a: (75., 0., 80., 0.),
                     defense=None, defense_reporter=None, separator='---',
                     test_dataset=data, distill_dataset=data, new_test_dataset=data,
                     K=torch.ones(1, 1), alpha=.5,
                     client_acc_history={3: [], 7: []}, client_asr_history={3: [], 7: []},
                     client_ai_asr_history={3: [], 7: []})
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            exec(compile(ast.Module(body=[report, loop], type_ignores=[]), str(ROOT/'main.py'), 'exec'), scope)
        text = stream.getvalue()
        self.assertEqual(len(meters), 1)
        self.assertEqual(meters[0].count, 2)
        self.assertEqual(len(requests), 6)
        self.assertEqual(text.count('Artificial Trigger SSIM |'), 1)
        self.assertIn('Artificial Trigger SSIM | stage: before_training |', text)
        self.assertLess(text.index('Artificial Trigger SSIM |'), text.index('Epoch 1/3'))
        for cid in (3, 7):
            self.assertEqual(scope['client_acc_history'][cid], [75., 75., 75.])
            self.assertEqual(scope['client_asr_history'][cid], [80., 80., 80.])

        # The report is independent of communication-round count and attack-free
        # / other-attack experiments must not invoke it.
        meters.clear()
        for enabled, attack_type, expected in ((0, 'Following_Nature', 0),
                                               (1, 'SADBA', 0),
                                               (1, 'Following_Nature', 1)):
            args.attack, args.attack_type, args.epochs = enabled, attack_type, 1
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exec(compile(ast.Module(body=[report], type_ignores=[]), str(ROOT/'main.py'), 'exec'), scope)
            self.assertEqual(output.getvalue().count('Artificial Trigger SSIM |'), expected)


if __name__ == '__main__':
    unittest.main()
