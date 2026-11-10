import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, scope=None):
    tree = ast.parse(path.read_bytes())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    namespace = dict(torch=torch, math=math, nn=torch.nn, DataLoader=torch.utils.data.DataLoader)
    namespace.update(scope or {})
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), str(path), 'exec'), namespace)
    return namespace


utilities = load_functions(ROOT/'utils.py', {
    'validate_artificial_trigger_strength', 'apply_white_and_paste_K', 'apply_K_with_blend'})
blend = utilities['apply_K_with_blend']
legacy = utilities['apply_white_and_paste_K']
validate = utilities['validate_artificial_trigger_strength']
evaluation = load_functions(ROOT/'Test.py', {'Artificial_evaluate_asr', 'compute_g'},
                            {'utils': SimpleNamespace(**utilities)})


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.detach().clone())
        value = inputs[:, :, -1, -1].mean(1)
        return torch.stack((-value, value), dim=1)


class ArtificialTriggerStrengthTests(unittest.TestCase):
    def test_image_strengths_zero_fractional_one_and_extrapolation(self):
        for channels, kernel in [
            (1, torch.tensor([[.1, -.2], [.3, 2.]])),
            (3, torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 3),
        ]:
            image = torch.linspace(-1, 1, channels * 4 * 5).reshape(channels, 4, 5)
            original, original_kernel = image.clone(), kernel.clone()
            args = SimpleNamespace(dataset='FashionMNIST' if channels == 1 else 'CIFAR10')
            for strength in (0., .25, .5, 1., 1.5, 2.):
                with self.subTest(channels=channels, strength=strength):
                    actual = blend(image, kernel, 2, alpha=strength, args=args)
                    expected = image.clone()
                    expected[:, -2:, -2:] = ((1 - strength) * image[:, -2:, -2:]
                                              + strength * kernel)
                    torch.testing.assert_close(actual, expected)
                    if strength == 1:
                        torch.testing.assert_close(actual, legacy(image, kernel, 2, args), rtol=0, atol=0)
                    if strength == 2:
                        self.assertGreater(actual[:, -2:, -2:].max().item(), 1.)
            torch.testing.assert_close(image, original, rtol=0, atol=0)
            torch.testing.assert_close(kernel, original_kernel, rtol=0, atol=0)

    def test_continuous_tabular_region_and_integer_token_compatibility(self):
        vector = torch.tensor([1., 2., 3., 4.])
        kernel = torch.tensor([-.5, .5])
        actual = blend(vector, kernel, 2, alpha=.5, args=SimpleNamespace(dataset='LOAN'))
        torch.testing.assert_close(actual, torch.tensor([1., 2., 1.25, 2.25]))
        tokens = torch.tensor([2, 3, 5, 0, 0])
        args = SimpleNamespace(dataset='AGNews')
        torch.testing.assert_close(blend(tokens, kernel, 2, alpha=0, args=args), tokens, rtol=0, atol=0)
        torch.testing.assert_close(blend(tokens, kernel, 2, alpha=1, args=args),
                                   legacy(tokens, kernel, 2, args), rtol=0, atol=0)
        for strength in (.5, 1.5, 2):
            with self.assertRaisesRegex(ValueError, 'token IDs'):
                blend(tokens, kernel, 2, alpha=strength, args=args)

    def test_strength_validation_and_exact_default_overwrite(self):
        for value in (-.1, -1, float('nan'), float('inf'), -float('inf'), True, False, '0.5', None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(value)
        for value in (0, .5, 1, 2, 10.):
            self.assertEqual(validate(value), float(value))
        # At s=1 an extreme original region must not contaminate the original K overwrite.
        image = torch.full((1, 2, 2), float('nan'))
        kernel = torch.tensor([[.125, -3.], [2., .5]])
        actual = blend(image, kernel, 2, alpha=1)
        torch.testing.assert_close(actual, kernel.unsqueeze(0), rtol=0, atol=0)
        self.assertTrue(torch.isnan(image).all())

    def make_evaluation(self, strength=None):
        images = torch.zeros(5, 1, 2, 2)
        images[:, 0, -1, -1] = torch.tensor([.2, .8, -.2, .9, -.5])
        labels = torch.tensor([0, 0, 1, 0, 0])
        data = torch.utils.data.TensorDataset(images, labels)
        args = SimpleNamespace(dataset='FashionMNIST', local_bs=2, device='cpu', back_target=1)
        if strength is not None:
            args.artificial_trigger_strength = strength
        model = RecordingModel()
        metrics = evaluation['Artificial_evaluate_asr'](model, data, torch.ones(1, 1), 1, .5, args)
        return images, model, metrics

    def test_actual_artificial_asr_uses_yaml_strength_without_changing_sample_cohort(self):
        for strength, expected_asr in [(0., 50.), (.5, 100.), (1., 100.), (2., 100.)]:
            with self.subTest(strength=strength):
                images, model, metrics = self.make_evaluation(strength)
                self.assertEqual(metrics[0], expected_asr)
                # Only original g<alpha non-target examples 0 and 4 are evaluated for all strengths.
                self.assertEqual(len(model.inputs), 2)
                for index, actual in zip((0, 4), model.inputs):
                    expected = images[index:index + 1].clone()
                    expected[:, :, -1:, -1:] = (1 - strength) * expected[:, :, -1:, -1:] + strength
                    torch.testing.assert_close(actual, expected)
        _, default_model, default_metrics = self.make_evaluation()
        _, full_model, full_metrics = self.make_evaluation(1.)
        self.assertEqual(default_metrics, full_metrics)
        for default, full in zip(default_model.inputs, full_model.inputs):
            torch.testing.assert_close(default, full, rtol=0, atol=0)

    def test_invalid_strength_fails_even_if_no_examples_are_selected(self):
        args = SimpleNamespace(dataset='FashionMNIST', local_bs=2, device='cpu',
                               back_target=1, artificial_trigger_strength=-1)
        empty = torch.utils.data.TensorDataset(torch.empty(0, 1, 2, 2), torch.empty(0, dtype=torch.long))
        with self.assertRaises(ValueError):
            evaluation['Artificial_evaluate_asr'](RecordingModel(), empty, torch.ones(1, 1), 1, .5, args)
        args.dataset, args.artificial_trigger_strength = 'AGNews', .5
        with self.assertRaisesRegex(ValueError, 'token IDs'):
            evaluation['Artificial_evaluate_asr'](RecordingModel(), empty, torch.ones(1), 1, .5, args)


if __name__ == '__main__':
    unittest.main()
