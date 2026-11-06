import ast
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
ATTACK = ROOT/'src/malicious_client.py'
REPORT = ROOT/'Draw/实验结果汇总.py'


def load_methods(path):
    tree = ast.parse(path.read_bytes())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Malicious_client')
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                and node.name in {'_log_logit_deviation', 'local_predict_logits'}]
    scope = dict(torch=torch, time=time)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])), str(path), 'exec'), scope)
    return scope['Malicious_client']


ClientMethods = load_methods(ATTACK)
spec = importlib.util.spec_from_file_location('deviation_report', REPORT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def metric_line(before, after, indices=None, epoch=1, cid=0):
    client = ClientMethods()
    client.id = cid
    client.poisoned_indices = list(range(len(before))) if indices is None else indices
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        client._log_logit_deviation(before, after, epoch)
    return stream.getvalue()


def historical_global_line(cid=0, epoch=2, l2=1, cosine=1, count=2):
    return (f'Client {cid} | Epoch {epoch} | Logit Deviation | scope: client_flattened | '
            f'l2_global: {l2} | cosine_global: {cosine} | '
            f'selected_samples: {count} | logit_elements: {2 * count}\n')


def field(line, name):
    value = re.search(rf'{name}: ([^ |\n]+)', line).group(1)
    return None if value == 'N/A' else float(value)


class LogitDeviationTests(unittest.TestCase):
    def test_sample_means_differ_from_flattened_metrics(self):
        before = torch.tensor([[1., 0.], [0., 2.], [10., 20.]])
        after = torch.tensor([[0., 1.], [0., 4.], [1000., -2000.]])
        first, second = before.clone(), after.clone()
        line = metric_line(before, after, [0, 1])
        self.assertNotAlmostEqual(field(line, 'l2_mean'), math.sqrt(6), places=6)
        self.assertNotAlmostEqual(field(line, 'cosine_mean'), 8 / math.sqrt(85), places=7)
        self.assertAlmostEqual(field(line, 'l2_mean'), (math.sqrt(2) + 2) / 2, places=6)
        self.assertAlmostEqual(field(line, 'cosine_mean'), .5, places=7)
        self.assertEqual(field(line, 'selected_samples'), 2)
        self.assertEqual(field(line, 'cosine_samples'), 2)
        torch.testing.assert_close(before, first, rtol=0, atol=0)
        torch.testing.assert_close(after, second, rtol=0, atol=0)

    def test_zero_rows_count_for_l2_but_are_excluded_from_cosine_mean(self):
        before = torch.tensor([[0., 0.], [1., 0.]])
        after = torch.tensor([[3., 4.], [1., 0.]])
        line = metric_line(before, after)
        self.assertAlmostEqual(field(line, 'l2_mean'), 2.5)
        self.assertEqual(field(line, 'cosine_mean'), 1)
        self.assertEqual(field(line, 'cosine_samples'), 1)
        zeros = torch.zeros(2, 2)
        for other in (zeros, torch.ones(2, 2)):
            line = metric_line(zeros, other)
            self.assertIsNone(field(line, 'cosine_mean'))
            self.assertAlmostEqual(field(line, 'l2_mean'), 0 if other is zeros else math.sqrt(2), places=6)
        unchanged = metric_line(before, before)
        self.assertEqual(field(unchanged, 'l2_mean'), 0)
        self.assertEqual(field(unchanged, 'cosine_mean'), 1)

    def test_empty_invalid_and_duplicate_indices(self):
        before, after = torch.ones(2, 3), torch.full((2, 3), 2.)
        for indices in ([], [-1, 5]):
            line = metric_line(before, after, indices)
            self.assertIsNone(field(line, 'l2_mean'))
            self.assertIsNone(field(line, 'cosine_mean'))
            self.assertEqual(field(line, 'cosine_samples'), 0)
        line = metric_line(before, after, [0, 0, -1, 5])
        self.assertEqual(field(line, 'selected_samples'), 1)
        self.assertAlmostEqual(field(line, 'l2_mean'), math.sqrt(3), places=6)

    def test_every_round_logging_preserves_predictions_and_rng(self):
        client = ClientMethods()
        client.id, client.model = 0, torch.nn.Identity()
        client.args = SimpleNamespace(attack_type='Following_Nature', local_bs=2, device='cpu',
                                      epochs=3, back_target=1, target_prob=.95)
        client.poisoned_indices = [0, 2]
        client.use_global_feedback = False
        client.adaptive_margin = True
        client.desired_margin, client.up_margin_step = .1, .05
        client.logit_solver = 'closed_form'
        client.logit_solver_seconds, client.logit_solver_samples = 0., 0
        client.optimize_poisoned_logits_by_margin = lambda logits, *args: logits + torch.tensor([0., 2.])
        client._log_logit_examples = lambda *args: None
        inputs = torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
        data = torch.utils.data.TensorDataset(inputs, torch.zeros(3, dtype=torch.long))
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            results = [client.local_predict_logits(data, epoch) for epoch in (1, 2, 3)]
        log = stream.getvalue()
        for epoch in (1, 2, 3):
            self.assertEqual(log.count(f'Epoch {epoch} | Logit Deviation | l2_mean:'), 1)
        torch.testing.assert_close(results[0], inputs)
        expected = inputs.clone()
        expected[[0, 2], 1] += 2
        for result in results[1:]:
            torch.testing.assert_close(result, expected)
        first = next(line for line in log.splitlines() if 'Epoch 1 | Logit Deviation' in line)
        self.assertEqual(field(first, 'l2_mean'), 0)
        self.assertEqual(field(first, 'cosine_mean'), 1)
        self.assertEqual(log.count('Logit Solver Timing'), 1)
        states = torch.get_rng_state().clone(), random.getstate()
        metric_line(inputs, expected, [0, 2])
        self.assertTrue(torch.equal(states[0], torch.get_rng_state()))
        self.assertEqual(states[1], random.getstate())

    def parse_log(self, lines, malicious=(0, 1)):
        config = dict(epochs=2, clients=2, malicious=list(malicious), dataset='CIFAR10',
                      adaptive_margin=True, use_global_feedback=True)
        text = ''.join(lines) + report.TEST_MARKER + '\n'
        text += 'Client 0 | Epoch 2 | Acc: 60, ASR: 80\nClient 1 | Epoch 2 | Acc: 80, ASR: 90\n'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'train.log'
            path.write_text(text)
            return report.read_final_result(path, config, 'desired_margin_sensitivity', Path(directory))

    def test_report_final_round_keeps_each_clients_whole_metrics_separate(self):
        # Historical whole-vector logs remain readable without conversion.
        lines = [historical_global_line(cid=cid, epoch=1, l2=0) for cid in (0, 1)]
        lines += [historical_global_line(cid=0), historical_global_line(cid=1, l2=2, cosine='N/A')]
        row = self.parse_log(lines)
        self.assertEqual(row['deviation_scope'], 'client_flattened')
        self.assertIsNone(row['l2_mean'])
        self.assertIsNone(row['cosine_mean'])
        self.assertEqual([r['l2_global'] for r in row['client_deviations']], [1., 2.])
        self.assertIsNone(row['client_deviations'][1]['cosine_global'])
        self.assertEqual(row['acc_mean'], 70)
        report.validate_results([row])
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            report.print_table([row])
        for value in ('Client 0', 'Client 1', '客户端整体'):
            self.assertIn(value, stream.getvalue())

    def test_report_legacy_means_remain_unchanged(self):
        lines = [f'Client {cid} | Epoch 2 | Logit Deviation | l2_mean: {l2} | cosine_mean: {cos} | '
                 f'selected_samples: {count} | cosine_samples: {count}\n'
                 for cid, l2, cos, count in [(0, 3, .2, 2), (1, 7, .6, 6)]]
        row = self.parse_log(lines)
        self.assertEqual(row['deviation_scope'], 'per_sample_mean')
        self.assertEqual(row['l2_mean'], 6)
        self.assertAlmostEqual(row['cosine_mean'], .5)
        self.assertEqual(row['client_deviations'], [])
        report.validate_results([row])
        report.validate_results(report.DATA)

    def test_report_rejects_partial_duplicate_and_mixed_final_metrics(self):
        before = torch.ones(2, 2)
        line = metric_line(before, before, epoch=2, cid=0)
        with self.assertRaises(ValueError):
            self.parse_log([line])
        with self.assertRaises(ValueError):
            self.parse_log([line, line], malicious=(0,))
        legacy = historical_global_line(cid=1)
        with self.assertRaises(ValueError):
            self.parse_log([line, legacy])
        empty = metric_line(before, before, indices=[], epoch=2, cid=0)
        row = self.parse_log([empty], malicious=(0,))
        report.validate_results([row])
        self.assertIsNone(row['l2_mean'])


if __name__ == '__main__':
    unittest.main()
