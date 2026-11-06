import contextlib
import copy
import io
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defense_reporting import DefenseReporter
from defenses import DefenseAggregator


def capture(reporter, epoch, defense, result=None, final=False):
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        reporter.report_round(epoch, defense, result, final=final)
    return stream.getvalue()


class DefenseReportingTests(unittest.TestCase):
    def test_fedtgd_per_round_counts_ids_and_cumulative_rates(self):
        ids = [2, 5, 9, 11]
        args = SimpleNamespace(aggregation='FedTGD', attack=1, malicious=[5, 9])
        reporter = DefenseReporter(args, ids)
        defense = SimpleNamespace(method='FedTGD', client_ids=ids,
                                  fedtgd_client_mask=torch.tensor([False, False, True, True]))
        first = capture(reporter, 1, defense)
        self.assertIn('filtered_client_ids: [2, 5]', first)
        self.assertIn('TPR: 50.00% (1/2)', first)
        self.assertIn('FPR: 50.00% (1/2)', first)
        self.assertNotIn('Total', first)
        defense.fedtgd_client_mask = torch.tensor([True, False, False, True])
        second = capture(reporter, 2, defense, final=True)
        self.assertIn('filtered_client_ids: [5, 9]', second)
        self.assertIn('TPR: 100.00% (2/2)', second)
        self.assertIn('FPR: 0.00% (0/2)', second)
        total = second.splitlines()[-1]
        self.assertIn('Defense Diagnostics Total', total)
        self.assertIn('rounds: 2', total)
        self.assertIn('TPR: 75.00% (3/4)', total)
        self.assertIn('FPR: 25.00% (1/4)', total)
        self.assertNotIn('TFR:', second)
        with self.assertRaises(ValueError):
            capture(reporter, 2, defense, final=True)

    def test_fedtgd_clean_baseline_and_missing_denominators(self):
        ids = [0, 1]
        defense = SimpleNamespace(method='FedTGD', client_ids=ids,
                                  fedtgd_client_mask=torch.tensor([False, True]))
        reporter = DefenseReporter(SimpleNamespace(aggregation='FedTGD', attack=0, malicious=[0]), ids)
        log = capture(reporter, 1, defense, final=True)
        self.assertIn('TPR: N/A (0/0)', log)
        self.assertIn('FPR: 50.00% (1/2)', log)
        reporter = DefenseReporter(SimpleNamespace(aggregation='FedTGD', attack=1, malicious=ids), ids)
        log = capture(reporter, 1, defense, final=True)
        self.assertIn('TPR: 50.00% (1/2)', log)
        self.assertIn('FPR: N/A (0/0)', log)

    def test_rla_and_expguard_only_print_actual_client_weights(self):
        ids = [5, 2, 9]
        for method in ('RLA', 'ExpGuard'):
            args = SimpleNamespace(aggregation=method, attack=1, malicious=[2])
            reporter = DefenseReporter(args, ids)
            defense = SimpleNamespace(method=method, client_ids=ids)
            setattr(defense, 'rla_weights' if method == 'RLA' else 'expguard_weights',
                    torch.tensor([.6, .3, .1], dtype=torch.float64))
            log = capture(reporter, 1, defense, final=True)
            self.assertEqual(len(log.splitlines()), 1)
            self.assertIn('client_weights: [5=0.6, 2=0.3, 9=0.1]', log)
            for unused in ('TPR:', 'FPR:', 'TFR:', 'NFR:', 'Total'):
                self.assertNotIn(unused, log)

    def test_selective_per_round_and_cumulative_public_sample_filter_rates(self):
        args = SimpleNamespace(aggregation='Selective-FD', attack=1, malicious=[1])
        reporter = DefenseReporter(args, [0, 1], [1, 1, 4], 6)
        defense = SimpleNamespace(method='SelectiveFD', client_ids=[0, 1])
        result = SimpleNamespace(valid_samples=torch.tensor([False, False, True, True, False, True]))
        first = capture(reporter, 1, defense, result)
        self.assertIn('TFR: 100.00% (2/2)', first)
        self.assertIn('NFR: 25.00% (1/4)', first)
        result.valid_samples = torch.tensor([True, True, False, True, False, False])
        second = capture(reporter, 2, defense, result, final=True)
        self.assertIn('TFR: 50.00% (1/2)', second)
        self.assertIn('NFR: 50.00% (2/4)', second)
        total = second.splitlines()[-1]
        self.assertIn('TFR: 75.00% (3/4)', total)
        self.assertIn('NFR: 37.50% (3/8)', total)
        self.assertNotIn('TPR:', second)

    def test_selective_empty_groups_unknown_reference_and_no_cross_method_output(self):
        args = SimpleNamespace(aggregation='SelectiveFD', attack=1, malicious=[1])
        defense = SimpleNamespace(method='SelectiveFD', client_ids=[0, 1])
        result = SimpleNamespace(valid_samples=torch.tensor([False, True]))
        none_natural = capture(DefenseReporter(args, [0, 1], [], 2), 1, defense, result, True)
        self.assertIn('TFR: N/A (0/0)', none_natural)
        self.assertIn('NFR: 50.00% (1/2)', none_natural)
        all_natural = capture(DefenseReporter(args, [0, 1], [0, 1], 2), 1, defense, result, True)
        self.assertIn('TFR: 50.00% (1/2)', all_natural)
        self.assertIn('NFR: N/A (0/0)', all_natural)
        unknown = capture(DefenseReporter(args, [0, 1], None, 2), 1, defense, result, True)
        self.assertIn('TFR: N/A | NFR: N/A | trigger_reference: unavailable', unknown)
        other = DefenseReporter(SimpleNamespace(aggregation='mean', attack=0, malicious=[]), [0, 1])
        self.assertEqual(capture(other, 1, object(), object(), True), '')

    def test_reporting_leaves_defense_results_history_and_all_rng_unchanged(self):
        logits = torch.tensor([[[3., 1., .5], [2., 1., .5]],
                               [[3.1, 1., .5], [2.1, 1., .5]],
                               [[.5, 3., 1.], [.5, 2., 1.]]], dtype=torch.float64)
        for method in ('FedTGD', 'RLA', 'ExpGuard', 'SelectiveFD'):
            args = SimpleNamespace(aggregation=method, attack=1, malicious=[1], seed=123)
            with contextlib.redirect_stdout(io.StringIO()):
                defense = DefenseAggregator(args, [0, 1, 2])
                result = defense.aggregate(logits.unbind(), 1)
            baseline = copy.deepcopy(defense)
            saved_result = copy.deepcopy(result)
            torch_state, numpy_state, python_state = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
            reporter = DefenseReporter(args, [0, 1, 2], [0], 2)
            capture(reporter, 1, defense, result, final=True)
            self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
            np.testing.assert_array_equal(numpy_state[1], np.random.get_state()[1])
            self.assertEqual(numpy_state[2:], np.random.get_state()[2:])
            self.assertEqual(python_state, random.getstate())
            for name in ('probabilities', 'valid_samples', 'feedback_logits'):
                torch.testing.assert_close(getattr(result, name), getattr(saved_result, name), rtol=0, atol=0)
            for name, value in defense.__dict__.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, baseline.__dict__[name], rtol=0, atol=0)
                elif isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, baseline.__dict__[name])
                else:
                    self.assertEqual(value, baseline.__dict__[name])
            with contextlib.redirect_stdout(io.StringIO()):
                actual = defense.aggregate((logits + .2).unbind(), 2)
                expected = baseline.aggregate((logits + .2).unbind(), 2)
            torch.testing.assert_close(actual.probabilities, expected.probabilities, rtol=0, atol=0)
            torch.testing.assert_close(actual.feedback_logits, expected.feedback_logits, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
