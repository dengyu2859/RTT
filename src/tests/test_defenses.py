import contextlib
import copy
import io
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from defenses import (DefenseAggregator, DefenseResult,
                      distill_defended, resolve_options)


def aggregator(method, clients=3, options=None):
    args = SimpleNamespace(aggregation=method, seed=20250927, defense=options)
    with contextlib.redirect_stdout(io.StringIO()):
        return DefenseAggregator(args, list(range(clients)))


def aggregate(controller, probabilities, epoch=1):
    with contextlib.redirect_stdout(io.StringIO()):
        return controller.aggregate(probabilities.log().unbind(), epoch)


class DefenseTests(unittest.TestCase):
    def test_rla_classwise_raw_means_spectral_majority_and_cosine_weights(self):
        reps = torch.tensor([[3., 1., .5], [3.1, 1., .5], [2.9, 1., .5],
                             [.5, 3., 1.], [.5, 3.1, 1.]], dtype=torch.float64)
        offsets = torch.tensor([[2., -1., .5], [-2., 1., -.5]], dtype=torch.float64)
        logits = reps[:, None, :] + offsets[None, :, :]
        args = SimpleNamespace(aggregation="RLA", seed=20250927, temperature=3.,
                               defense={"probability_temperature": 99.})
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            controller = DefenseAggregator(args, list(range(5)))
            result = controller.aggregate(logits.unbind(), 1)
        torch.testing.assert_close(controller.rla_representatives, reps)
        labels = controller.rla_cluster_labels
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[3:])), 1)
        self.assertNotEqual(labels[0], labels[3])
        center = reps[:3].mean(0)
        weights = F.cosine_similarity(reps, center.expand_as(reps), dim=1).clamp_min(0)
        weights /= weights.sum()
        expected = (logits * weights[:, None, None]).sum(0)
        torch.testing.assert_close(controller.rla_center, center)
        torch.testing.assert_close(controller.rla_weights, weights)
        self.assertGreater(controller.rla_weights[3].item(), 0)  # Outside cluster still weighted.
        torch.testing.assert_close(result.feedback_logits, expected.float())
        torch.testing.assert_close(result.probabilities, (expected / 3).softmax(-1).float())
        probability_mean = (logits.softmax(-1) * weights[:, None, None]).sum(0)
        self.assertFalse(torch.allclose(result.probabilities, probability_mean.float()))
        self.assertTrue(result.valid_samples.all())
        self.assertIn("reference_clients: [0, 1, 2]", stream.getvalue())

    def test_rla_degenerate_inputs_singletons_and_negative_cosines(self):
        for logits in [torch.zeros(3, 2, 3), torch.ones(3, 2, 3),
                       torch.tensor([[[2., -1.], [1., 3.]]])]:
            controller = aggregator("RLA", clients=len(logits))
            with contextlib.redirect_stdout(io.StringIO()):
                result = controller.aggregate(logits.unbind(), 1)
            torch.testing.assert_close(result.feedback_logits, logits.mean(0))
            torch.testing.assert_close(controller.rla_weights,
                                       torch.full((len(logits),), 1. / len(logits), dtype=torch.float64))
        controller = aggregator("RLA", clients=2)
        logits = torch.tensor([[[1., 0.]], [[-1., 0.]]])
        with contextlib.redirect_stdout(io.StringIO()):
            result = controller.aggregate(logits.unbind(), 1)
        torch.testing.assert_close(controller.rla_weights, torch.tensor([1., 0.], dtype=torch.float64))
        torch.testing.assert_close(result.feedback_logits, logits[0])
        # A zero center has undefined cosine: documented equal-weight fallback.
        controller = aggregator("RLA", clients=2, options={"rla": {"n_clusters": 1}})
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            result = controller.aggregate(logits.unbind(), 1)
        self.assertIn("uniform_fallback: True", stream.getvalue())
        torch.testing.assert_close(result.feedback_logits, logits.mean(0))

    def test_rla_preserves_rng_and_is_equivariant_to_upload_order(self):
        logits = torch.tensor([[[3., 1.]], [[3.1, 1.]], [[2.9, 1.]],
                               [[1., 3.]], [[1., 3.1]]])
        torch.manual_seed(123)
        np.random.seed(123)
        random.seed(123)
        states = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        controller = aggregator("RLA", clients=5)
        with contextlib.redirect_stdout(io.StringIO()):
            result = controller.aggregate(logits.unbind(), 1)
        self.assertTrue(torch.equal(states[0], torch.get_rng_state()))
        self.assertEqual(states[2], random.getstate())
        numpy_state = np.random.get_state()
        self.assertEqual(states[1][0], numpy_state[0])
        np.testing.assert_array_equal(states[1][1], numpy_state[1])
        self.assertEqual(states[1][2:], numpy_state[2:])
        order = [3, 1, 4, 0, 2]
        with contextlib.redirect_stdout(io.StringIO()):
            permuted = DefenseAggregator(SimpleNamespace(aggregation="RLA", seed=20250927), order)
            actual = permuted.aggregate(logits[order].unbind(), 1, order)
        torch.testing.assert_close(actual.feedback_logits, result.feedback_logits)
        torch.testing.assert_close(permuted.rla_weights, controller.rla_weights[order])
        # RLA recomputes weights instead of accumulating history across rounds.
        with contextlib.redirect_stdout(io.StringIO()):
            repeated = controller.aggregate(logits.unbind(), 2)
        torch.testing.assert_close(repeated.feedback_logits, result.feedback_logits)

    def check_raw_aggregation_kd(self, method):
        x = torch.tensor([[1., -1.], [.5, 1.], [-.5, .2]])
        dataset = torch.utils.data.TensorDataset(x, torch.zeros(3, dtype=torch.long))
        args = SimpleNamespace(aggregation=method, seed=20250927, temperature=3.,
                               local_bs=3, distill_lr=.1, distill_momentum=0.,
                               distill_ep=1, device="cpu")
        logits = torch.tensor([[[2., -1.], [1., 3.], [-2., 1.]],
                               [[3., -1.], [2., 3.], [-2., 2.]]])
        model = torch.nn.Linear(2, 2, bias=False)
        expected = copy.deepcopy(model)
        with contextlib.redirect_stdout(io.StringIO()):
            result = DefenseAggregator(args, [0, 1]).aggregate(logits.unbind(), 1)
        optimizer = torch.optim.SGD(expected.parameters(), lr=.1)
        optimizer.zero_grad()
        loss = F.kl_div(F.log_softmax(expected(x) / 3, dim=1),
                        (result.feedback_logits / 3).softmax(-1), reduction="batchmean") * 9
        loss.backward()
        optimizer.step()
        client = SimpleNamespace(id=0, args=args, model=model, global_logits=None)
        distill_defended(client, dataset, result, 1)
        torch.testing.assert_close(client.model.weight, expected.weight)
        torch.testing.assert_close(client.global_logits, result.feedback_logits)

    def test_raw_aggregation_kd_matches_original_temperature_logits_objective(self):
        for method in ("RLA", "ExpGuard", "FedTGD"):
            with self.subTest(method=method):
                self.check_raw_aggregation_kd(method)

    def test_expguard_global_scores_history_and_raw_logit_weighting(self):
        q = torch.tensor([
            [[.90, .05, .05], [.80, .12, .08], [.7, .1, .2]],
            [[.80, .10, .10], [.72, .18, .10], [.1, .2, .7]],
            [[.10, .20, .70], [.08, .12, .80], [.1, .7, .2]],
        ], dtype=torch.float64)
        controller = aggregator("ExpGuard", options={"chunk_size": 2})
        cumulative = torch.zeros(3, dtype=torch.float64)
        for epoch in (1, 2, 3):
            # Different clients deviate in different rounds. An identical final round
            # must retain the weight differences accumulated in earlier rounds.
            values = q if epoch == 1 else (q.roll(1, 0) if epoch == 2 else q[:1].expand_as(q))
            sample_scores = []
            for sample in range(values.shape[1]):
                predictions = values[:, sample]
                _, vectors = torch.linalg.eigh(torch.cov(predictions.T))
                sample_scores.append(((predictions - predictions.mean(0)) @ vectors[:, -1]).abs())
            scores = torch.stack(sample_scores).mean(0)
            cumulative += scores
            weights = (-cumulative).softmax(0)
            # Client-specific row offsets disappear in softmax scoring but must remain
            # in raw-logit aggregation and in feedback to the unchanged attack.
            offsets = torch.arange(9, dtype=torch.float64).reshape(3, 3, 1) * .4
            logits = values.log() + offsets
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                result = controller.aggregate(logits.unbind(), epoch)
            expected = (logits * weights[:, None, None]).sum(0)
            torch.testing.assert_close(controller.expguard_scores, scores, atol=1e-10, rtol=1e-6)
            torch.testing.assert_close(controller.expguard_weights, weights)
            torch.testing.assert_close(controller.log_weights.softmax(0), weights)
            torch.testing.assert_close(result.feedback_logits, expected.float())
            torch.testing.assert_close(result.probabilities, expected.softmax(-1).float())
            self.assertTrue(result.valid_samples.all())
            self.assertIn("client_scores:", stream.getvalue())
            self.assertIn("client_weights:", stream.getvalue())
        self.assertFalse(torch.allclose(weights, torch.full_like(weights, 1 / 3)))

    def test_expguard_eta_zero_raw_mean_and_chunk_independence(self):
        logits = torch.tensor([
            [[5., -1.], [-2., 3.], [1., 2.]],
            [[-3., 2.], [1., -2.], [3., 1.]],
            [[2., 1.], [4., -1.], [-1., 3.]],
        ], dtype=torch.float64)
        controllers = [aggregator("ExpGuard", options={"chunk_size": size}) for size in (1, 2, 256)]
        with contextlib.redirect_stdout(io.StringIO()):
            results = [c.aggregate(logits.unbind(), 1) for c in controllers]
        for c, result in zip(controllers[1:], results[1:]):
            torch.testing.assert_close(c.expguard_scores, controllers[0].expguard_scores)
            torch.testing.assert_close(result.feedback_logits, results[0].feedback_logits)
        controller = aggregator("ExpGuard", options={"expguard": {"eta": 0.}})
        with contextlib.redirect_stdout(io.StringIO()):
            result = controller.aggregate(logits.unbind(), 1)
        torch.testing.assert_close(result.feedback_logits, logits.mean(0).float())
        torch.testing.assert_close(controller.expguard_weights, torch.full((3,), 1 / 3, dtype=torch.float64))

    def test_expguard_identical_predictions_and_single_client(self):
        q = torch.tensor([[[.2, .8]]] * 3, dtype=torch.float64)
        controller = aggregator("ExpGuard")
        result = aggregate(controller, q)
        torch.testing.assert_close(controller.log_weights, torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(result.probabilities, q[0].float())
        controller = aggregator("ExpGuard", clients=1)
        torch.testing.assert_close(aggregate(controller, q[:1]).probabilities, q[0].float())

    def test_fedtgd_filters_separated_outlier_without_attack_metadata(self):
        q = torch.tensor([[[.9, .05, .05]]] * 5 + [[[.02, .02, .96]]], dtype=torch.float64)
        result = aggregate(aggregator("FedTGD", clients=6), q)
        torch.testing.assert_close(result.probabilities, q[0].float())

    def test_fedtgd_all_noise_is_conservative_raw_logit_mean(self):
        logits = torch.tensor([[[4., -1.], [2., 1.]], [[-2., 3.], [5., -1.]]])
        controller = aggregator("FedTGD", clients=2)
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            result = controller.aggregate(logits.unbind(), 1)
        expected = logits.mean(0)
        torch.testing.assert_close(result.feedback_logits, expected)
        torch.testing.assert_close(result.probabilities, expected.softmax(-1))
        self.assertTrue(controller.fedtgd_client_mask.all())
        self.assertIn("all_noise_fallback: True", stream.getvalue())

    def test_fedtgd_topk_keeps_original_class_coordinates(self):
        q = torch.tensor([[[.9, .1]], [[.1, .9]]], dtype=torch.float64)
        controller = aggregator("FedTGD", clients=2, options={
            "fedtgd": {"eps": 2.0, "min_samples": 1, "top_k": 1, "cosine_threshold": .5}})
        result = aggregate(controller, q)
        # Sorting just top-k values would incorrectly make opposite classes identical.
        torch.testing.assert_close(result.probabilities, q[0].float())

    def test_fedtgd_selects_whole_clients_using_aligned_public_predictions(self):
        # Identical class means hide the flipped sample alignment of the outlier.
        q = torch.tensor([[[.9, .1], [.1, .9], [.8, .2]]] * 5 +
                         [[[.1, .9], [.9, .1], [.8, .2]]], dtype=torch.float64)
        torch.testing.assert_close(q[0].mean(0), q[-1].mean(0))
        offsets = torch.arange(6, dtype=torch.float64)[:, None, None] * .01
        offsets[-1] = 100.
        logits = q.log() + offsets
        for chunk in (1, 2, 256):
            controller = aggregator("FedTGD", clients=6, options={"chunk_size": chunk})
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                result = controller.aggregate(logits.unbind(), 1)
            self.assertEqual(controller.fedtgd_client_mask.tolist(), [True] * 5 + [False])
            # Excluded even on the last row, where its probabilities match the majority.
            torch.testing.assert_close(result.feedback_logits, logits[:5].mean(0).float())
            self.assertTrue(result.valid_samples.all())
            self.assertIn("rejected_client_ids: [5]", stream.getvalue())

    def test_fedtgd_global_topk_cosine_matches_flattened_reference(self):
        q = torch.tensor([[[.9, .1], [.1, .9]],
                          [[.9, .1], [.9, .1]],
                          [[.1, .9], [.1, .9]]], dtype=torch.float64)
        options = {"chunk_size": 1, "fedtgd": {
            "eps": 2., "min_samples": 1, "top_k": 1, "cosine_threshold": .7}}
        controller = aggregator("FedTGD", clients=3, options=options)
        with contextlib.redirect_stdout(io.StringIO()):
            result = controller.aggregate(q.log().unbind(), 1)
        # Aligned top-1 sparse vectors have similarities [[1,.5,.5],[.5,1,0],[.5,0,1]].
        # Whole-dataset anchor is client 0; threshold .7 keeps only client 0.
        self.assertEqual(controller.fedtgd_client_mask.tolist(), [True, False, False])
        torch.testing.assert_close(result.feedback_logits, q[0].log().float())

    def test_fedtgd_ties_are_stable_under_upload_order_and_rng_unchanged(self):
        q = torch.tensor([[[.9, .1], [.8, .2]], [[.1, .9], [.2, .8]]], dtype=torch.float64)
        options = {"fedtgd": {"eps": .2, "min_samples": 1}}
        a = aggregator("FedTGD", clients=2, options=options)
        states = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        with contextlib.redirect_stdout(io.StringIO()):
            first = a.aggregate(q.log().unbind(), 1)
        self.assertTrue(torch.equal(states[0], torch.get_rng_state()))
        np.testing.assert_array_equal(states[1][1], np.random.get_state()[1])
        self.assertEqual(states[1][2:], np.random.get_state()[2:])
        self.assertEqual(states[2], random.getstate())
        with contextlib.redirect_stdout(io.StringIO()):
            b = DefenseAggregator(SimpleNamespace(aggregation="FedTGD", seed=1, defense=options), [1, 0])
            second = b.aggregate(q.flip(0).log().unbind(), 1, [1, 0])
        torch.testing.assert_close(first.feedback_logits, second.feedback_logits)
        self.assertEqual(a.fedtgd_client_mask.tolist(), [True, False])
        self.assertEqual(b.fedtgd_client_mask.tolist(), [False, True])
        # A singleton falls back to its original logits at the default min_samples=5.
        c = aggregator("FedTGD", clients=1)
        with contextlib.redirect_stdout(io.StringIO()):
            single = c.aggregate(q[:1].log().unbind(), 1)
        torch.testing.assert_close(single.feedback_logits, q[0].log().float())

    def test_selective_server_only_all_client_mean_and_feedback_alignment(self):
        q = torch.tensor([
            [[.9, .1], [.9, .1], [.1, .9]],
            [[.7, .3], [.1, .9], [.3, .7]],
        ], dtype=torch.float64)
        controller = aggregator("SelectiveFD", clients=2, options={
            "selective_fd": {"server_threshold": .5}})
        # No preparation or local datasets are supplied. All client rows contribute.
        result = aggregate(controller, q)
        self.assertEqual(result.valid_samples.tolist(), [True, False, True])
        torch.testing.assert_close(result.probabilities, q.mean(0).float())
        torch.testing.assert_close(result.feedback_logits.softmax(-1),
                                   torch.tensor([[.8, .2], [.5, .5], [.2, .8]]))
        next_q = torch.tensor([[[.5, .5], [.9, .1], [.5, .5]]] * 2, dtype=torch.float64)
        result = aggregate(controller, next_q, epoch=2)
        self.assertEqual(result.valid_samples.tolist(), [False, True, False])
        torch.testing.assert_close(result.feedback_logits.softmax(-1),
                                   torch.tensor([[.8, .2], [.9, .1], [.2, .8]]))

    def test_selective_server_threshold_boundaries_alias_and_single_client(self):
        q = torch.tensor([[[.8, .2], [.5, .5]], [[.8, .2], [.5, .5]]], dtype=torch.float64)
        for name in ("SelectiveFD", "Selective-FD"):
            for threshold, accepted in [(0., [False, False]), (.4, [True, False]), (2., [True, True])]:
                with self.subTest(name=name, threshold=threshold):
                    controller = aggregator(name, clients=2, options={
                        "selective_fd": {"server_threshold": threshold}})
                    result = aggregate(controller, q)
                    self.assertEqual(result.valid_samples.tolist(), accepted)
                    torch.testing.assert_close(result.probabilities, q.mean(0).float())
                    if threshold == 0:
                        torch.testing.assert_close(result.feedback_logits.softmax(-1), torch.full((2, 2), .5))
        controller = aggregator("SelectiveFD", clients=1, options={"selective_fd": {"server_threshold": 2.}})
        torch.testing.assert_close(aggregate(controller, q[:1]).probabilities, q[0].float())

    def test_selective_legacy_client_options_are_ignored_without_rng_changes(self):
        q = torch.tensor([[[.95, .05], [.5, .5]], [[.85, .15], [.5, .5]]], dtype=torch.float64)
        # Removed parameters, including formerly invalid values, no longer fit or calibrate anything.
        old_options = {"selective_fd": {"server_threshold": .5, "max_local_samples": 0,
                       "basis_size": 0, "uniform_samples": 0, "validation_fraction": -1,
                       "client_quantile": -1, "regularization": 0}}
        clean = aggregator("SelectiveFD", clients=2, options={"selective_fd": {"server_threshold": .5}})
        torch_state, np_state, py_state = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        legacy = aggregator("SelectiveFD", clients=2, options=old_options)
        actual, expected = aggregate(legacy, q), aggregate(clean, q)
        self.assertEqual(legacy.options["selective_fd"], {"server_threshold": .5})
        torch.testing.assert_close(actual.probabilities, expected.probabilities)
        self.assertTrue(torch.equal(actual.valid_samples, expected.valid_samples))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        np.testing.assert_array_equal(np_state[1], np.random.get_state()[1])
        self.assertEqual(np_state[2:], np.random.get_state()[2:])
        self.assertEqual(py_state, random.getstate())

    def test_distillation_skips_rejected_rows_and_does_not_softmax_targets_twice(self):
        args = SimpleNamespace(local_bs=3, device=torch.device("cpu"), distill_lr=.1,
                               distill_momentum=0., distill_ep=1, temperature=3.)
        client = SimpleNamespace(id=0, args=args, model=torch.nn.Linear(2, 2, bias=False), global_logits=None)
        expected_model = copy.deepcopy(client.model)
        x = torch.tensor([[1., 0.], [1e6, 1e6], [0., 1.], [-1e6, -1e6]])
        dataset = torch.utils.data.TensorDataset(x, torch.zeros(4, dtype=torch.long))
        probs = torch.tensor([[.8, .2], [.1, .9], [.3, .7], [.7, .3]])
        result = DefenseResult(probs, torch.tensor([True, False, True, False]), probs.log())
        optimizer = torch.optim.SGD(expected_model.parameters(), lr=.1)
        optimizer.zero_grad()
        loss = F.kl_div(F.log_softmax(expected_model(x[[0, 2]]) / 3, dim=1),
                        probs[[0, 2]], reduction="batchmean") * 9
        loss.backward()
        optimizer.step()
        distill_defended(client, dataset, result, 1)
        torch.testing.assert_close(client.model.weight, expected_model.weight)
        torch.testing.assert_close(client.global_logits, result.feedback_logits)
        before = client.model.weight.clone()
        with contextlib.redirect_stdout(io.StringIO()):
            distill_defended(client, dataset, DefenseResult(probs, torch.zeros(4, dtype=torch.bool), probs.log()), 2)
        torch.testing.assert_close(client.model.weight, before)

    def test_validation_and_state_invariants(self):
        for options in [{"rla": {"n_clusters": 0}}, {"rla": {"n_clusters": 1.5}},
                        {"unexpected": 1}, {"expguard": {"eta": -1}},
                        {"fedtgd": {"eps": 0}}, {"selective_fd": {"server_threshold": 3}},
                        {"selective_fd": {"unknown_option": 0}}]:
            with self.assertRaises(ValueError):
                resolve_options(options)
        controller = aggregator("ExpGuard")
        with self.assertRaises(ValueError):
            controller.aggregate([torch.zeros(1, 2)] * 3, 1, [2, 1, 0])
        with self.assertRaises(ValueError):
            controller.aggregate([torch.full((1, 2), float("nan"))] * 3, 1)
        controller = aggregator("SelectiveFD")
        aggregate(controller, torch.full((3, 2, 2), .5))
        with self.assertRaises(ValueError):
            aggregate(controller, torch.full((3, 3, 2), .5))


if __name__ == "__main__":
    unittest.main()
