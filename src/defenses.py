"""Opt-in defenses for the NBA prediction-sharing protocol.

FedTGD is an explicitly labelled PRINCIPLE ADAPTATION: client-level DBSCAN on
whole-public-dataset RMS probability distances, largest-cluster selection, then
class-aligned top-k cosine filtering over the whole public dataset. One retained
client set applies to every public sample; aggregate retained clients' raw logits.
Its full published selection rules were unavailable; it is not an exact replica.

ExpGuard implements EG+Filter: per-sample leading-eigenvector projection,
mean absolute client score over public samples, and persistent exp(-eta*score)
weights shared by every public sample from a client. The original EG+Filter
score is retained; the requested aggregation uses raw logits rather than
probability averaging. See https://github.com/ZIB-IOL/FedDistill/blob/main/byzantine/defences.py
and utilities.py. Gram matrices avoid constructing C-by-C covariance matrices.

SelectiveFD is the requested SERVER-ONLY ablation: mean all clients' probability
predictions and retain public samples with low server L1-to-one-hot ambiguity.
No client-side selection, private-data reads, density fitting, or upload masks.
This is not the complete two-selector method of https://arxiv.org/html/2304.01731v3.

RLA follows the user-specified four-step algorithm: raw classwise mean logits,
spectral clustering, largest-cluster center, and normalized nonnegative cosine
weights on all uploaded logits. Concrete conventions are in defense_notes.txt;
this is not a claim of reproducing an independently verified paper implementation.

These modules never inspect attack labels, trigger regions, poisoned indices,
malicious-client IDs, or test labels. Default settings are not ASR-tuned.
"""

import copy
from dataclasses import dataclass
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


DEFENSE_METHODS = {"RLA", "FedTGD", "ExpGuard", "SelectiveFD", "Selective-FD"}
DEFAULTS = {
    "probability_temperature": 1.0,
    "chunk_size": 256,
    "fedtgd": {"top_k": 3, "eps": 0.5, "min_samples": 5, "cosine_threshold": 0.5},
    "expguard": {"eta": 1.0},
    "rla": {"n_clusters": 2},
    "selective_fd": {"server_threshold": 1.0},
}


# Accept old saved configs without reviving the removed client-side operations.
REMOVED_SELECTIVE_OPTIONS = {
    "max_local_samples", "basis_size", "uniform_samples", "validation_fraction",
    "client_quantile", "regularization",
}


def resolve_options(options):
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("defense must be a YAML mapping")
    result = copy.deepcopy(DEFAULTS)
    for key, value in options.items():
        if key not in result:
            raise ValueError(f"Unknown defense option: {key}")
        if isinstance(result[key], dict):
            if key == "selective_fd" and isinstance(value, dict):
                value = {name: setting for name, setting in value.items()
                         if name not in REMOVED_SELECTIVE_OPTIONS}
            if not isinstance(value, dict) or set(value) - set(result[key]):
                raise ValueError(f"Invalid defense.{key} options")
            result[key].update(value)
        else:
            result[key] = value

    def number(value, name, minimum, maximum=None, strict=False, integer=False):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (integer and type(value) is not int)
                or (value <= minimum if strict else value < minimum)
                or (maximum is not None and value > maximum)):
            raise ValueError(f"Invalid defense setting {name}: {value}")

    number(result["probability_temperature"], "probability_temperature", 0, strict=True)
    number(result["chunk_size"], "chunk_size", 1, integer=True)
    number(result["rla"]["n_clusters"], "rla.n_clusters", 1, integer=True)
    f = result["fedtgd"]
    number(f["top_k"], "fedtgd.top_k", 1, integer=True)
    number(f["eps"], "fedtgd.eps", 0, strict=True)
    number(f["min_samples"], "fedtgd.min_samples", 1, integer=True)
    number(f["cosine_threshold"], "fedtgd.cosine_threshold", 0, 1)
    number(result["expguard"]["eta"], "expguard.eta", 0)
    number(result["selective_fd"]["server_threshold"], "selective_fd.server_threshold", 0, 2)
    return result


@dataclass
class DefenseResult:
    probabilities: torch.Tensor
    valid_samples: torch.Tensor
    feedback_logits: torch.Tensor


class DefenseAggregator:
    """One instance per experiment, preserving ExpGuard weights and full feedback alignment."""

    def __init__(self, args, client_ids):
        self.method = "SelectiveFD" if args.aggregation == "Selective-FD" else args.aggregation
        if self.method not in DEFENSE_METHODS:
            raise ValueError(f"Unknown defense: {self.method}")
        self.options = resolve_options(getattr(args, "defense", None))
        if self.method == "SelectiveFD":
            supplied = getattr(args, "defense", None) or {}
            ignored = sorted(set(supplied.get("selective_fd", {})) & REMOVED_SELECTIVE_OPTIONS)
            if ignored:
                print(f"SelectiveFD server-only: ignored removed client-side options {ignored}")
        args.defense = copy.deepcopy(self.options)
        args.defense_implementation = {
            "RLA": "user_spec_raw_mean_spectral_cosine_v1",
            "FedTGD": "principle_adaptation_client_filter_raw_logits_v2",
            "ExpGuard": "EG+Filter_client_history_raw_logits_v2",
            "SelectiveFD": "server_only_L1_filter_v2",
        }[self.method]
        self.client_ids = list(client_ids)
        if not self.client_ids or len(set(self.client_ids)) != len(self.client_ids):
            raise ValueError("Defense requires distinct, stable client IDs")
        self.log_weights = torch.zeros(len(self.client_ids), dtype=torch.float64)
        self.feedback_logits = None
        self.seed = int(args.seed)
        self.distill_temperature = float(getattr(args, "temperature", 1.0))
        if self.method in {"RLA", "ExpGuard", "FedTGD"} and (
                not math.isfinite(self.distill_temperature) or self.distill_temperature <= 0):
            raise ValueError(f"{self.method} requires a finite positive distillation temperature")
        print(f"Defense enabled: {self.method} | implementation: {args.defense_implementation}")
        print(f"Defense settings: {self.options}")

    def _rla(self, logits):
        from sklearn.cluster import SpectralClustering

        # One classwise mean vector per client, using every public row without labels.
        representatives = logits.mean(1)
        n = len(representatives)
        unit = F.normalize(representatives, p=2, dim=1, eps=1e-15)
        k = min(self.options["rla"]["n_clusters"], n)
        # Sort only to make clustering and size-tie decisions independent of upload order.
        order = np.argsort(self.client_ids, kind="stable")
        warning_messages = []
        if k == 1 or torch.allclose(unit, unit[:1].expand_as(unit), atol=1e-12, rtol=0):
            labels = np.zeros(n, dtype=np.int64)
        elif k == n:
            # Spectral clustering into N clusters is exactly N singleton groups.
            labels = np.empty(n, dtype=np.int64)
            labels[order] = np.arange(n)
        else:
            affinity = ((unit @ unit.T).clamp(-1, 1).numpy() + 1.0) / 2.0
            np.fill_diagonal(affinity, 1.0)
            # Separate RNG prevents defense clustering from consuming training RNG.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                sorted_labels = SpectralClustering(
                    n_clusters=k, affinity="precomputed", assign_labels="kmeans",
                    n_init=10, random_state=self.seed % (2 ** 32),
                ).fit_predict(affinity[np.ix_(order, order)])
            warning_messages = [str(item.message) for item in caught]
            labels = np.empty(n, dtype=np.int64)
            labels[order] = sorted_labels
        groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        # Equal-sized groups: choose the lexicographically smallest sorted client-ID set.
        reference = min(groups, key=lambda group: (
            -len(group), tuple(sorted(self.client_ids[i] for i in group))))
        center = representatives[reference].mean(0)
        similarity = (unit @ F.normalize(center, p=2, dim=0, eps=1e-15)).clamp(-1, 1)
        weights = similarity.clamp_min(0)
        uniform_fallback = bool(weights.sum() <= 1e-15)
        if uniform_fallback:
            weights = torch.full_like(weights, 1.0 / n)
        else:
            weights /= weights.sum()
        pooled = (logits * weights[:, None, None]).sum(0)
        # Small audit state: no public-sample copies, labels, or attack metadata.
        self.rla_representatives = representatives.clone()
        self.rla_cluster_labels = labels.copy()
        self.rla_center = center.clone()
        self.rla_weights = weights.clone()
        members = [self.client_ids[i] for i in reference]
        clusters = ", ".join(f"{cid}={int(label)}" for cid, label in zip(self.client_ids, labels))
        description = ", ".join(f"{cid}={w:.6g}" for cid, w in zip(self.client_ids, weights.tolist()))
        return pooled, (f"clusters: [{clusters}] | reference_clients: {members} | "
                        f"client_weights: [{description}] | uniform_fallback: {uniform_fallback} | "
                        f"spectral_warnings: {warning_messages}")

    def _fedtgd(self, logits):
        from sklearn.cluster import DBSCAN

        opts = self.options["fedtgd"]
        probabilities = (logits / self.options["probability_temperature"]).softmax(-1)
        n, m, classes = probabilities.shape
        chunk = self.options["chunk_size"]
        # RMS of per-public-sample Euclidean distances: no growth with public size.
        # Keep aligned sample/class coordinates rather than collapsing to class means.
        squared = np.zeros((n, n), dtype=np.float64)
        for start in range(0, m, chunk):
            flat = probabilities[:, start:start + chunk].numpy().reshape(n, -1)
            norms = (flat * flat).sum(1)
            squared += np.maximum(norms[:, None] + norms[None, :] - 2 * flat @ flat.T, 0)
        distances = np.sqrt(squared / m)
        np.fill_diagonal(distances, 0)
        # Stable IDs only resolve order/ties; no attack identity is consulted.
        order = np.argsort(self.client_ids, kind="stable")
        labels = np.empty(n, dtype=np.int64)
        labels[order] = DBSCAN(eps=opts["eps"], min_samples=opts["min_samples"],
                               metric="precomputed").fit_predict(distances[np.ix_(order, order)])
        groups = [np.flatnonzero(labels == label) for label in sorted(set(labels) - {-1})]
        all_noise = not groups
        anchor_id = None
        if all_noise:
            accepted = np.arange(n)  # Preserve the explicitly logged conservative fallback.
        else:
            candidates = min(groups, key=lambda group: (
                -len(group), tuple(sorted(self.client_ids[i] for i in group))))
            candidates = np.asarray(sorted(candidates, key=lambda i: self.client_ids[i]))
            # Whole-dataset cosine of top-k-per-row sparse vectors; class positions stay aligned.
            gram = np.zeros((len(candidates), len(candidates)), dtype=np.float64)
            k = min(opts["top_k"], classes)
            for start in range(0, m, chunk):
                q = probabilities[candidates, start:start + chunk].numpy()
                columns = np.argsort(-q, axis=2, kind="stable")[:, :, :k]
                sparse = np.zeros_like(q)
                np.put_along_axis(sparse, columns, np.take_along_axis(q, columns, axis=2), axis=2)
                flat = sparse.reshape(len(candidates), -1)
                gram += flat @ flat.T
            norms = np.sqrt(np.maximum(np.diag(gram), 0))
            similarity = np.clip(gram / np.maximum(norms[:, None] * norms[None, :], 1e-15), -1, 1)
            anchor = int(np.argmax(similarity.mean(1)))
            anchor_id = self.client_ids[candidates[anchor]]
            accepted_mask = similarity[anchor] >= opts["cosine_threshold"]
            accepted_mask[anchor] = True
            accepted = candidates[accepted_mask]
        keep = torch.zeros(n, dtype=torch.bool)
        keep[accepted] = True
        # A single client mask controls ALL public rows and classes in this round.
        pooled = logits[keep].mean(0)
        self.fedtgd_client_mask = keep.clone()
        self.fedtgd_cluster_labels = labels.copy()
        kept = sorted(self.client_ids[i] for i in accepted)
        rejected = sorted(self.client_ids[i] for i in range(n) if not keep[i])
        return pooled, (f"clients_kept: {len(kept)}/{n} | kept_client_ids: {kept} | "
                        f"rejected_client_ids: {rejected} | anchor_client: {anchor_id} | "
                        f"all_noise_fallback: {all_noise}")

    def _expguard(self, logits):
        """Score the whole public dataset, update history, weight each client's entire logits."""
        probabilities = (logits / self.options["probability_temperature"]).softmax(-1)
        n, m, _ = probabilities.shape
        score_sum = torch.zeros(n, dtype=torch.float64)
        chunk = self.options["chunk_size"]
        for start in range(0, m, chunk):
            values = probabilities[:, start:start + chunk].permute(1, 0, 2)
            centered = values - values.mean(1, keepdim=True)
            gram = centered @ centered.transpose(1, 2)
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            # If X=U S V^T, projection on leading V equals leading U multiplied by S.
            projections = eigenvectors[:, :, -1].abs() * eigenvalues[:, -1:].clamp_min(0).sqrt()
            score_sum += projections.sum(0)
        scores = score_sum / m  # Matches the authors' mean absolute projection implementation.
        self.log_weights -= self.options["expguard"]["eta"] * scores
        self.log_weights -= self.log_weights.max()  # Stable common rescaling of persistent weights.
        weights = self.log_weights.softmax(0)
        # One weight per client, broadcast over ALL samples and classes, after full scoring.
        result = (logits * weights[:, None, None]).sum(0)
        self.expguard_scores = scores.clone()
        self.expguard_weights = weights.clone()
        description = ", ".join(f"{cid}={weight:.6g}" for cid, weight in zip(self.client_ids, weights.tolist()))
        score_description = ", ".join(f"{cid}={score:.6g}" for cid, score in zip(self.client_ids, scores.tolist()))
        return result, f"client_scores: [{score_description}] | client_weights: [{description}]"

    def _selective(self, probabilities):
        # Every client contributes to every public sample before the server selector.
        pooled = probabilities.mean(0)
        ambiguity = 2 * (1 - pooled.max(1).values)
        valid = ambiguity <= self.options["selective_fd"]["server_threshold"]
        return pooled, valid, (
            f"clients_used: {len(probabilities)} | selection: server_only | "
            f"distill_samples: {int(valid.sum())}/{len(valid)}")

    @torch.no_grad()
    def aggregate(self, logits_list, epoch, client_ids=None):
        if client_ids is not None and list(client_ids) != self.client_ids:
            raise ValueError("Defense client order changed between rounds")
        if len(logits_list) != len(self.client_ids):
            raise ValueError("Defense received the wrong number of clients")
        logits = torch.stack([value.detach().cpu().double() for value in logits_list])
        if logits.ndim != 3 or logits.shape[1] == 0 or logits.shape[2] < 2:
            raise ValueError("Defense expects logits shaped [clients, public_samples, classes>=2]")
        if not torch.isfinite(logits).all():
            raise ValueError("Defense received NaN/Inf logits")
        if self.method in {"RLA", "ExpGuard", "FedTGD"}:
            # Client-level weighting/filtering of raw logits, preserving attack feedback scale.
            pooled, description = {
                "RLA": self._rla, "ExpGuard": self._expguard, "FedTGD": self._fedtgd,
            }[self.method](logits)
            if self.feedback_logits is not None and self.feedback_logits.shape != pooled.shape:
                raise ValueError("Public dataset shape changed between defense rounds")
            feedback = pooled.float()
            if not torch.isfinite(feedback).all():
                raise ValueError(f"{self.method} aggregate exceeds float32 feedback range")
            self.feedback_logits = feedback
            probabilities = (pooled / self.distill_temperature).softmax(-1).float()
            print(f"Defense | Epoch {epoch} | method: {self.method} | {description}")
            return DefenseResult(probabilities, torch.ones(logits.shape[1], dtype=torch.bool),
                                 feedback.clone())
        probabilities = (logits / self.options["probability_temperature"]).softmax(-1)
        pooled, valid, description = {
            "SelectiveFD": self._selective,
        }[self.method](probabilities)
        pooled = pooled.clamp_min(0)
        pooled /= pooled.sum(-1, keepdim=True)
        if self.feedback_logits is None:
            self.feedback_logits = torch.full_like(pooled, -math.log(pooled.shape[1]), dtype=torch.float32)
        if self.feedback_logits.shape != pooled.shape:
            raise ValueError("Public dataset shape changed between defense rounds")
        feedback = pooled.float().clamp_min(torch.finfo(torch.float32).tiny).log()
        # Preserve original public-row indexing for the existing attack's global feedback.
        # Rejected rows retain previous feedback (uniform at initialization), and do not train models.
        self.feedback_logits[valid] = feedback[valid]
        print(f"Defense | Epoch {epoch} | method: {self.method} | {description}")
        return DefenseResult(pooled.float(), valid.clone(), self.feedback_logits.clone())


def distill_defended(client, public_dataset, result, epoch):
    """Shared defense-only KD adapter. Original benign/attack source code is untouched."""
    if hasattr(client, "global_logits"):
        client.global_logits = result.feedback_logits.clone()
    indices = result.valid_samples.nonzero(as_tuple=False).flatten()
    if not len(indices):
        print(f"Client {client.id} | Epoch {epoch} | Defense distillation skipped: no accepted samples")
        return
    dataset = Subset(public_dataset, indices.tolist())
    targets = result.probabilities[indices]
    args = client.args
    loader = DataLoader(dataset, batch_size=args.local_bs, shuffle=False)
    client.model.train()
    optimizer = torch.optim.SGD(client.model.parameters(), lr=args.distill_lr,
                                momentum=args.distill_momentum)
    for _ in range(args.distill_ep):
        for batch, (inputs, _) in enumerate(loader):
            offset = batch * args.local_bs
            teacher = targets[offset:offset + len(inputs)].to(args.device, non_blocking=True)
            optimizer.zero_grad()
            student = client.model(inputs.to(args.device, non_blocking=True))
            # Targets are already probabilities: a second softmax would change the defense output.
            loss = F.kl_div(F.log_softmax(student / args.temperature, dim=1), teacher,
                            reduction="batchmean") * args.temperature ** 2
            loss.backward()
            optimizer.step()
