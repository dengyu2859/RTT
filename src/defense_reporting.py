"""Post-round diagnostics only: ground truth never enters a defense's decisions.

Read already-computed client masks, weights and public-sample acceptance masks.
Do not rerun inference, trigger selection, aggregation, training or parameter updates.
"""

import torch


def _rate(numerator, denominator):
    return (f"{100 * numerator / denominator:.2f}% ({numerator}/{denominator})"
            if denominator else "N/A (0/0)")


class DefenseReporter:
    """One experiment's read-only diagnostics and cumulative decision counts."""

    METHODS = {"FedTGD", "RLA", "ExpGuard", "SelectiveFD"}

    def __init__(self, args, client_ids, natural_trigger_indices=None, public_sample_count=None):
        self.method = "SelectiveFD" if args.aggregation == "Selective-FD" else args.aggregation
        self.client_ids = list(client_ids)
        # Match actually instantiated malicious clients, including clean-baseline runs.
        self.malicious_ids = (set(getattr(args, "malicious", [])) & set(self.client_ids)
                              if getattr(args, "attack", False) else set())
        self.public_sample_count = public_sample_count
        self.natural_mask = None
        if self.method == "SelectiveFD" and natural_trigger_indices is not None:
            if type(public_sample_count) is not int or public_sample_count < 0:
                raise ValueError("Diagnostic public sample count must be a nonnegative integer")
            self.natural_mask = torch.zeros(public_sample_count, dtype=torch.bool)
            for index in natural_trigger_indices:
                index = int(index)
                if not 0 <= index < public_sample_count:
                    raise ValueError("Diagnostic trigger index lies outside the public dataset")
                self.natural_mask[index] = True
        self.rounds = 0
        self.last_epoch = 0
        self.totals = {name: 0 for name in (
            "tp", "fn", "fp", "tn", "natural_filtered", "natural_count",
            "nontrigger_filtered", "nontrigger_count")}

    @torch.no_grad()
    def report_round(self, epoch, defense, result, final=False):
        if self.method not in self.METHODS:
            return
        if defense.method != self.method or list(defense.client_ids) != self.client_ids:
            raise ValueError("Diagnostic method/client order differs from aggregation")
        if epoch <= self.last_epoch:
            raise ValueError("Defense diagnostics require increasing, nonduplicate rounds")
        self.rounds += 1
        self.last_epoch = epoch
        prefix = f"Defense Diagnostics | Epoch {epoch} | method: {self.method}"

        if self.method == "FedTGD":
            keep = defense.fedtgd_client_mask.detach().cpu().bool().clone()
            if keep.shape != (len(self.client_ids),):
                raise ValueError("Diagnostic FedTGD mask has the wrong client count")
            filtered = {cid for cid, retained in zip(self.client_ids, keep.tolist()) if not retained}
            benign = set(self.client_ids) - self.malicious_ids
            counts = dict(tp=len(filtered & self.malicious_ids),
                          fn=len(self.malicious_ids - filtered),
                          fp=len(filtered & benign), tn=len(benign - filtered))
            for name, value in counts.items():
                self.totals[name] += value
            print(f"{prefix} | filtered_client_ids: {sorted(filtered)} | "
                  f"TPR: {_rate(counts['tp'], counts['tp'] + counts['fn'])} | "
                  f"FPR: {_rate(counts['fp'], counts['fp'] + counts['tn'])}")
            if final:
                total = self.totals
                print(f"Defense Diagnostics Total | Epoch {epoch} | method: FedTGD | rounds: {self.rounds} | "
                      f"TPR: {_rate(total['tp'], total['tp'] + total['fn'])} | "
                      f"FPR: {_rate(total['fp'], total['fp'] + total['tn'])}")

        elif self.method in {"RLA", "ExpGuard"}:
            values = defense.rla_weights if self.method == "RLA" else defense.expguard_weights
            weights = values.detach().cpu().double().clone()
            if weights.shape != (len(self.client_ids),) or not torch.isfinite(weights).all():
                raise ValueError("Diagnostic client weights have an invalid shape/value")
            description = ", ".join(f"{cid}={weight:.8g}" for cid, weight in zip(self.client_ids, weights.tolist()))
            print(f"{prefix} | client_weights: [{description}]")

        else:  # SelectiveFD / Selective-FD: sample decisions on the public distillation dataset.
            accepted = result.valid_samples.detach().cpu().bool().clone()
            if accepted.shape != (self.public_sample_count,):
                raise ValueError("Diagnostic SelectiveFD mask has the wrong public sample count")
            if self.natural_mask is None:
                print(f"{prefix} | TFR: N/A | NFR: N/A | trigger_reference: unavailable")
            else:
                filtered = ~accepted
                counts = dict(
                    natural_filtered=int((filtered & self.natural_mask).sum()),
                    natural_count=int(self.natural_mask.sum()),
                    nontrigger_filtered=int((filtered & ~self.natural_mask).sum()),
                    nontrigger_count=int((~self.natural_mask).sum()),
                )
                for name, value in counts.items():
                    self.totals[name] += value
                print(f"{prefix} | TFR: {_rate(counts['natural_filtered'], counts['natural_count'])} | "
                      f"NFR: {_rate(counts['nontrigger_filtered'], counts['nontrigger_count'])} | "
                      f"trigger_reference: public_distill")
            if final:
                if self.natural_mask is None:
                    print(f"Defense Diagnostics Total | Epoch {epoch} | method: SelectiveFD | "
                          f"rounds: {self.rounds} | TFR: N/A | NFR: N/A | trigger_reference: unavailable")
                else:
                    total = self.totals
                    print(f"Defense Diagnostics Total | Epoch {epoch} | method: SelectiveFD | "
                          f"rounds: {self.rounds} | "
                          f"TFR: {_rate(total['natural_filtered'], total['natural_count'])} | "
                          f"NFR: {_rate(total['nontrigger_filtered'], total['nontrigger_count'])}")
