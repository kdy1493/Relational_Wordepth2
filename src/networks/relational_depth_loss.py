"""relational_depth_loss.py 

Relational depth supervision loss for object-level ordering constraints.

Key behaviors (intentionally enforced):
- Representative object depth mode: ONLY 'median' or 'statistical' (mean is disabled)
- Valid depth gating: valid_min_depth < depth < valid_max_depth
- Object gating: min_valid_pixels on (mask ∧ valid_depth); relations touching invalid objects are dropped
- Relation normalization: 'behind' is converted to 'front' by swapping subject/object

This file is a cleaned-up version:
- no legacy unused helpers
- no duplicated docstrings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalDepthLoss(nn.Module):
    def __init__(
        self,
        margin_rank: float = 0.1,
        min_pixels: int = 20,
        min_valid_pixels: int | None = None,
        repr_mode: str = "median",
        valid_min_depth: float = 0.1,
        valid_max_depth: float = 10.0,
        statistical_alpha: float = 1.0,
        debug_relational: bool = False,
    ):
        """Relational loss enforcing object depth ordering.

        Args:
            margin_rank: hinge margin for ordering constraint (same units as depth_pred).
            min_pixels: legacy fallback if min_valid_pixels is None.
            min_valid_pixels: minimum count of valid pixels (mask ∧ valid_depth) per object.
            repr_mode: 'median' or 'statistical'.
                      - median: representative = median(depth in mask ∧ valid)
                      - statistical: representative = mean + alpha * std
            valid_min_depth / valid_max_depth: valid depth range used when computing representative depth.
            statistical_alpha: alpha for statistical representative.
            debug_relational: print some debug info (kept minimal).
        """
        super().__init__()
        self.margin_rank = float(margin_rank)
        self.min_pixels = int(min_pixels)
        self.min_valid_pixels = min_valid_pixels
        self.repr_mode = str(repr_mode).lower()
        if self.repr_mode not in {"median", "statistical"}:
            raise ValueError("repr_mode must be 'median' or 'statistical' (mean intentionally disabled).")

        self.valid_min_depth = float(valid_min_depth)
        self.valid_max_depth = float(valid_max_depth)
        self.statistical_alpha = float(statistical_alpha)
        self.relu = nn.ReLU()
        self.debug_relational = bool(debug_relational)

        # Last-batch stats for analysis (read-only from outside)
        # Keys: num_relations, num_satisfied, sum_violation
        self.last_stats: dict[str, float] = {
            "num_relations": 0.0,
            "num_satisfied": 0.0,
            "sum_violation": 0.0,
        }

    def forward(self, depth_pred, masks_batch, relations_batch):
        """Compute relational depth loss.

        Args:
            depth_pred: (B,1,H,W)
            masks_batch: list length B; each Tensor (N_i,Hm,Wm)
            relations_batch: list length B; each list[dict]
        """
        device = depth_pred.device
        dtype = depth_pred.dtype
        B, _, H_d, W_d = depth_pred.shape

        total_loss = depth_pred.new_tensor(0.0)
        valid_rel_count = 0
        sum_violation = depth_pred.new_tensor(0.0)
        num_satisfied = 0

        # min_valid fallback
        min_valid = self.min_valid_pixels
        if min_valid is None:
            min_valid = self.min_pixels
        min_valid = int(min_valid)

        for b in range(B):
            cur_depth = depth_pred[b, 0]  # (H,W)
            cur_masks = masks_batch[b]
            cur_rels = relations_batch[b]

            if cur_masks is None or cur_rels is None or len(cur_rels) == 0:
                continue

            if not torch.is_tensor(cur_masks):
                cur_masks = torch.as_tensor(cur_masks)
            if cur_masks.dim() == 2:
                cur_masks = cur_masks.unsqueeze(0)

            # Resize masks to depth resolution if needed
            if tuple(cur_masks.shape[-2:]) != (H_d, W_d):
                cur_masks = F.interpolate(
                    cur_masks.unsqueeze(1).float(),
                    size=(H_d, W_d),
                    mode="nearest",
                ).squeeze(1)
            cur_masks = cur_masks.to(device=device).float()

            N_obj = cur_masks.shape[0]

            # Valid depth mask
            valid_depth = (cur_depth > self.valid_min_depth) & (cur_depth < self.valid_max_depth)

            # Compute representative depth per object
            obj_depths = torch.empty((N_obj,), device=device, dtype=dtype)
            obj_valid = torch.zeros((N_obj,), device=device, dtype=torch.bool)

            for k in range(N_obj):
                mk = cur_masks[k] > 0.5
                mk_valid = mk & valid_depth
                cnt = int(mk_valid.sum().item())
                if cnt < min_valid:
                    obj_depths[k] = torch.nan
                    continue

                vals = cur_depth[mk_valid].to(dtype)

                if self.repr_mode == "median":
                    obj_depths[k] = vals.median()
                else:  # statistical
                    mu = vals.mean()
                    sigma = vals.std(unbiased=False)
                    obj_depths[k] = mu + self.statistical_alpha * sigma

                obj_valid[k] = True

            # Normalize relations: behind -> swap to front
            rels = []
            for rel in cur_rels:
                rel_type = str(rel.get("relation", "front")).lower()
                if rel_type not in {"front", "behind"}:
                    continue
                s_idx = int(rel.get("subject_idx"))
                o_idx = int(rel.get("object_idx"))
                if rel_type == "behind":
                    s_idx, o_idx = o_idx, s_idx
                if s_idx < 0 or s_idx >= N_obj or o_idx < 0 or o_idx >= N_obj:
                    continue
                if (not obj_valid[s_idx]) or (not obj_valid[o_idx]):
                    continue
                conf = float(rel.get("confidence", 1.0))
                rels.append((s_idx, o_idx, conf))

            if len(rels) == 0:
                continue

            idx_A = torch.tensor([r[0] for r in rels], device=device, dtype=torch.long)
            idx_B = torch.tensor([r[1] for r in rels], device=device, dtype=torch.long)
            confidence = torch.tensor([r[2] for r in rels], device=device, dtype=dtype)

            d_A = obj_depths[idx_A]
            d_B = obj_depths[idx_B]

            finite = torch.isfinite(d_A) & torch.isfinite(d_B)
            if not finite.any():
                continue

            d_A = d_A[finite]
            d_B = d_B[finite]
            confidence = confidence[finite]

            violation = self.relu(d_A - d_B + self.margin_rank)
            loss_vec = confidence * violation

            total_loss = total_loss + loss_vec.sum()
            valid_rel_count += int(loss_vec.numel())
            sum_violation = sum_violation + violation.sum()
            # count how many relations perfectly satisfy the margin (violation == 0)
            num_satisfied += int((violation == 0).sum().item())

            if self.debug_relational and b == 0:
                # Minimal debug sample
                print(f"[RelLoss] b={b} objs={N_obj} valid_objs={int(obj_valid.sum())} rels={len(rels)}")

        if valid_rel_count == 0:
            # Reset stats when no valid relations
            self.last_stats = {
                "num_relations": 0.0,
                "num_satisfied": 0.0,
                "sum_violation": 0.0,
            }
            return depth_pred.new_tensor(0.0)

        # Store last-batch stats (used for computing RSR/mean violation in the trainer)
        self.last_stats = {
            "num_relations": float(valid_rel_count),
            "num_satisfied": float(num_satisfied),
            "sum_violation": float(sum_violation.item()),
        }
        return total_loss / valid_rel_count

