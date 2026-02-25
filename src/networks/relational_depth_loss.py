"""
relational_depth_loss.py
Relational Depth Supervision Loss for WorDepth

Enforces depth ordering constraints based on object relations:
- 'front': subject is closer than object (depth_subject < depth_object)
- 'behind': subject is farther than object (depth_subject > depth_object)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalDepthLoss(nn.Module):
    """
    Relational depth supervision loss.

    Input format (matching the four-pipeline architecture):
        depth_pred:      (B, 1, H, W) - predicted depth map
        masks_batch:     list of length B, each item is Tensor(N_i, H_mask, W_mask)
        relations_batch: list of length B, each item is list of dict:
            {
                'subject_idx': int,
                'object_idx':  int,
                'relation':    'front' | 'behind',
                'confidence':  float (optional, default 1.0)
            }

    Purpose:
        - 'front':  depth(subject) < depth(object)
        - 'behind': depth(subject) > depth(object)
    """

    def __init__(self,
                 margin_rank=0.1,
                 min_pixels=20,
                 use_median=False,
                 debug_relational=False):
        """
        Args:
            margin_rank: margin for front/behind relations (meters)
            min_pixels:  minimum pixels in mask to consider object (ignore too small objects)
            use_median:  if True, use median depth; if False, use mean depth
            debug_relational: if True, enable debug prints
        """
        super().__init__()
        self.margin_rank = margin_rank
        self.min_pixels = min_pixels
        self.use_median = use_median
        self.relu = nn.ReLU()
        self.debug_relational = debug_relational

    def compute_object_depth(self, depth_map, mask):
        """
        Compute representative depth for an object region.
        
        Args:
            depth_map: (H, W) float tensor
            mask:      (H, W) float tensor (will be binarized at 0.5)
        
        Returns:
            avg_depth: scalar tensor or None if mask too small
        """
        # Move mask to same device as depth_map and apply binary threshold
        mask = (mask > 0.5).float().to(depth_map.device)
        num_pixels = mask.sum()

        if num_pixels.item() < self.min_pixels:
            return None

        # Extract depths in masked region
        masked_depth = depth_map * mask
        valid_depths = masked_depth[mask > 0.5]  # (num_pixels,)
        
        if valid_depths.numel() == 0:
            return None
        
        # Compute representative depth
        if self.use_median:
            # Median is more robust to outliers
            avg_depth = torch.median(valid_depths)
        else:
            # Mean is simpler and differentiable
            avg_depth = valid_depths.mean()
        
        return avg_depth

    def forward(self, depth_pred, masks_batch, relations_batch):
        """
        Optimized: Compute relational depth loss with full vectorization (no for-loop over relations).
        Args:
            depth_pred:      (B, 1, H, W) predicted depth
            masks_batch:     list of length B, each Tensor(N_i, H, W)
            relations_batch: list of length B, each list[dict]
        Returns:
            loss: scalar tensor (0 if no valid relations)
        """
        import torch.nn.functional as F
        device = depth_pred.device
        B, _, H_d, W_d = depth_pred.shape

        total_loss = torch.tensor(0.0, device=device, dtype=depth_pred.dtype)
        valid_rel_count = 0

        for b in range(B):
            cur_depth = depth_pred[b, 0]             # (H_d, W_d)
            cur_masks = masks_batch[b]              # Tensor(N_i, H_m, W_m)
            cur_rels  = relations_batch[b]          # list[dict]

            # Skip if no masks or relations
            if cur_masks is None or cur_masks.numel() == 0 or len(cur_rels) == 0:
                continue

            # Resize masks to match depth resolution

            if cur_masks.shape[-2:] != (H_d, W_d):
                cur_masks = F.interpolate(
                    cur_masks.unsqueeze(1).float(),   # (N, 1, H_m, W_m)
                    size=(H_d, W_d),
                    mode='nearest'
                ).squeeze(1)                           # (N, H_d, W_d)
            else:
                cur_masks = cur_masks.float()

            # 항상 cur_depth와 동일한 디바이스로 이동
            cur_masks = cur_masks.to(cur_depth.device)

            N_obj = cur_masks.shape[0]
            # 1. 객체별 대표 깊이 (mean)
            mask_sum = cur_masks.sum(dim=(1,2)) + 1e-8  # (N_obj,)
            obj_depths = (cur_depth * cur_masks).sum(dim=(1,2)) / mask_sum  # (N_obj,)

            # 2. 관계 정보 벡터화
            rels = []
            for rel in cur_rels:
                rel_type = rel.get('relation', 'front').lower()
                if rel_type not in {'front', 'behind'}:
                    continue
                # 'behind'는 subject/object swap
                if rel_type == 'behind':
                    rel = rel.copy()
                    rel['subject_idx'], rel['object_idx'] = rel['object_idx'], rel['subject_idx']
                    rel['relation'] = 'front'
                rels.append(rel)
            if len(rels) == 0:
                continue

            idx_A = torch.tensor([r['subject_idx'] for r in rels], device=device, dtype=torch.long)
            idx_B = torch.tensor([r['object_idx'] for r in rels], device=device, dtype=torch.long)
            confidence = torch.tensor([float(r.get('confidence', 1.0)) for r in rels], device=device, dtype=depth_pred.dtype)

            # 3. 관계별 d_A, d_B
            d_A = obj_depths[idx_A]  # (n_rel,)
            d_B = obj_depths[idx_B]  # (n_rel,)

            # 4. margin (front/behind 동일)
            margin = self.margin_rank
            coeff = 1.0

            # 5. violation 및 loss
            violation = self.relu(d_A - d_B + margin)
            loss_vec = coeff * confidence * violation
            total_loss = total_loss + loss_vec.sum()
            valid_rel_count += loss_vec.numel()

        if valid_rel_count == 0:
            return depth_pred.new_tensor(0.0)
        return total_loss / valid_rel_count


class CombinedDepthLoss(nn.Module):
    """
    Combines SILog loss with Relational loss.
    
    Useful wrapper for easy integration into existing training code.
    """
    
    def __init__(self, 
                 si_loss_fn,
                 relational_loss_fn,
                 weight_relational=0.1):
        """
        Args:
            si_loss_fn: SILogLoss instance
            relational_loss_fn: RelationalDepthLoss instance
            weight_relational: weight for relational loss term
        """
        super().__init__()
        self.si_loss = si_loss_fn
        self.relational_loss = relational_loss_fn
        self.weight_relational = weight_relational
    
    def forward(self, depth_pred, depth_gt, masks=None, relations=None):
        """
        Args:
            depth_pred: (B, 1, H, W)
            depth_gt:   (B, 1, H, W)
            masks:      list[Tensor] or None
            relations:  list[list[dict]] or None
        
        Returns:
            total_loss: combined loss
            loss_dict:  dict with individual loss components
        """
        # Standard depth loss
        si_loss = self.si_loss(depth_pred, depth_gt)
        
        # Relational loss (only if masks and relations provided)
        if masks is not None and relations is not None:
            rel_loss = self.relational_loss(depth_pred, masks, relations)
        else:
            rel_loss = torch.tensor(0.0, device=depth_pred.device)
        
        # Combined loss
        total_loss = si_loss + self.weight_relational * rel_loss
        
        # Return breakdown for logging
        loss_dict = {
            'si_loss': si_loss.item(),
            'relational_loss': rel_loss.item() if isinstance(rel_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
