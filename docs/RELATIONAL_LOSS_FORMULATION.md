# Relational Depth Loss: Formulation (Paper-Style)

This document provides a formal, paper-ready formulation of the relational depth supervision loss used in this project. It can be used as a methodology subsection in a manuscript.

---

## 3.X Relational Depth Supervision

We introduce a relational depth loss that supervises monocular depth prediction using **object-level relative depth relations** (e.g., “object A is in front of object B”) instead of, or in addition to, pixel-wise depth targets. Such relations are easier to obtain from language or human annotation and provide ordinal constraints that improve depth consistency across objects.

### Problem setup

Let \(\mathcal{I}\) denote an RGB image and \(\hat{D} \in \mathbb{R}^{H \times W}\) the predicted depth map (in metric units, e.g., meters). For each image we assume:

- **Object masks**: a set of binary masks \(\{\mathcal{M}_k\}_{k=1}^{K}\), where \(\mathcal{M}_k \subseteq \Omega\) and \(\Omega = \{1,\ldots,H\}\times\{1,\ldots,W\}\); each mask indicates the support of the \(k\)-th object.
- **Relations**: a set of triplets \(\mathcal{R} = \{(a_i, b_i, w_i)\}\) where \((a_i, b_i)\) are object indices and \(w_i \in (0,1]\) is an optional confidence. The semantic meaning is “object \(a_i\) is in front of object \(b_i\)” (i.e., closer to the camera).

We only use **front** relations; “behind” relations are converted to “front” by swapping subject and object, so all constraints are expressed as “subject closer than object.”

### Representative depth per object

Depth predictions are defined on the image grid; relations are defined between objects. We therefore summarize the predicted depth over each object region by a **representative depth** \(d_k\) for object \(k\).

Let \(\mathcal{V} = \{(u,v) \in \Omega : D_{\min} < \hat{D}(u,v) < D_{\max}\}\) denote the set of pixels with valid depth in a given range \([D_{\min}, D_{\max}]\). For each object \(k\), define the valid support \(\mathcal{S}_k = \mathcal{M}_k \cap \mathcal{V}\). If \(|\mathcal{S}_k| < N_{\min}\) (a minimum number of pixels), we discard object \(k\) from all relations. Otherwise, the representative depth \(d_k\) is computed from \(\{\hat{D}(u,v) : (u,v) \in \mathcal{S}_k\}\) in one of two ways:

- **Median (default):**
  \[
  d_k = \mathrm{median}\bigl\{\hat{D}(u,v) : (u,v) \in \mathcal{S}_k\bigr\}.
  \]
  The median is robust to outliers and partial occlusions within the mask.

- **Statistical:**
  \[
  d_k = \mu_k + \alpha \sigma_k, \quad \mu_k = \frac{1}{|\mathcal{S}_k|}\sum_{(u,v)\in\mathcal{S}_k} \hat{D}(u,v), \quad \sigma_k^2 = \frac{1}{|\mathcal{S}_k|}\sum_{(u,v)\in\mathcal{S}_k} (\hat{D}(u,v) - \mu_k)^2,
  \]
  where \(\alpha \geq 0\) is a hyperparameter. This encourages the “front” object’s depth to be smaller than the “back” object’s depth with a margin that scales with uncertainty.

### Ordering constraint and loss

For a relation “\(a\) is in front of \(b\),” we require the representative depth of \(a\) to be **smaller** than that of \(b\) (smaller depth = closer). We enforce this with a hinge-style margin. Let \(m > 0\) be a margin (in the same units as depth). The **violation** for one relation \((a, b, w)\) is:

\[
v(a,b) = \max\bigl(0,\, d_a - d_b + m\bigr).
\]

So the constraint \(d_a \leq d_b - m\) is satisfied when \(v(a,b) = 0\); otherwise the loss penalizes how much \(d_a\) exceeds \(d_b - m\).

The **relational depth loss** for an image is the confidence-weighted average over all valid relations in the batch:

\[
\mathcal{L}_{\mathrm{rel}} = \frac{1}{|\mathcal{R}_{\mathrm{valid}}|} \sum_{(a,b,w) \in \mathcal{R}_{\mathrm{valid}}} w \cdot v(a,b),
\]

where \(\mathcal{R}_{\mathrm{valid}}\) is the set of relations whose subject and object both have valid representative depths (i.e., sufficient valid pixels and finite \(d_a, d_b\)). If \(\mathcal{R}_{\mathrm{valid}}\) is empty, \(\mathcal{L}_{\mathrm{rel}} = 0\).

### Total training objective

The total loss is:

\[
\mathcal{L} = \mathcal{L}_{\mathrm{depth}} + \lambda_{\mathrm{rel}} \, \mathcal{L}_{\mathrm{rel}},
\]

where \(\mathcal{L}_{\mathrm{depth}}\) is the primary depth loss (e.g., SILog or L1 on valid pixels) and \(\lambda_{\mathrm{rel}}\) is a scalar weight (e.g., 0.1). Gradients from \(\mathcal{L}_{\mathrm{rel}}\) flow through the representative depths \(d_k\) into \(\hat{D}\), so the network learns to respect the annotated ordinal relations while still fitting pixel-wise targets when available.

### Implementation details (summary)

- **Relation normalization**: “Behind” relations are converted to “front” by swapping subject and object, so only one ordering rule is implemented.
- **Valid depth range**: \([D_{\min}, D_{\max}]\) (e.g., 0.1 m and 10 m) avoids invalid or saturated depth values when computing \(d_k\).
- **Mask resolution**: If masks are at a different resolution than \(\hat{D}\), they are resized to the depth map size (e.g., nearest-neighbor) before computing \(\mathcal{S}_k\).
- **Batch**: For a batch of images, \(\mathcal{L}_{\mathrm{rel}}\) is summed over all images and divided by the total number of valid relations in the batch (average, not sum), so the scale is independent of batch size and relation count.

---

## Reference (citation-style)

If you use this formulation in a paper, you can describe it as follows:

> We supervise depth with an auxiliary relational loss. For each image we have object masks and a set of “front” relations (subject closer than object). We compute a representative depth per object (median or mean+α·std over mask ∩ valid-depth pixels) and enforce each relation with a hinge loss \(\max(0, d_s - d_o + m)\), averaged over relations and scaled by a weight λ_rel.

---

*Source: `src/networks/relational_depth_loss.py`, `src/relational_train.py`. Parameters: `rel_weight` (λ_rel), `rel_margin` (m), `rel_repr` (median | statistical), `rel_valid_min/max_depth` (D_min, D_max), `rel_min_valid_pixels` (N_min), `rel_statistical_alpha` (α).*
