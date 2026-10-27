# Third-party model – rights and provenance record

Recorded 2026-10-27. To be completed and reviewed by the project lead before the model is used in the product.

| Item | Value |
|---|---|
| Repository | https://github.com/SlockAhuja/rsna-knee-mri-model |
| Commit used | `a254152c1e7cf4aa1d7391cb9260d54a4185118c` |
| Checkpoint | `best_model.pth`, 152,040,442 bytes |
| SHA-256 | `68bd5d3fd040fa6e4c464d99d2c7dcacaebd2a40388ec0902133d4e26be0f791` |
| Download source | Git LFS object of the commit above (media.githubusercontent.com) |
| Code licence | MIT (copy kept at `third_party/rsna_knee_kaggle_model/LICENSE`) |
| Does the licence explicitly cover the weights? | **Unclear** – the MIT notice is repository-wide; no separate statement for the checkpoint. [TO CONFIRM] |
| Training data | Stated by the author as the "RSNA Knee Abnormality Detection" challenge data. Competition/data terms: **not yet reviewed**. [TO CONFIRM] |
| Institutional status | Individual third-party repository; not an official RSNA, Stanford or MIT release |
| Training metadata inside checkpoint | `epoch: 1`, `best_epoch: 1`, `best_metric: 0.5` (config planned 25 epochs) – indicates minimal training |
| Loading safety | Upstream code uses `torch.load(..., weights_only=False)`; this project loads with `weights_only=True` – passed |

## Conclusion so far

Usable for an **internal technical demonstration of the pipeline**. Not suitable for incorporation in the
product until (a) rights to the weights and training data are confirmed for commercial use and (b) a model with
documented training and validation is available. Escalated to: [project lead], [date].
