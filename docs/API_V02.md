# v0.2 API and CLI

v0.2 adds trained-factor search in `modelchord.repair_v2`; the old `modelchord.repair` API remains available. The supported registry is a bounded float32 three-layer ReLU classifier with explicit input representation and ordered classes. Unsupported architectures, tied tensors, incompatible ancestry, precision, labels, preprocessing, tokenizer or declared license policy reject. Base identity declarations are producer attestations; the supplied base's actual tensor identity is checked. [Original contract](API.md).

## Static inference

```python
import torch
from modelchord import Artifact

artifact = Artifact.open("experiments/v02/main/vision-seed11-repair/model")
x = torch.stack([torch.zeros(64), torch.ones(64)])
logits = artifact.predict(x)
print(artifact.manifest["architecture"]["classes"])
print(logits.softmax(1))
```

Vision input is `[batch,64]` float32, flattened 8×8 row-major pixels divided by 16. Text uses the versioned `examples/text_features.py`, 256 signed byte features and classes `[ham, spam]`. `Artifact.open` validates the content-bound manifest and safe tensors; `predict` rejects malformed, empty, nonfinite and overflowing inputs/outputs. It needs only the exported artifact, without parent models, routing or remote code.

## Trained-factor repair

```python
from modelchord import Artifact, Capability
from modelchord.repair_v2 import SearchBudget, search, reference_search

ordinary = Artifact.open("artifacts/vision-seed11-v2/regmean-0")
# Calibration and selection map each name to distinct (float32_x, long_y).
# The API receives selection targets directly; add the planning buffer here.
caps = [Capability("clean", .91, protected=True, max_regression=.02),
        Capability("left", .83), Capability("right", .83),
        Capability("inverted", .83)]
final, report = search(ordinary, calibration, selection, caps,
    "exports/new-v2-repair", budget=SearchBudget(rank=8, stages=4,
                                                steps_per_trial=30))
print(report["status"], report["final"])
```

`calibration` and `selection` are caller-supplied dictionaries with one CPU tensor pair per named capability. Each batch has 1–8192 examples; x is float32 `[N,input_dim]`, y is int64 `[N]` with valid class indices. Names must be unique and match both dictionaries. Exact batch hashes are bound into the report and identical calibration/selection batches reject. Source IDs, transformed-view grouping, semantic duplicate separation and permission to use examples remain the caller's responsibility.

`SearchBudget` defaults are rank 8, four stages, 30 steps per trial, Adam learning rate .02, block norm bound 4, deficit temperature .08 and initialization seed 20261030. The API accepts rank 0–64, stages 0–8, steps 1–256, and requires `rank >= 2 * stages`. Set both rank and stages to zero for the no-training path. Rank counts the sum of accepted block ranks, an upper bound on the additive correction, not the rank or size of deployed dense weights.

Each stage trains six fresh layer/rank interventions with frozen existing weights and biases. Only final disjoint-selection checkpoints are eligible; temporary internal losses can cross protected margins. Effective protected floors use the fixed original parent accuracy minus the allowed regression. The objective orders worst nonnegative accuracy deficit, sum of deficits, then mean cross entropy. The best feasible improving proposal is folded into dense weights. Rejected candidates preserve the prior accepted model. Stop at feasible selection targets or the bounded search frontier. An unsuccessful search still exports its best retained artifact and reports `no acceptable candidate found`; it is not an impossibility proof.

Use fresh output directories. Exports do not overwrite existing artifacts. Source artifacts remain unchanged. Reports include final/initial metrics, effective selection floors, every trial, rank, optimizer steps, training/selection accesses, teacher calls, exact data hashes and helper timing. `acceptable candidate found` is a development-selection status. Independent locked confirmation is a separate acceptance gate.

`mode="uniform_factor"` removes deficit weighting; `protect=False` and `fixed_rank=True` are explicit ablations. `reference_search(..., mode="joint", stages=3)` trains six full-weight trajectories; `mode="fixed_factor"` uses fixed ranks (3,3,2). These are separately labeled references, not hidden full-weight training inside repair. Match their stage count to the repair's executed stages for access-count comparisons. Matching does not imply equal FLOPs.

v2 uses the fixed classifier evaluator with integer correct counts, cross entropy, ten-bin expected calibration error and examples. The configurable [Evaluator protocol](API.md) and [custom evaluator example](../examples/custom_evaluator.py) remain in the v1 `repair(..., evaluator=...)` API. v2 `search` does not accept that hook; adding a different metric requires a separately specified method and new confirmation evidence.

## CLI

Create `capabilities.json` with requested floors, before the buffer:

```json
[
  {"name":"clean","floor":0.88,"protected":true,"max_regression":0.02},
  {"name":"left","floor":0.80},
  {"name":"right","floor":0.80},
  {"name":"inverted","floor":0.80}
]
```

```bash
modelchord --version
modelchord inspect experiments/v02/main/vision-seed11-repair/model
modelchord check artifacts/vision-seed11-v2/base artifacts/vision-seed11-v2/expert-left
modelchord repair-v2 artifacts/vision-seed11-v2/regmean-0 --data study/workbench.npz --capabilities capabilities.json --output exports/new-v2-repair --rank 8 --stages 4 --steps 30 --planning-margin .03
modelchord predict experiments/v02/main/vision-seed11-repair/model study/demo-vision-inputs.json
modelchord serve --project . --port 8765
```

NPZ keys are `cal_x_NAME`, `cal_y_NAME`, `x_NAME`, `y_NAME`. Loading disallows pickle. The CLI adds its planning margin to requested floors, capped at 1; the API example already supplies planning targets. CLI stage count is capped at `min(requested_stages, rank//2)`, and its step limit is 1–64. A zero effective stage count disables the planning buffer. Do not add the margin twice. Output JSON and the exported `build-report.json` sidecar record requested floors, effective selection targets, applied buffer and required-floor fraction. The manifest retains the core search provenance. Exit code 2 describes validation/load errors; a negative research result is expressed in the report, not a process crash.

`modelchord merge` still exposes data-free ordinary methods; RegMean needs per-expert calibration activations and is available through the API and workbench. The local workbench is catalog-bound, loopback-only and rejects foreign Host/Origin headers and arbitrary artifact paths. It is a development interface, without public hosting or authentication.
