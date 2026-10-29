# Independent core review

Scope: `src/modelchord/{artifacts,models,merge,repair,evaluate}.py`, checked against the original objective, 2026-10-29. Final execution: `PYTHONPATH=src .venv/bin/python -m pytest tests/test_core.py -q`, **40 passed in 20.89 seconds**. This core review is not a whole-project release sign-off. Tests do not establish meaningful expert gains, holdout performance, independent installation, UI quality or public release.

## Findings and disposition

1. **Unverified actual base identity.** Initial compatibility compared manifest strings without verifying the actual base state. Altered weights could claim the original base ID and produce incorrect task deltas. Builder added actual `state_digest(base.state())` verification. Regression test passes.
2. **Post-open mutation.** Initial streamed reads trusted subsequently replaced bytes. Builder added file-digest and canonical-manifest verification per read. Tests reject valid changed tensors and changed provenance. This is not proof against malicious concurrent file-replacement races or signed authenticity.
3. **Overlapping calibration bins.** The initial ECE implementation counted confidence 0.5 twice, reporting 1.0 instead of 0.5 on a zero-logit control. Builder changed to disjoint integer bin IDs. Regression test passes.
4. **Exact calibration/selection reuse.** Builder added batch hashes and rejects identical batches across splits. Regression test passes. Reordered or transformed views of the same original still require the caller's source-ID ledger; that limit is recorded by the API.
5. **Numeric overflow.** Observed finite parameters `1e38` plus allowed input `1e6` export and return `[inf, inf]` from predict. Builder added nonfinite-logit rejection to inference and classification. Regression test passes for both paths.
6. **License obligations.** Initial compatibility allowed differing source licenses while export declared only the base license and omitted expert attribution/provenance. Builder now rejects differing source-license policy and retains source manifest hashes, licenses and provenance in the merge recipe. Narrow-policy regression test passes. This is not legal analysis of arbitrary license combinations.
7. **Evaluation counts.** Initial repair counts omitted the final selection pass. Builder now counts it and reports calibration metric, gradient and total evaluated examples separately. The evaluator-call-count regression test verifies the final selection pass is included.

## Coverage

Tests reject matching-shape artifacts with different tokenizer, preprocessing, representation, task family or class order; unsupported architecture, ties, precision or license; corrupt bytes and nonfinite tensors. They verify atomic export failure, immutable destinations and unchanged source hashes. Algebra controls cover exact task arithmetic, averaging, full-density DARE, TIES sign conflict, orthogonal TSV subspaces and equal/singular-statistics RegMean.

Repair tests use deterministic linear classifiers with actual gradient/SVD corrections. An achievable floor is met with rank one. A conflicting protected task rejects damaging edits and retains original weights. Disabling protection permits the edit and records that ablation. Zero rank and already-acceptable ordinary models return truthful failure/fast-path status without corrections. Candidate and rank limits are checked.

## Limits

Only registered MLP architecture and float32 tensors are supported. Expert ancestry declarations remain trusted metadata bound to hashes; arbitrary training history cannot be inferred from weights alone. Full-file hashing per tensor read adds cost and does not establish foundation-model scalability. Output models have fixed class lists. Training provenance, expert gains, confirmation isolation and recipe licenses need study/release inspection.

Reviewed source SHA-256 identities after final passing execution:

```text
artifacts.py 908faf46c7e982118746011a637d3605d6dbac24fc27ac34d1a74d28cf418826
models.py 1ca814e55883b0c38a69c937a635053f2e53bf1c4070e02a963a69974728f8e6
evaluate.py c0ae7ac3104c35cfc84b90f9e91e623cb82b96df52a73402b0307575a8f70c7e
merge.py 5bf66f243242bce1025740b4dd105f254800198977be48663288b76bc5cf18bb
repair.py 307c52fe607c6e2864c01f9c03f3833fbf94b6f0d8e4e0b47a2002368c77922b
```
