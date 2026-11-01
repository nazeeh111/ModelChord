# Independent v0.2 scientific and implementation review

Reviewed 2026-10-29 from the original objective, current source, registered protocol, locked artifacts, source ledgers and recorded outcomes. This review did not train or optimize models, rerun confirmation or transfer inference, change a bound file, or publish anything. Its numerical checks use existing metrics, synthetic fixtures and weight differences only. Product integration and public release are separate gates below.

## Decision

The selected vision-seed11 repair supports a **bounded empirical retention success** under the registered required contract. Its selection gate passed before confirmation. All four required confirmation floors and both clean-regression bounds pass. This supports neither general repair superiority nor a successful text-transfer claim. The overall original objective must remain qualified until the current-artifact deployment, installation, interface and release gates have executed.

No acceptance-decision, source-overlap, rank-ceiling or finite-run access-counter error was found in the reviewed frozen suite. The recorded outcomes are internally consistent. The study's main scientific limits are consequential and must accompany the result.

## Exact acceptance and comparative evidence

All rows below use the same 3,823 vision sources or 1,760 eligible YouTube comments per task. Four transformed views are correlated views of those sources, not independent samples.

| Candidate | Clean | Left | Right | Inverted | Acceptance |
| --- | ---: | ---: | ---: | ---: | --- |
| Vision11 repair | 3483/3823 = 91.11% | 3155/3823 = 82.53% | 3192/3823 = 83.49% | 3250/3823 = 85.01% | Accepted required contract |
| Vision11 joint | 90.27% | 86.74% | 88.20% | 91.55% | Accepted required contract |
| Vision11 fixed factors | 90.58% | 84.12% | 87.78% | 88.39% | Accepted required contract |
| Vision29 repair | 90.32% | 81.17% | 85.87% | 84.91% | Rejected: planning selection failed |

Vision11 repair meets 4/4 required floors, zero worst required deficit, and 3/4 strengthened planning floors on confirmation. Its clean accuracy is 7.79 points above its RegMean parent and 2.64 points below its base, satisfying the registered 2-point parent and 10-point base regression limits. The 3-point planning buffer was a selection rule, not an added confirmation acceptance condition. The failed left planning margin must remain visible.

Joint and fixed-factor controls are better on all three transformed vision11 tasks, while repair is slightly better on clean. They also satisfy the required contract. This is a quality tradeoff, not a general repair win or proof that targeted repair was necessary. Joint is stronger than repair on every vision29 confirmation task. Vision29 repair and fixed factors remain unaccepted because their pre-confirmation strengthened selection gates failed, even though required confirmation floors pass.

Vision11 fixed-rank and no-protection ablations have exactly the same weight hash as the main repair. Their identical metrics are real ties, not evidence that those mechanisms improved this run. The uniform-loss adaptive ablation passes the required contract but trades task accuracies against the main repair. Priority has no established general advantage from this single reused initialization.

Text repair passes the old SMS development selection gate but fails all four transferred floors: clean 1042/1760 = 59.20%, uppercase 1064/1760 = 60.45%, leet 1126/1760 = 63.98%, underscores 1023/1760 = 58.13%. Its worst required deficit is 15.80 points. Joint and fixed-factor controls also fail all four floors, but each exceeds repair on all four pooled task accuracies. All five original video subsets are reported; none may be selected as a favorable substitute. This is negative distribution-shifted YouTube transfer. There is no fresh SMS confirmation and no accepted text deployment under the transfer contract.

I independently recomputed exact count ratios, required-floor fractions, deficits, clean regressions and acceptance booleans for every recorded participant. All agree with `confirmation-results.json`. No confirmation inference was repeated.

## Immutability and source separation

- Rehashed all 272 paths in `confirmation-lock.json`: zero mismatches. Independently verified the result/lock hashes against `post-confirmation-integrity.json`.
- Checked all 327 original v0.1 bound paths. All 326 ordinary paths remain byte-identical. The original initializer snapshot matches the original lock and commit `36adfda70544d7af2ff9ffe22f70cd4a8fc48fa8`; the current initializer is exactly that snapshot with the declared version string replacement. No old model, recipe or numerical bytes changed. The local checkout lacks a `v0.1.0` tag, so local tag verification was unavailable; remote tag/release preservation remains a publication check.
- Current `src/modelchord/repair_v2.py` is byte-identical to frozen `experiments/v02/repair_search.py`. Final main registrations bind their source snapshots, arrays, reports and weights. The superseded main-v1 attempt and development pilots remain retained, rather than disappearing after the exact-count amendment.
- The vision cohort is the original optical TRAIN file and the development corpus is the original TEST file. The retained official names file documents the 30 versus 13 different writers. This is reversed benchmark direction, with no individual writer IDs. The recorded complete four-view collision audit found zero overlap or distinct-source transformed duplicates. It supports new source examples and the documented writer split, not a new task family or unseen expert/seed test.
- Independently regenerated all 16 text calibration/selection arrays using `sms_data`, `split_sources(..., 20261030)` and `text_batches`. They exactly match `text-development.npz`; the complete old source-ID ledger also matches. This check parsed sources without model inference. A reusable source-only preparation/audit command should accompany reproduction, since the runner currently only consumes the binary file.
- Text exact deduplication excludes repeats and conflicts, and compares against all 1,306 used SMS records. It does not prove absence of near duplicates, shared templates or overlap with unused SMS ham. The independent collection has different topic and spam semantics.

Hash locks establish the identity of the current evidence, not independently authenticated wall-clock proof that no human saw a cohort earlier. The retained freeze/start/result sequence and source structure are consistent with the declared one-shot protocol. No contradictory evidence was found.

## Method, budgets and accounting

Factor search trains only a newly added `B @ A`; all preceding weights and biases are frozen. Orthogonal random A with zero B gives a zero initial correction with a learnable B gradient. The acceptance objective is lexicographic worst nonnegative accuracy shortfall, sum of shortfalls, then mean cross entropy. Detached calibration accuracy deficits weight a differentiable cross-entropy surrogate. The protected reference remains the original parent, rather than resetting at each accepted stage. Full trajectories may cross temporary losses; only their final selection checkpoints can be accepted. This is labeled data-assisted training and local intervention search, not training-free merging or a global causal explanation.

| Suite | Repair updates | Labeled training accesses | Selection accesses | Repair trials | Accepted rank upper bound |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vision11 | 540 | 386640 | 21600 | 18 | 6 |
| Vision29 | 720 | 515520 | 28080 | 24 | 5 |
| Text43 | 360 | 187200 | 10976 | 12 | 4 |

For each suite, both references exactly match those update, training-access and selection-access totals. All finite trajectories complete; no failures were silently omitted. Six reference trajectories each have one checkpoint per matched stage; their `trial_count` is checkpoint count, not independent-trajectory count. Training-access counts are repeated labeled view accesses, not unique examples. Selection totals include initial and final evaluations. Every repair stage evaluates all six layer/rank proposals; accumulated selected ranks remain within eight. Zero teacher calls are recorded for these new searches.

The controls use competent predeclared Adam learning-rate grids from the same selected ordinary parent and the same permitted inputs and fixed protection gate. The fixed all-layer control has ranks (3,3,2), with a disclosed different parameterization and overall correction bound. The cleaner priority comparison is the uniform-loss adaptive intervention search. The old TSV spectral and RegMean data-informed methods actually executed and their fixed selected artifacts are evaluated here. Their v1 tuning history is retained; they were not newly retuned on confirmation.

Equal accesses and update/checkpoint counts do not establish equal FLOPs or equal total build work. The reports' search timers exclude initial model construction/evaluation, data preparation and final export. The recorded RSS values are cumulative process high-water marks across a sequential suite, not isolated per-method peaks. RegMean's inherited three parent activation forwards / 537 calibration accesses per vision parent and expert training are separate shared costs. No exact-compute efficiency, isolated memory superiority or total-build speed advantage is established. New confirmation accounting records 469204 task, 18812 reference-clean and 56320 per-video accesses; per-video text reporting intentionally repeats inference.

Selected dense vision weights occupy 52856 bytes and its complete frozen artifact directory 70923 bytes. Selected dense text weights occupy 74840 bytes and its artifact directory 88377 bytes. These include the classifier weights and heads; low-rank repair does not shrink the deployed dense architecture. Retained parents/adapters and source/report bundles are additional repository/release storage. A deployment-only byte number must be labeled accordingly.

Weight-only checks found unchanged biases and additive differences consistent with the registered correction-rank upper bounds at singular-value tolerance 1e-5: vision11 layers (4,2,0), vision29 (4,1,0), text (4,0,0). This tolerance check is supplementary to construction/trace accounting, not a claim of exact mathematical rank for floating-point dense differences. Synthetic factor/folded-logit checks cover the factorization behavior.

Ran `.venv/bin/python -m pytest -q -p no:cacheprovider experiments/v02/test_search.py`: **19 passed**. These tests exercise zero effect/learnable gradients, folding, projection, rollback, fixed reference protection, exact floor boundaries, budget/access checks, reference snapshot isolation, invalid inputs and ablations. They do not establish empirical efficacy or adversarial ML robustness.

## Remaining delivery gates at this review snapshot

The parent task owns integration and release. Existing v1 independent-install, inference, timing, interface and demo evidence cannot be relabeled as measurements of the new v2 artifacts. Before reporting a completed v2 release, retain and inspect:

1. Fresh wheel/clean-install changed-input execution of the two fixed v2 deployment artifacts with parents, caches and network unavailable, plus the documented independent applications. The negative text transfer status must stay visible even though its static software integration can run.
2. Actual current-artifact forward and integrated application latency, with warmup, repeat count, thread/device and preprocessing scope. The frozen confirmation timer is a whole-cohort evaluator time, not single-request inference latency. Use scoped memory labels or fresh process measurements if making per-method comparisons.
3. Current CLI/API failure tests and accessible workbench checks against the actual integrated source; a fresh demo and owner walkthrough must show the v2 contract and honest negative cases. README/results/launch text and task state currently retain v1-era conclusions and need explicitly versioned v2 reporting.
4. Clean clone installation, remote CI, public source/release assets, downloads, topics and links; preservation of the v1 history/tag/release. No such remote completion was inferred from local source or reports.

Scientific scope remains deliberately small: artificial digit transformations, two reused expert initializations, one historical source collection, small repeated selection sets, no held-out task combinations or fresh expert seeds, no confidence/significance or population-level floor guarantee, and a character-feature classifier adapter integration rather than LLM composition. Point-estimate contract attainment on the locked vision cohort is the supported positive result. Future numerical or recipe tuning requires genuinely untouched confirmation evidence.

## Subsequent factual closure

The public v0.1.0 tag is now present locally and resolves to `36adfda70544d7af2ff9ffe22f70cd4a8fc48fa8`, matching the original initializer snapshot and historical lock. `scripts/prepare_v2_development.py` now retains the source-only text array preparation/check command. `scripts/reproduce_v2.py` provides the fresh-workspace replay route; `reproduction-check.json` records all 12 main weight files reproducing exactly, with official download hashes checked and no new confirmation replay.

`scripts/measure_v2_build.py` now supplies three isolated fixed-recipe vision11 repeats. I read the source and `build-profile.json`: each method executes in its own process and requires its final weight hash to match the frozen candidate. Whole-call times are repair 8.91 s, joint 16.77 s and fixed factors 17.61 s; process peaks are 273940480, 253444096 and 228442112 bytes respectively. Scope includes data/parent loading, initial selection, complete search and static export, excludes expert training/inherited RegMean construction, and includes imports in process RSS. Separate cold-process timers include imports. These single noisy repeats close the absence of isolated build-memory observations, but establish no general speed or memory superiority. Current-artifact inference latency and independent-install/remote delivery checks remain separately owned gates until their evidence exists.

`runtime-profile.json` now supplies the fixed artifacts' actual batch-one runtime distributions (20 warmups, 200 repeats, CPU two intra-operation / one inter-operation threads, warm file cache). Vision loaded logits median/p95 are 0.0411/0.0566 ms; safe integrated preprocessing/open/validation/loading/probability/label median/p95 are 31.4114/113.4156 ms. Text corresponding values are 0.0284/0.2241 ms and 21.2254/32.1559 ms. These exclude process/import/CLI startup and HTTP transport. I checked the stored scopes and artifact identities. Background activity was uncontrolled; hardware-model query was denied and no hardware identity is inferred. Runtime RSS includes both artifacts and all timing blocks. These observations close the absence of current-artifact timing evidence, not independent-install or remote publication gates.
