# v0.2 results and costs

The fixed vision11 repair meets the required contract on the locked 3,823-source optical-digit cohort. Text transfer fails. Joint and fixed-factor training are strong alternatives, so the positive vision result is retention under this contract, not repair superiority.

Evidence: [protocol](../experiments/v02/PROTOCOL.md), [pre-confirmation lock](../experiments/v02/confirmation-lock.json), [all exact outcomes](../experiments/v02/confirmation-results.json), [main recipe registry](../experiments/v02/main/registry.json), [integrity readback](../experiments/v02/post-confirmation-integrity.json) and [independent review](../experiments/v02/review/FINAL_REVIEW.md). The [v0.1 report](RESULTS.md) remains unchanged.

## Acceptance

Required vision floors are clean 88%, left/right/inverted 80%, clean regression at most 2 points from the ordinary parent and 10 points from base. Development selection adds 3 points to each requested floor. The protected-reference floor can raise selection requirements further. Selection failure cannot be reversed by a favorable confirmation score. Confirmation acceptance uses required floors and protected bounds, not the extra planning buffer.

Deployment was fixed to `main/vision-seed11-repair/model` before confirmation. It passed planning selection with clean 92.59%, left 84.44%, right 84.44% and inverted 85.19%. Confirmation passes 4/4 required floors, zero worst required deficit and both clean-regression bounds. Clean improves 7.79 points over RegMean and regresses 2.64 points from base. The strengthened planning buffer passes 3/4 confirmation tasks; left misses 83%.

| Vision11 repair task | Correct / sources | Accuracy | Cross entropy | Calibration error |
| --- | ---: | ---: | ---: | ---: |
| clean | 3483/3823 | 91.11% | 0.3154 | 0.0207 |
| left | 3155/3823 | 82.53% | 0.5293 | 0.0407 |
| right | 3192/3823 | 83.49% | 0.5410 | 0.0285 |
| inverted | 3250/3823 | 85.01% | 0.6317 | 0.1942 |

Calibration error is ten-bin expected calibration error, the weighted gap between confidence and observed accuracy. A low error does not establish correctness on a particular input. Inversion remains poorly calibrated despite meeting its accuracy floor.

## Comparisons and ablations

All accuracies below use the same full cohort, 3,823 sources per task. Exact numerators, loss, calibration, regressions and artifact identities for every method are in the machine-readable outcomes.

| Initialization / method | Clean | Left | Right | Inverted | Required floors | Planning selection |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 11 RegMean parent | 83.31% | 78.16% | 68.35% | 27.47% | 0/4 | Ordinary reference |
| 11 repair | 91.11% | 82.53% | 83.49% | 85.01% | 4/4 | Passed |
| 11 joint | 90.27% | 86.74% | 88.20% | 91.55% | 4/4 | Passed |
| 11 fixed factors | 90.58% | 84.12% | 87.78% | 88.39% | 4/4 | Passed |
| 11 uniform-loss adaptive | 90.79% | 83.15% | 86.14% | 82.19% | 4/4 | Passed |
| 11 fixed rank | 91.11% | 82.53% | 83.49% | 85.01% | 4/4 | Passed |
| 11 no protection | 91.11% | 82.53% | 83.49% | 85.01% | 4/4 | Passed |
| 29 RegMean parent | 83.73% | 66.89% | 78.00% | 61.71% | 0/4 | Ordinary reference |
| 29 repair | 90.32% | 81.17% | 85.87% | 84.91% | 4/4 | Failed; remains unaccepted |
| 29 joint | 90.43% | 87.21% | 88.78% | 90.43% | 4/4 | Passed |
| 29 fixed factors | 89.33% | 81.87% | 85.61% | 86.37% | 4/4 | Failed; remains unaccepted |

Joint and fixed-factor references exceed repair on all three transformed vision11 tasks, with slightly lower clean accuracy. Joint exceeds repair on every vision29 task. Repair is therefore unnecessary to establish that this small architecture can meet the contract. Fixed-rank and no-protection ablations have exactly the primary repair's weight hash. They establish ties here, not benefits of those mechanisms. Uniform-loss adaptive search is the cleaner priority ablation; this single run does not establish a general advantage.

Both seeds' bases, individual specialists, arithmetic, TIES/DARE-style, TSV and RegMean are also evaluated. The spectral TSV and data-informed RegMean methods actually executed in v0.1; their fixed selected artifacts are reused unchanged and evaluated on the new cohort. They were not retuned on confirmation. Their exact new metrics are retained even when poor. Ensembles and conflict/insufficient-calibration/easy controls remain in the historical study; they are not newly repeated v0.2 experiments.

## Text transfer remains negative

The text integration is a 256→64→32→2 classifier over signed byte features. Three same-base rank-4 specialist adapters target uppercase, leet and underscore domains; vision and text weights are never mixed. No balanced fresh SMS cohort remained after v0.1. YouTube comments have the same ham/spam labels but different topic and spam semantics, so this is explicitly shifted transfer, not SMS reconfirmation.

| Text43 repair task | Correct / sources | Accuracy | Cross entropy | Calibration error |
| --- | ---: | ---: | ---: | ---: |
| clean | 1042/1760 | 59.20% | 1.5081 | 0.2561 |
| uppercase | 1064/1760 | 60.45% | 1.5866 | 0.2956 |
| leet | 1126/1760 | 63.98% | 0.9124 | 0.1637 |
| underscores | 1023/1760 | 58.13% | 1.4071 | 0.2426 |

| Method | Clean | Uppercase | Leet | Underscores | Required floors | Worst deficit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Arithmetic parent | 56.02% | 58.18% | 57.39% | 56.53% | 0/4 | 18.98 points |
| Repair | 59.20% | 60.45% | 63.98% | 58.13% | 0/4 | 15.80 points |
| Joint | 64.09% | 61.70% | 68.30% | 61.59% | 0/4 | 10.91 points |
| Fixed factors | 65.97% | 62.84% | 65.23% | 60.34% | 0/4 | 11.66 points |

All three new text methods passed old SMS development selection, then failed every transferred floor (75% clean, 72% transformed). Joint and fixed factors exceed repair on all four pooled task accuracies. All five original videos' eligible subsets are reported, without favorable subgroup selection. The artifact remains runnable, but its transfer deployment is unaccepted.

## Search budget and build costs

Each repair stage runs six 30-update trajectories, one for each layer × rank {1,2}. Six joint/fixed-factor trajectories run the same executed stage count, retaining their best permitted 30-update checkpoints. All new searches have zero teacher calls. A training access is a repeated labeled transformed example, not a unique source.

| Suite | Optimizer updates | Training accesses | Selection accesses | Candidate/checkpoint evaluations | Accepted repair rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Vision11 | 540 | 386640 | 21600 | 18 | 6 |
| Vision29 | 720 | 515520 | 28080 | 24 | 5 |
| Text43 | 360 | 187200 | 10976 | 12 | 4 |

Within each suite, repair, joint and fixed-factor totals match exactly. The three vision11 ablations also match the main repair totals. Repair's count describes separate intervention trajectories; reference count describes checkpoints from six continuous trajectories. Initial/final selection evaluations are included. No nonfinite trials were omitted in these completed runs.

Matching accesses, updates and checkpoints does not match FLOPs, gradient parameter count, decomposition/projection/copying work or total expert-training cost. The inherited vision RegMean parent additionally used three parent activation forwards / 537 calibration accesses. Expert training and conventional baseline tuning are historical costs, shared by the new same-start methods rather than counted as free new capability creation.

Fresh-process fixed-recipe vision11 repeats in [build-profile.json](../experiments/v02/build-profile.json) reproduced all three frozen weights exactly:

| Method | Whole build call | Cold process incl. imports | Process peak RSS, bytes | Maximum simultaneous trainable parameters |
| --- | ---: | ---: | ---: | ---: |
| Repair | 8.91 s | 16.27 s | 273940480 | 320 |
| Joint | 16.77 s | 29.12 s | 253444096 | 13098 |
| Fixed factors | 17.61 s | 33.65 s | 228442112 | 1108 |

These single noisy repeats use CPU / two threads. Whole-call timing includes data/parent load, initial selection, full search and static export, but excludes imports, expert training and inherited RegMean construction. Cold process time adds imports. RSS includes imports and runtime overhead. The original sequential reports' RSS values are cumulative process high-water marks, not isolated build peaks. Repair has fewer trainable parameters but higher measured RSS here. No memory or general speed-efficiency claim follows. See the [measurement script](../scripts/measure_v2_build.py).

All deployed dense layers and heads count in storage. Vision11 repair has 52856 weight bytes and 70923 bytes in its complete frozen model directory; text has 74840 and 88377 respectively. Joint/fixed-factor weights have the same dense architecture size. Low-rank correction is a training constraint, not deployed-model compression. Retained bases, experts, adapters and study bundles consume additional storage.

## Static inference latency

[runtime-profile.json](../experiments/v02/runtime-profile.json) binds the two fixed artifacts and retains every timing sample. Measurements use batch one, CPU / two intra-operation threads and one inter-operation thread, 20 warmups and 200 repeats on macOS arm64, Python 3.12.13 and PyTorch 2.9.1. The hardware model query was denied; no specific CPU model is inferred. Inputs are synthetic digit pixels and an author-written scheduling sentence, with no confirmation data.

| Artifact / measured scope | Median, ms | 95th percentile, ms |
| --- | ---: | ---: |
| Vision, loaded forward / logits only | 0.0411 | 0.0566 |
| Vision, preprocessing + loaded forward + probabilities/label | 0.0495 | 0.1439 |
| Vision, safe integrated request with open/validation/loading | 31.4114 | 113.4156 |
| Text, loaded forward / logits only | 0.0284 | 0.2241 |
| Text, features + loaded forward + probabilities/label | 0.2301 | 3.3617 |
| Text, safe integrated request with open/validation/loading | 21.2254 | 32.1559 |

`Artifact.predict` rehashes and loads weights even when its Artifact handle is reused. Its safe integrated request includes preprocessing, `Artifact.open`, per-call validation/loading, inference, softmax and label selection. Forward-only values do not describe that application path. These timings exclude process/import/CLI startup and HTTP transport; operating-system file caches are warm. Blocks execute sequentially and background activity is uncontrolled, so values are noisy, non-additive and unsuitable for a general speed claim. The process peak RSS, 199868416 bytes, includes imports, both artifacts and all timing blocks; it is not an isolated per-artifact memory measurement. The [runtime script](../scripts/measure_v2_runtime.py) includes all layers and classifier heads.

## Provenance, reproduction and limits

The vision cohort contains all 3,823 original UCI TRAIN rows. The old 1,797-row development corpus is the original TEST file. Official metadata identifies different writer groups, 30 versus 13; this reverses the original benchmark direction. All four new transformed views were checked against all four old views: zero collisions and zero distinct-source transformed duplicates. Four views remain correlated, so there are 3,823 source images, not 15,292 independent observations. Text exact deduplication retains 1,760 comments (919 ham, 841 spam), with no exact match to the 1,306 used SMS records. This does not rule out shared templates or near duplicates. [Source audit](../experiments/v02/review/DATA.md).

All 272 v0.2 bound paths were independently rehashed unchanged. v0.1's 326 ordinary bound paths remain unchanged; the one version-string exception retains exact original bytes. The first main attempt and all pilots remain archived; the final main rerun used exact integer-count accuracy gates before new cohort metrics were opened. No revision after confirmation is presented as an untouched test.

[reproduction-check.json](../experiments/v02/reproduction-check.json) records a complete fresh-workspace execution with exact weight matches for all 12 main recipes and official source archive hash checks, without replaying confirmation. See [reproduction commands](REPRODUCTION_V02.md). This reproduces development optimization, not unseen research evidence. The independent synthetic suite passes 19 tests; these establish software properties rather than accuracy claims. Current release-install, UI and public CI checks are separate delivery evidence and must not be inferred from this scientific report.

The study uses artificial domains from one historical collection and two reused expert initializations. It holds out source examples, not expert seeds or task combinations. There is no external pretrained model, foundation-model transfer, statistical significance, simultaneous population-floor confidence or universal capability guarantee. Repeated selection can overfit. Functional interventions establish local measured candidate effects, not global causal interference or novelty. Failed bounded search means no acceptable candidate found under that search, not impossibility.

Attribution: Alpaydin and Kaynak (1998), [Optical Recognition of Handwritten Digits](https://doi.org/10.24432/C50P49); Almeida and Hidalgo (2011), [SMS Spam Collection](https://doi.org/10.24432/C5CC84); Alberto and Lochter (2015), [YouTube Spam Collection](https://doi.org/10.24432/C58885). UCI datasets retain [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Normalization, transformations, duplicate exclusion and reversed optical file roles are documented changes. Raw SMS and YouTube messages are not published. [Baseline/method attribution](ATTRIBUTION.md).
