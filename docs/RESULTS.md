# Measured results

One locked confirmation run, 29 October 2026. The model and recipe bytes did not change after confirmation. [Raw metrics](../study/results.json), [frozen selection decisions](../study/deployment-selection.json), [all attempted recipes](../study/recipe-registry.json), and [integrity audit](../study/confirmation-integrity.json) are retained.

**No acceptable composed candidate passed the frozen selection contract.** Vision repairs reduced measured deficits but did not retain every required capability. Text met its confirmation floors, but its prior selection gate failed, so it remains unaccepted. These are experimental artifacts, not production-ready all-capability models.

## Vision confirmation

Accuracy in percent. Main floors: clean 88; left/right/inverted 80. Each row evaluates 360 original source images in each of four domains. Every transformed view stays within its source partition. Writer IDs are unavailable. Seed29 reuses seed 11 merge scalars; its earlier pilot exposure prevents a wholly unseen-seed claim.

| Seed | Method | Clean | Left | Right | Inverted | Floors met | Worst deficit, points |
|---|---|---:|---:|---:|---:|---:|---:|
| 11 | base | 95.8 | 46.4 | 45.0 | 0.6 | 1/4 | 79.4 |
| 11 | expert-left | 70.3 | 94.2 | 19.2 | 1.9 | 1/4 | 78.1 |
| 11 | expert-right | 64.7 | 29.2 | 93.9 | 0.0 | 1/4 | 80.0 |
| 11 | expert-inverted | 25.8 | 20.3 | 14.2 | 93.6 | 1/4 | 65.8 |
| 11 | task_arithmetic | 28.3 | 39.7 | 24.2 | 27.2 | 0/4 | 59.7 |
| 11 | ties | 40.0 | 35.8 | 39.2 | 12.8 | 0/4 | 67.2 |
| 11 | dare | 33.3 | 27.5 | 16.1 | 20.0 | 0/4 | 63.9 |
| 11 | tsv | 59.2 | 35.3 | 51.4 | 10.0 | 0/4 | 70.0 |
| 11 | regmean | 84.7 | 78.3 | 66.7 | 27.8 | 0/4 | 52.2 |
| 11 | repair-strongest | 86.9 | 79.2 | 67.8 | 57.2 | 0/4 | 22.8 |
| 11 | joint_finetune_strongest | 92.5 | 80.0 | 74.2 | 55.8 | 2/4 | 24.2 |
| 11 | ensemble-reference | 75.8 | 80.3 | 70.0 | 2.8 | 1/4 | 77.2 |
| 29 | base | 94.7 | 44.7 | 45.8 | 0.6 | 1/4 | 79.4 |
| 29 | expert-left | 67.8 | 92.8 | 23.9 | 0.8 | 1/4 | 79.2 |
| 29 | expert-right | 62.5 | 31.4 | 94.7 | 0.3 | 1/4 | 79.7 |
| 29 | expert-inverted | 11.7 | 12.8 | 11.4 | 95.8 | 1/4 | 76.3 |
| 29 | task_arithmetic | 19.4 | 14.7 | 28.9 | 40.3 | 0/4 | 68.6 |
| 29 | ties | 31.4 | 21.9 | 28.6 | 35.3 | 0/4 | 58.1 |
| 29 | dare | 18.3 | 7.2 | 17.5 | 26.7 | 0/4 | 72.8 |
| 29 | tsv | 67.8 | 46.1 | 40.0 | 23.1 | 0/4 | 56.9 |
| 29 | regmean | 88.1 | 67.8 | 79.2 | 61.7 | 1/4 | 18.3 |
| 29 | repair-strongest | 92.2 | 74.7 | 81.7 | 77.8 | 2/4 | 5.3 |
| 29 | joint_finetune_strongest | 92.8 | 73.1 | 80.6 | 77.5 | 2/4 | 6.9 |
| 29 | ensemble-reference | 71.4 | 79.2 | 59.7 | 6.4 | 0/4 | 73.6 |

TSV and RegMean actually executed. Weight-only operators were tuned with selection labels, so the whole selection pipeline is not data-free. TSV uses compressed/polar-decorrelated task subspaces; RegMean uses calibration activation statistics, bias augmentation and declared regularization. These are disclosed small-MLP implementations, not claims of exact full-paper replication. The probability ensemble retains three complete experts and is a quality/cost reference.

RegMean→repair improves clean accuracy by 2.2 and 4.2 points on seeds11/29. Clean regression against the corresponding base is 8.9 and 2.5 points, within the registered 10-point limit on confirmation, while absolute and other task floors still fail. On seed 11 the joint comparator meets more floors (2/4) than repair (0/4), while repair has a slightly smaller worst deficit. Neither dominates the other.

## Text adapter confirmation

Balanced exact-text-deduplicated SMS subset, 262 source messages across four domains. Original labels are [ham, spam]. This is fixed character-feature classification, not a generative language-model benchmark. Near-duplicate templates can span splits.

| Method | Clean | Uppercase | Leet | Underscores | Floors met | Worst deficit, points |
|---|---:|---:|---:|---:|---:|---:|
| base | 93.5 | 79.0 | 61.5 | 70.6 | 2/4 | 10.5 |
| expert-uppercase | 84.7 | 95.0 | 59.2 | 79.4 | 3/4 | 12.8 |
| expert-leet | 86.6 | 71.4 | 91.2 | 71.0 | 2/4 | 1.0 |
| expert-underscores | 78.6 | 77.9 | 54.6 | 93.1 | 3/4 | 17.4 |
| task_arithmetic | 86.3 | 90.8 | 67.2 | 85.5 | 3/4 | 4.8 |
| repair | 86.3 | 90.1 | 78.2 | 85.9 | 4/4 | 0.0 |

Text floors were clean75% and each transformed domain72%, with at most3 points clean regression relative to the ordinary merge. Repair leet selection accuracy was71.43%, below72%. Its later78.24% confirmation result cannot retroactively accept it. Clean confirmation accuracy stays86.26% after repair, a7.25-point regression from the text base. Three rank 4 specialist adapters were trained with frozen base weights, and their folded dense experts match adapter logits exactly on three changed inputs each ([audit](../study/text-adapter-audit.json)).

## Controls and ablations

All following vision seed 11 controls were registered before confirmation. Main floors apply except the explicitly custom easy control.

| Recipe | Gradient example accesses | Accepted rank upper bound | Floors met | Worst deficit, points |
|---|---:|---:|---:|---:|
| repair | 537 | 3 | 0/4 | 54.9 |
| ablation-no_priority | 2864 | 8 | 0/4 | 36.1 |
| ablation-fixed_rank | 537 | 2 | 0/4 | 54.9 |
| ablation-no_protection | 537 | 3 | 0/4 | 54.9 |
| ablation-no_correction | 0 | 0 | 0/4 | 59.7 |
| insufficient-calibration | 2 | 1 | 0/4 | 59.7 |
| easy-merge | see report | 0 | 4/4 | 0.0 |
| conflict-ordinary | see report | 0 | 0/4 | 79.4 |
| conflict-repaired | see report | 5 | 0/4 | 75.0 |

Removing priority performs better here but uses 2864 gradient accesses versus 537, so it does not isolate priority at equal data cost. Fixed rank1 matches default arithmetic-repair metrics with a smaller accepted rank; removing protection is identical in this run. These ablations do not establish individual benefits of priority, variable-rank allocation or protection. Core tests separately verify that the protection gate rejects damaging proposals.

The easy merge repeats one expert three times and uses custom selection-derived floors (selection accuracy minus3 points), including very weak inverted/right requirements. It verifies the zero-correction fast path; it is not success on the main contract. Conflict repair and one-example-per-task calibration both fail the main contract. Pilot inversion specialists were weak; their failures and the pre-confirmation training amendment remain in the ledger.

## Cost and calibration

Strongest repair and same-start joint fine-tuning each use4 gradient steps and716 calibration example accesses, with zero distillation-teacher calls. Joint draws from the same permitted four-domain pool. Both select with the same held-out selection partition. Repair tests 72 interventions and evaluates 79,920 selection examples plus 2864 calibration-metric examples, totaling 83,500 model-example accesses including gradients. Each joint LR trial evaluates 1080 selection examples and the three-LR search is logged separately. **This is a matched gradient-data/step budget, not a matched total-compute, selection-cost or exact-FLOP comparison.** RegMean also incurs3 parent activation forwards/537 calibration examples before repair. No general efficiency/superiority claim follows.

Text repair uses 520 gradient accesses,58,016 selection evaluations and2080 calibration-metric examples (60,616 total), rank upper bound6,72 interventions, zero parent or distillation calls.

| Selected experimental artifact | Total stored bytes | Weights bytes, full head included | Warm forward median ms | p95 ms |
|---|---:|---:|---:|---:|
| vision seed 11 | 105,297 | 52,856 | 0.032417 | 0.121008 |
| text | 128,029 | 74,840 | 0.031438 | 0.150837 |

vision seed 11 ten-bin expected calibration errors: clean=0.0524, inverted=0.1547, left=0.0717, right=0.1732.
text ten-bin expected calibration errors: clean=0.1133, leet=0.1288, underscores=0.0961, uppercase=0.0779.

All 143 retained artifacts occupy8,870,241 bytes, including all heads and the retained text adapter factors/metadata. A fresh finalization process peaked at252,641,280 bytes RSS and repeated arithmetic merge+repair in14.02s with identical weights; its process scope also includes ledger/joint-control setup and excludes expert training. Confirmation process peak RSS was233,897,984 bytes. Per-seed expert training/build timing and per-candidate wall times remain in raw results/registry. Strongest repair took20.84s and11.63s on seeds11/29; selected same-start joint trials took0.46s and3.15s. These noisy local wall times are not normalized throughput benchmarks.

Latency is cached model-forward time only, Apple M4 CPU, four Torch threads, batch1,20 warmups and100 repeats. It excludes preprocessing, disk loading, CLI startup and workbench requests. Large p95 outliers are retained. No speedup ratio is claimed. Core artifact loading verifies hashes and includes manifest overhead in stored bytes.

## Evidence limits

No external pretrained weights were used. The public datasets are old and may be present in external pretraining corpora; this local-from-scratch study makes no claims about future imported foundation models. Digit domains are artificial transformations from one source, SMS templates may overlap, there are only two vision initializations and one text initialization, and no confidence interval or significance claim is made. Source indices are disjoint and exact duplicates grouped; this does not establish writer/template-disjoint generalization.

The low-rank method is labeled data-assisted optimization. Gradient similarity only proposes directions; controlled selection interventions establish local measured effects, not a general causal account of interference. Empirical floors are not universal behavior guarantees. Removing an expert from a recipe is rebuilding, not machine unlearning. Failed bounded search means no acceptable candidate found, not impossibility.

Both selected static artifacts changed outputs in fresh installed processes while the OS denied all parent-project reads, caches and network ([vision](../study/independent-vision.json), [text](../study/independent-text.json)). The initial clean-install failures are retained; the missing runtime packaging dependency was fixed explicitly. Software malformed/nonfinite/adversarial-input checks do not certify adversarial ML robustness.
