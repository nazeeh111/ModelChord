# Independent review of locked confirmation and release readiness

Reviewed 2026-10-29 after the one-shot confirmation. No numerical core, locked source, artifact, protocol, selection or result file was changed by this review. The reviewer directly read `study/results.json`, deployment selection, independent-run evidence, confirmation integrity, study source and current release tree.

## Evidence integrity: pass

Independent hashing found zero mismatches across the 327 source/artifact files bound by confirmation lock SHA256 `a10af8c063df6b8b810fa3cb1253b1fba5e0c91cd235d005e10fa20944402a56`. Results SHA256 is `75b7080377895c1199d9802c8d48f6c09e237fbc22fcdf8784f459b05f13ca45`, matching the integrity record. The deployment-selection file still matches its pre-confirmation locked hash. Each selected vision/text artifact maps to exactly one confirmation row. No evidence of post-confirmation recipe or numerical-source revision was found.

## Actual result and permitted claims

**No composed vision model met all four main capability floors.** For seed 11, strongest ordinary RegMean's worst deficit was .52222; strongest-method repair reduced it to .22778, with accuracies clean .86944, left .79167, right .67778, inverted .57222. All four remain below the .88/.80 floors. For seed 29, RegMean's worst deficit was .18333; strongest-method repair reduced it to .05278, with clean .92222, left .74722, right .81667, inverted .77778. Two of four floors are met. These are measured improvements on this suite, not evidence that the declared retention requirements were achieved or a proof of impossibility.

The source experts are real competent controls: their intended confirmation tasks are approximately .942/.939/.936 on seed 11 and .928/.947/.958 on seed 29. Severe composition failure therefore cannot be hidden by presenting three weak specialists as successful skills. Static parent-free export works independently of whether the composed model meets quality floors.

Text repair met all four **confirmation** floors: clean .86260, uppercase .90076, leet .78244, underscores .85878. Ordinary arithmetic had leet .67176, below its .72 floor. However, the frozen text candidate missed its **selection** leet floor (.71429 versus .72). Both frozen vision and text deployment selections therefore retain `no acceptable candidate found`. The improved final text score must not retroactively promote the candidate, change its selection status, change floors, or justify another tuning pass against the same final set. A public research artifact can be run and inspected with this status clearly disclosed.

The easy control meets its own custom selection-derived floors with zero accepted correction rank; it is not a main-suite all-floor success. Conflict and insufficient-calibration controls remain failures. Their presence is evidence of exercised failure paths, not repaired capability retention.

## Matched comparison and ablations

The strongest-start ordinary joint fine-tuning comparison is particularly important:

| Seed | RegMean + bounded repair worst deficit | Same-start joint FT worst deficit | Floors met, repair / joint |
| --- | ---: | ---: | ---: |
| 11 | .22778 | .24167 | 0/4 / 2/4 |
| 29 | .05278 | .06944 | 2/4 / 2/4 |

Both selected candidates use 716 gradient-example accesses in each seed. Repair has a slightly smaller worst deficit; joint fine-tuning meets more floors on seed 11 and retains better clean accuracy. This is a frontier tradeoff, not unqualified repair superiority. Two initializations on one shared source split do not support statistical or broad generalization claims.

Per-candidate gradient-example accesses, gradient steps and distillation-teacher calls match the declared comparators. Repair uses far more repeated selection evaluations; ordinary fine-tuning searches three learning rates and changes full weights. Exact FLOPs, total search cost and selection compute are not matched. RegMean's three expert activation forwards and 537 activation examples are recorded separately from zero distillation calls. Describe this as a gradient-data/step matched comparison, not a fully compute-matched study.

The ablations do not establish every proposed repair component as beneficial. On seed 11, no-priority repair has worst deficit .36111 versus prioritized arithmetic repair .54944, while consuming 2,864 versus 537 gradient-example accesses. That cost difference prevents attributing the difference to priority alone. Fixed-rank and no-protection ablations have the same displayed confirmation accuracies as prioritized arithmetic repair; the current suite supplies no demonstrated quality advantage for those components. No-correction retains worst deficit .59667, so the bounded correction does improve the arithmetic starting point on the tested objective. Keep all these outcomes visible.

## Static execution and installation: supported locally

`study/independent-vision.json` and `study/independent-text.json` each record return code zero from an independent wheel installation using isolated Python, a copied standalone artifact, OS-denied project/parent and cache access, and OS-denied network. Both produce changed outputs for changed inputs and explicitly verify project/network denial. Their full class heads are present. Recorded deployed artifact totals are 105,297 bytes for vision and 128,029 bytes for text, including substantial recipe metadata. Dense weights, manifests and full stored-artifact totals must not be confused with adapter-only size.

These checks establish local independent static inference. They do not establish successful capability retention, public clean-clone installation or remote CI. Failed earlier independent attempts are retained separately. `pyproject.toml` now declares the previously missing `packaging` dependency explicitly.

Warm batch-one model-forward timings are approximately hundredths of a millisecond, with all heads/folded updates present. They exclude input preprocessing, artifact loading, Python/CLI startup and application overhead. Total retained artifact storage is 8,870,241 bytes across the study, not the deployment size of one selected model. The build-memory record's broader finalization-process scope and the confirmation-process peak are separately labeled; neither supports a per-method memory-superiority claim.

## Remaining release gates at this review

The implementation, typed API/CLI, safe numeric artifacts, compatibility failures, actual spectral/data-assisted baselines, negative controls, bounded repair, editable architecture source, workbench screenshots/checks, evaluator example and two application source examples exist. Independent local static execution has evidence. The fixed small MLP registry and character-feature text integration are honestly scoped; no arbitrary foundation-model support or universal classification is established.

The source tree still had a placeholder README saying measured results would replace it before release. No completed results walkthrough or genuine short demo video was present at this snapshot. `examples/predict.py` mentions nonexistent old paths and `study/summary.json`; `docs/API.md` uses pilot `vision-seed11` paths. Update these unlocked explanatory files to the actual frozen selection and include the measured negative status. Those documentation changes need no numerical revision or new confirmation set.

Public source publication, public clean-clone execution, remote CI success, versioned GitHub Release assets/downloads/links and owner walkthrough were not yet completed at this review. The workflow exists, but a workflow file is not an executed CI result. A publicly released experimental feasibility tool with honest negative results is supportable once these gates are verified. It must not be described as having solved the vision retention requirements or as having established broad repair superiority.

No hardware action or additional paid resource is required to finish the remaining scoped packaging/documentation/publication gates. Achieving better scientific performance would be a new study with fresh untouched confirmation data, not a permitted amendment to these fixed results. Report unmet quality requirements directly rather than treating publication or UI polish as a replacement.

## Documentation and demo readback before publication

The completed README, RESULTS, API, INTEGRATIONS, REPRODUCTION, WALKTHROUGH and launch draft were compared directly with the frozen metrics. The displayed task accuracies, floors, worst deficits, same-start joint tradeoffs, text selection failure, mixed ablations, example-access counts, rank bounds, ECE values, artifact/weight bytes and latency/build-time values match the reports. Negative acceptance status remains explicit. No unqualified superiority or all-floor vision-success claim was found.

The docs explicitly distinguish the small digit-domain suite from general image classification, the character-feature adapter integration from LLM composition, exact duplicate/source isolation from writer/template isolation, and gradient-data/step matching from total-compute matching. They retain attribution, the pre-confirmation pilot amendment and two-seed limitations. Relative Markdown links in the reviewed root/docs files resolve.

Two material presentation/startup defects were reported and corrected in unlocked files: vision paths accidentally contained `vision-seed 11-v2` instead of the actual `vision-seed11-v2`, and the RESULTS cost table's text row was separated from its header by an ECE paragraph. The current readback has correct paths and a contiguous table.

The initial reproduction wrapper could not initialize a new study: it omitted the registry required by frozen training source and claimed a dataset download that neither wrapper nor training source performed. The revised unlocked wrapper now creates an empty registry/private-data directory, downloads only the official public UCI zip, verifies its registered SHA256 before saving, and supports prepare-only validation. It preserves the locked numerical source. This source readback does not claim a complete fresh reproduction execution; that remains a separately reported check.

The demo is accurately described as a 20-second, silent, captioned edit of completed real UI captures, rather than a continuous recording. The zero-frame screencast attempt and unsuccessful first encode are disclosed. Captions state the inverted digit is misclassified and that changed confidence demonstrates responsiveness rather than correctness. The selected v2 capture/encode manifest records source/master hashes and frame metadata. The failed v1 delivery is excluded by `.gitignore`. No unsupported live-motion, correctness or capability-retention claim was found in this description.

Documentation and claim framing are suitable for a scoped experimental negative-result release. Public source, remote CI, clean-clone execution and versioned-release download/link checks still require their actual publication evidence. This update does not alter the frozen selections, results or research conclusions.
