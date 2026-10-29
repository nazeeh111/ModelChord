# Launch draft

MODELCHORD is an experimental model-composition toolkit: set capability floors, combine compatible specialists, inspect what breaks, and attempt a bounded repair before exporting one static model.

The first study is deliberately small and reproducible. It trains three digit-domain specialists from a common base, compares arithmetic/TIES/DARE, TSV and RegMean, then tests low-rank repair against ordinary joint fine-tuning. A separate same-base text-adapter integration uses frozen character features.

The result is mixed. Vision worst-task deficits shrink on two seeds, but no composed model meets every required floor. Text passes confirmation after missing selection, so it stays unaccepted. The ablations also fail to establish that each proposed component helps. All 143 attempts, failed repairs, source hashes and per-task results are included.

The exported artifacts run with parents and network unavailable. There is a local workbench, typed API/CLI, evaluator extension, two applications and a versioned release.

Source and evidence: https://github.com/nazeeh111/ModelChord

This is a draft for relevant technical sharing, not a claim of adoption or general superiority. No external messages or social posts were sent.
