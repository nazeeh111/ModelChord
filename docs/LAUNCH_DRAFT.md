# Launch draft

MODELCHORD composes compatible specialists into one static classifier under explicit capability floors, measures the shortfalls, and attempts a bounded low-rank repair.

In v0.2, the selected digit model passes all four required floors on 3,823 locked source images: 91.11% clean, 82.53% left-shifted, 83.49% right-shifted and 85.01% inverted. The data-assisted repair folds into ordinary dense weights rather than calling parent experts at inference.

The comparisons matter. Ordinary joint and fixed-factor training also pass the vision contract and score better on all three transformed tasks. Two ablations produce identical weights to repair. The separate character-feature text-adapter integration fails every floor on shifted YouTube spam comments; there is no fresh SMS confirmation. This is a bounded vision retention result, not proof that repair is necessary or generally superior.

The repository includes a local workbench, API/CLI, independent application examples, frozen source/data identities, failed attempts, exact per-task metrics and fresh-workspace reproduction evidence. All 12 v0.2 main weight outputs reproduced exactly on the tested environment. The original v0.1 negative-result study is retained. Installation, runtime isolation, interface and public-release checks are distinct delivery evidence; use their actual recorded scope when sharing.

[Source and evidence](https://github.com/nazeeh111/ModelChord) · [v0.2 results](RESULTS_V02.md) · [Reproduction](REPRODUCTION_V02.md) · [Short demo](media/v02/README.md)

Draft only. No external message or social post is authorized by this document. No invented adoption, general novelty, compute-efficiency or universal capability claim.
