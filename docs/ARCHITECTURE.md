# Architecture

Editable source: [ARCHITECTURE.mmd](ARCHITECTURE.mmd).

```mermaid
flowchart LR
    S[Same-base specialist artifacts] --> C{Compatibility validator}
    B[Immutable common base] --> C
    C -->|Reject contract mismatch| X[Actionable incompatibility]
    C -->|Compatible| M[Layerwise conventional merge]
    M --> O[Independent static baseline]
    O --> G{Capability floors met?}
    G -->|Yes| E[Export static safe tensors]
    G -->|No| P[Measure deficits and protected margins]
    D[Permitted calibration examples] --> P
    P --> I[Propose low-rank layer interventions]
    I --> V{Disjoint selection evidence}
    V -->|Improve worst deficit and preserve margins| F[Fold accepted correction]
    V -->|Reject| R[Rollback / measured frontier]
    F -->|Budget remains| P
    F --> E
    R --> E
    E --> A[Independent application with parents unavailable]
    H[Untouched confirmation set] --> Q[Final report and uncertainty]
    E --> Q
```

The exported application loads one static artifact. Ensembles are comparison references only. Repair changes only bounded low-rank matrix directions and folds accepted edits into the artifact; ordinary full-weight fine-tuning is an explicitly separate baseline. The model and its full classification head are included in all storage and latency reports.
