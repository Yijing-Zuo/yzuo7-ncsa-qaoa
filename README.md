# Shallow MaxCut QAOA

Research code for unweighted MaxCut at QAOA depths `p=1,2`: exact cut
enumeration, analytic p1 scores, statevector expectations and gradients,
single-run L-BFGS-B optimization, graph generation, topology features and
Parquet graph libraries. Configurations are small development settings;
production experiments and machine learning are not included.

Use Python 3.12. From the repository root, install the pinned environment
and run a small example in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation .
.\.venv\Scripts\python.exe scripts/score_graph.py --graph triangle --p 1
.\.venv\Scripts\python.exe scripts/optimize_graph.py --graph cycle5 --p 2
```

On Linux/macOS, substitute `.venv/bin/python`; those platforms have not been
verified. The numerical backend is PennyLane 0.43.0 `default.qubit`,
`shots=None`, Autograd, adjoint first derivatives and double precision.
Lightning is an installed dependency; the circuit uses `default.qubit`.
For predictable development resource use, set `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` before running Python.

The score command prints JSON with the exact optimum, one cut witness,
expectation, loss, gradient and device counts. The optimization command saves
a complete JSON attempt under `data/development/optimization/`, including
accepted and trial points, the final accepted score, `best_seen`, stop reason
and measured computation counts. Both commands support `--help` and `--output`.

To create a small, frozen development library:

```powershell
.\.venv\Scripts\python.exe scripts/build_library.py --output data/development/example-library --evidence validation/example-library.json
```

The supplied configuration generates connected regular, ER, BA and SBM graphs,
deduplicates by topology, saves provenance and fixed graph splits, and computes
exact answers for sizes 12/16. Sizes 20/24 receive features with an explicit
missing exact answer. Existing library directories are never overwritten;
choose a new output and evidence path for each build. Read saved libraries
with `qaoa_study.records.read_library` rather than regenerating them.

The positive cut observable is `C = sum((I - Z_u Z_v)/2)` and the loss is
`-expectation(C)`. Angles are radians ordered `[gamma..., beta...]`.
Starting from `|+>`, each layer applies cost `IsingZZ(-gamma)` on edges,
then mixer `RX(2*beta)` on nodes. Optimization uses unbounded angles and one
joint loss/gradient request per objective call; development budgets are not
final production settings. Exact enumeration is exponential and supports
chunking through 24 vertices; this is not a production performance claim.

References: [Farhi, Goldstone and Gutmann (1411.4028v1, Section I)](https://arxiv.org/abs/1411.4028v1)
defines QAOA; the p1 edge formula follows corrected
[Wang et al. (1706.02998v2, Eq. 14 and Appendix A)](https://arxiv.org/abs/1706.02998v2).
The implementation uses [PennyLane 0.43.0](https://github.com/PennyLaneAI/pennylane/tree/v0.43.0),
chosen in place of the original study proposal's CUDA-Q.
