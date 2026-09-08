# Shallow MaxCut QAOA

Research code for unweighted MaxCut at QAOA depths `p=1,2`: exact cut
enumeration, analytic p1 scores, statevector expectations and gradients,
single-run L-BFGS-B optimization, frozen graph libraries, reproducible B1
task/reference/evaluation pools, and read-only labels and diagnostics.
Production and 100-graph pilot configurations are executable candidates;
neither has been run or frozen by a pilot. Machine learning and B2–B7 are
outside the current implementation.

Use Python 3.12. From the repository root, install the pinned environment
in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

On Linux/macOS, substitute `.venv/bin/python`; those platforms have not been
verified. The numerical backend is PennyLane 0.43.0 `default.qubit`,
`shots=None`, Autograd, adjoint first derivatives and double precision.
`default.qubit` is the CPU reference. Explicit `lightning.gpu` selection uses
the same circuit and optimizer, requires a separately prepared Linux GPU
environment, and never falls back to CPU. GPU execution has not been validated.
For predictable development resource use, set `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` before running Python.

Keep the editable (`-e`) installation: experiment provenance locates the
working code, configuration and entry points through the installed module.
Do not edit this checkout while a batch is running or awaiting resume.

On an allocated Linux GPU session, use a separate Python 3.12 environment.
The following pinned GPU additions are installation candidates until verified
on the actual device; they do not install a system NVIDIA driver:

```bash
python3.12 -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -c requirements-lock.txt \
  "PennyLane-Lightning-GPU==0.43.0" "custatevec-cu12==1.9.0"
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/verify_backend.py --backend lightning.gpu --sizes 4 6
python scripts/verify_backend.py --execute --backend lightning.gpu \
  --sizes 4 6 --depths 1 2 --sample-gpu-memory \
  --output data/manual/backend/small-vjp.json
```

The first verification command only prints a plan. Check the executed report
before progressing to one size at a time (12, 16, 20, 24), using
`--config configs/experiments-candidate.json` and a fresh output filename.
These fixed 3-regular cases do not establish all-family or full-pool readiness.
A Git clone has no `snapshot.json`; `snapshot.py --verify` only applies to
archives created by `snapshot.py`, not to an ordinary clone.

For a small calculation, use the same functions as the experiment worker:

```python
import networkx as nx
from qaoa_study.exact import exact_maxcut
from qaoa_study.qaoa import make_qaoa, qaoa_loss_and_gradient
from qaoa_study.optimize import OptimizerSettings, optimize_qaoa

graph = nx.cycle_graph(5)
theta = [0.37, -0.81, 0.21, 0.46]  # gamma1, gamma2, beta1, beta2
print(exact_maxcut(graph))
print(qaoa_loss_and_gradient(theta, make_qaoa(graph, 2)))
result = optimize_qaoa(graph, 2, theta, OptimizerSettings(max_evaluations=12))
print(result["C_final"], result["stop_reason"])
```

The result includes accepted and trial points, the final accepted score,
`best_seen`, stop reason and measured computation counts. `records.save_run`
can save a complete attempt to a new path.

To create a small, frozen development library:

```powershell
.\.venv\Scripts\python.exe scripts/build_library.py --output data/development/example-library --evidence validation/example-library.json
```

The default development configuration generates connected regular, ER, BA and SBM graphs,
deduplicates by topology, saves provenance and fixed graph splits, and computes
exact answers for sizes 12/16. Sizes 20/24 receive features with an explicit
missing exact answer. Existing library directories are never overwritten;
choose a new output and evidence path for each build. Read saved libraries
with `qaoa_study.records.read_library` rather than regenerating them.
`check_library` checks cut witnesses and record/feature consistency; it does
not independently prove that a stored cut is optimal. Independent correctness
tests remain in the complete local development repository.

Plan and check tasks without quantum, grid, gradient or exact-solver computation:

```powershell
.\.venv\Scripts\python.exe scripts/experiment.py plan --library data/development/example-library --config configs/experiments-candidate.json --output data/development/example-batch
.\.venv\Scripts\python.exe scripts/experiment.py run --batch data/development/example-batch --output data/development/example-attempts
```

The small development library deliberately cannot satisfy the full candidate
design: these commands report missing counts, coverage and exact answers and
return code 2. They do not silently complete the library or start experiments.
The full graph candidate covers 1504 graphs in 32 cells and requires exact
answers for all four sizes. Building it needs `build_library.py --config
configs/graphs-production-candidate.json --allow-study-build` and an explicit
new output/evidence path; this is a manual, potentially expensive operation.

For a verified complete batch, execution requires `experiment.py run --execute`.
Use `--roles reference` to run reference tasks separately, `--max-tasks` to limit
actual attempts, and `--shard-index I --shard-count N` for deterministic shards.
The limit includes retries and is not a wall-clock limit. `completed` counts
selected tasks across the batch, including faults; `valid_completed` excludes
faults. Missing/fault totals apply to the reported role/shard `work_scope`.
Repeated invocations verify task, source, environment and record integrity,
skip completed valid tasks, and preserve every fault/retry attempt. Do not
change source files within a resumable batch or overlap old/new sharding plans.
Completed files cannot be overwritten. Budget/iteration limits are normal
outcomes; numerical/device/program faults return a nonzero status after saving.

Freeze references and rebuild summaries from existing records only:

Complete `run --roles reference --execute` first, freeze complete references,
then run `--roles tier1 evaluation gradient_diagnostic --execute` and summarize.
The optimizer itself does not read the reference. A zero exit code does not
establish that all planned pools or shards have finished.

```powershell
.\.venv\Scripts\python.exe scripts/experiment.py freeze --batch data/manual/batch --attempts data/manual/attempts --output data/manual/references-v1
.\.venv\Scripts\python.exe scripts/experiment.py summarize --batch data/manual/batch --attempts data/manual/attempts --references data/manual/references-v1 --output data/manual/summary-v1
```

Use new output directories for every reference/summary version. An incomplete
reference freeze returns 2 and remains explicitly provisional. Summary JSON
and Parquet retain planned denominators, missing/fault counts and provenance;
a successful summary command does not mean that complete 50-restart labels
exist. Evaluation results never redefine a frozen reference.

`scripts/verify_backend.py` defaults to a no-computation plan; `--execute`
explicitly runs fixed-parameter comparisons and complete restarts. It supports
`--sizes`, `--depths`, `--backend`, and the candidate optimizer configuration.
Defaults are small sizes 4 and 6. Every requested GPU case retains the fixed
CPU score/gradient comparison; only the selected backend runs a complete
restart unless `--cpu-restart` is supplied. Reports mark unrequested CPU
restarts explicitly. Large sizes and GPU availability remain unverified.
Both backends explicitly request adjoint device VJP; CPU checks use double
precision. Raw Tracker `vjps` and `derivatives` are separate work counters.
New records use version 3; older records remain readable without adding missing
VJP/environment information. Resume requires matching recorded computation
dependencies and GPU model/driver. GPU UUID/host/placement belong to attempts
and do not prevent identical devices from working on disjoint shards.
`scripts/snapshot.py --output ARCHIVE.zip --library LIBRARY --config CONFIG`
packages the actual working runtime/configuration files and one frozen library,
excluding private material, old Git history and environments. After extraction
and dependency installation, `scripts/snapshot.py --verify DIRECTORY` checks
the snapshot hashes. `jobs/experiment.slurm` calls the same experiment entry;
resource/account settings must be supplied for the actual platform.

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
