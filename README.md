# Shallow MaxCut QAOA

Research code for unweighted MaxCut at QAOA depths `p=1,2`: exact cut
enumeration, analytic p1 scores, statevector expectations and gradients,
single-run L-BFGS-B optimization, frozen graph libraries, reproducible B1
task/reference/evaluation pools, read-only labels and diagnostics, and
training-derived B2 fixed starts and graph-only B3 theoretical starts with single-start comparisons.
Supplied production and pilot configurations are candidates; use the frozen
batch manifests for adopted settings. Machine learning and B4–B7 remain
outside the current implementation.

Use Python 3.12. From the repository root, install the pinned environment
in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
```

On Linux/macOS, substitute `.venv/bin/python`; those platforms have not been
verified by the local B2 checks. The numerical backend is PennyLane 0.43.0 `default.qubit`,
`shots=None`, Autograd, adjoint first derivatives and double precision.
`default.qubit` is the CPU reference. Explicit `lightning.gpu` selection uses
the same circuit and optimizer, requires a separately prepared Linux GPU
environment, and never falls back to CPU. GPU validation must establish the
actual device and environment before execution.
For predictable development resource use, set `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` before running Python.

Keep the editable (`-e`) installation: experiment provenance locates the
working code, configuration and entry points through the installed module.
Do not edit this checkout while a batch is running or awaiting resume.
Keep its original runtime snapshot and environment for continued execution;
a new B2 batch does not make an edited checkout compatible with old B1 resume.

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

For larger new batches, `run --partitioned` stores active records by graph and
experiment role. A completed group is sealed into one checksummed `.tgz`, with
the original start, result and I/O bytes retained; verified loose duplicates
are then removed. Failures and interrupted-start evidence remain in the group.
On Linux, archive contents and the sealed directory and its parent are synced
before loose records are removed, including interrupted-cleanup recovery.
A sync error retains those loose records and stops compaction; this relies on
the filesystem honoring `fsync`, not on a power-failure guarantee for the server.
Recognized unpublished temporary bytes are preserved but never count as results.
Old flat directories remain readable; select a new output for this layout.
Use one writer per partitioned output; an OS lock rejects overlapping writers
and is released when the process exits. `--graph-ids ID ...` limits record reads
to named frozen graphs. Partitioned outputs do not use task shard flags.
`completed_scope=selected_groups` means completion counts cover the selected
graphs and roles; `planned` still describes the entire frozen batch. A task
limit is not evidence of batch completion. A full read-only audit still checks
every group. Hashed record paths use extended Windows paths when needed.

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

For large outputs, add `summarize --partitioned-output` to write each graph's
JSON/Parquet tables separately and a final root manifest linking their hashes
and completion counts. Trace memory is limited to graph-sized groups; the
frozen task list and identity metadata still scale with the batch. The root
manifest is published only after all graph summaries succeed. Its `complete`
flag requires all configured tasks to have valid executions and all required
references to be complete; scientific success-label counts are separate.
An interrupted summary directory is retained; choose a new summary output.

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

## B2 fixed starts

`learning.py` fits a common-symmetry medoid and a separately frozen, one-pass
aligned coordinate median from complete training references at each depth.
The distance is Euclidean radians on gamma period 2pi, beta period pi/2,
quotiented by simultaneous reversal of all angles. It is a parameter distance,
not an energy metric. LOFO retains the original training/evaluation boundary
and excludes multi-family classes on both sides.

Angle rules `b2-common-quotient-v2` resolve sign choices using the coordinate
with the largest separation between the two representatives. This prevents a
near-zero leading angle from deciding a distant aligned-median lift. Older
rule-v1 fits/batches require their original source for replay; refit and plan
into new paths with v2, reusing the audited B1 references. Existing artifacts
are not migrated or overwritten.

With existing B1 data, use new output paths for each fit and batch:

```bash
python scripts/experiment.py fit-b2 --batch B1_BATCH --attempts B1_ATTEMPTS \
  --references B1_REFERENCES --regime random --output B2_FIT.json
python scripts/experiment.py plan-b2 --batch B1_BATCH --attempts B1_ATTEMPTS \
  --references B1_REFERENCES --fit B2_FIT.json \
  --config configs/warm-benchmark-candidate.json --output B2_BATCH
python scripts/experiment.py run --batch B2_BATCH --output B2_ATTEMPTS
# Explicitly select the validated backend and add --execute to run.
python scripts/experiment.py summarize-b2 --batch B2_BATCH --attempts B2_ATTEMPTS \
  --b1-batch B1_BATCH --b1-attempts B1_ATTEMPTS --output B2_SUMMARY
```

For LOFO, fit with `--regime lofo --fold regular` (or er/ba/sbm), using a
separate output for every fold. Fitting requires all declared training
references, without requiring the 50-restart success labels; planning also
requires complete target references. Original references are re-derived from
validated attempts and bound by identity. B2 inherits the frozen B1 optimizer
settings. Existing B1 traces are compared only when objective/optimizer source,
budgets and execution environments agree; this read permission never changes
the strict source/environment rules for resuming B1.

Each variant has one start per graph. B1 repeats estimate random single-start
performance: average within each graph first, then give every graph equal
weight. Initial score, terminal score/success, first hit and complete costs
remain separate. Missing/faulted runs keep their planned denominators and
suppress formal paired intervals. JSON includes comparisons and first-hit
curves; Parquet retains graph/restart tables. Unreached median/p90 are null.
The two variants are never selected by their scores on individual test graphs.

## B3 graph-only starts

B3 uses one deterministic initialization per evaluation graph and depth. At
p1 it chooses maximum-degree tree gamma and analytically optimizes beta at
that gamma. At p2 it scales the published infinite-degree angles by the
arctangent of the inverse square root of actual mean degree minus one.
The rule and source constants are recorded in each preparation artifact.
These are finite-graph heuristics, with no general optimality or improvement
guarantee. They require neither training labels nor target-graph optimization.

Prepare once in the final source and numerical environment, matching the
original B1/B2 execution environment. Planning checks compatibility with B1
before execution. Read, dry-run, resume and summary never regenerate angles.

```bash
python scripts/experiment.py prepare-b3 --library LIBRARY \
  --config configs/b3-candidate.json --backend default.qubit --output INITIALIZATIONS.json
python scripts/experiment.py plan-b3 --initializations INITIALIZATIONS.json \
  --batch B1_BATCH --attempts B1_ATTEMPTS --references B1_REFERENCES \
  --config configs/b3-candidate.json --output B3_BATCH
python scripts/experiment.py run --batch B3_BATCH --output B3_ATTEMPTS --partitioned
# Select the matching backend and explicitly add --execute to run.
python scripts/experiment.py summarize-b3 --batch B3_BATCH --attempts B3_ATTEMPTS \
  --b1-batch B1_BATCH --b1-attempts B1_ATTEMPTS \
  --b2-case B2_RANDOM_BATCH B2_RANDOM_ATTEMPTS \
  --b2-case B2_REGULAR_BATCH B2_REGULAR_ATTEMPTS \
  --b2-case B2_ER_BATCH B2_ER_ATTEMPTS \
  --b2-case B2_BA_BATCH B2_BA_ATTEMPTS \
  --b2-case B2_SBM_BATCH B2_SBM_ATTEMPTS --output B3_SUMMARY
```

The candidate declares both depths and all five B2 cases. LOFO reuses B3
attempts; it creates no new runs. Comparisons include paired complete-call
differences and intervals, with preparation and reference costs separated by
depth. External optimization of the published constants is unmeasured,
not zero. Missing results retain planned denominators and suppress intervals.

`snapshot.py --checkpoint` archives B3 source, library, configuration,
preparation, complete attempts, summary and a read-only reproduction script.
It requires `--batch`, `--attempts`, `--summary`, and `--dependencies` in
addition to the existing `--output`, `--library`, and `--config`. Dependencies
declare canonical B1/B2 paths and SHA256 values plus the explicit comparison
input paths; archives are referenced without nesting. Verification with
`--external-root` additionally checks these separately retained dependencies.

## B4 Ridge initialization

B4 predicts one angle vector from topology features, then uses the same B1
optimizer and budget. A separate scalar Ridge predicts the original 50-restart
success rate; it never selects an angle or adds a restart. Both heads use only
the frozen global training graphs. Inner CV refits preprocessing and the angle
anchor. Gamma uses sin/cos, beta uses sin(4 beta)/cos(4 beta), with joint sign
and the graph's applicable exact degree-parity symmetries.

The candidate fixes five groups: L, U, F=U+S, J+U, J+U+S. J is a complete p1
joint-edge dictionary, not a sufficient p2 description. Spectral-group gains
measure representation utility, not uniquely nonlocal information. Three
random/F whole-row shuffles are descriptive controls. The full declaration
has 56 angle models, 56 success models and 5,200 single-start tasks.

```bash
python scripts/experiment.py fit-b4 --batch B1_BATCH --attempts B1_ATTEMPTS \
  --references B1_REFERENCES --config configs/b4-candidate.json --output B4_FITS
python scripts/experiment.py predict-b4 --fits B4_FITS --library ORIGINAL_LIBRARY \
  --output B4_PREDICTIONS.json
python scripts/experiment.py plan-b4 --batch B1_BATCH --attempts B1_ATTEMPTS \
  --references B1_REFERENCES --predictions B4_PREDICTIONS.json \
  --config configs/b4-candidate.json --output B4_BATCH
```

`fit-b4` resumes complete model files after checking frozen inputs; other
artifacts are create-only. `run` uses the existing worker, with `--execute`
required. `summarize-b4` takes B4 `--batch/--attempts`, original
`--b1-batch/--b1-attempts`, all five `--b2-case BATCH ATTEMPTS` pairs,
`--b3-batch/--b3-attempts`, and `--output`. It reports full first-hit curves,
terminal quality, calls, prediction errors and paired graph intervals. Missing
attempts retain planned denominators and make the result provisional.

CPU learning dependencies are pinned separately from the historical compute
environment fingerprint. Prediction/task planning does not retrain a model or
evaluate candidate angles. `snapshot.py --checkpoint-b4 --fits B4_FITS` adds the
complete feature/training/model artifacts to the existing checkpoint workflow;
its dependencies must name canonical B1/B2/B3 archives. Complete checkpoint
creation rejects incomplete summaries. Formal pools/GPU execution remain
explicit user operations, separate from local CPU validation.

Optional repeated `--support PATH` includes explicitly selected B4 run/driver
or backend-validation evidence inside the checkpoint; paths must stay inside
the project and cannot include the whole project or previous canonical archives.
CV and final fitting are timed separately. Unmeasured historical/offline I/O
costs are reported as unknown, rather than treated as zero.
