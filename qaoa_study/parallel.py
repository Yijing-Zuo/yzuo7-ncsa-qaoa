"""Bounded independent-task execution with optional GPU/MIG memory admission.

Workers run whole attempts and keep their own CUDA contexts. This module does
not split a statevector, alter task seeds, retry failures, or manage checkpoints.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from collections import deque
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import multiprocessing
import os
from math import ceil
from pathlib import Path
from time import perf_counter, sleep


MIB = 1024 ** 2


@dataclass(frozen=True)
class ParallelSettings:
    workers: int = 1
    gpu_device: str | None = None
    reserve_mib: int = 2048
    state_copies: int = 4
    worker_overhead_mib: int = 512
    memory_fraction: float = 0.85
    gpu_utilization_limit: int = 85
    poll_seconds: float = 2.0
    adaptive: bool = False
    initial_workers: int = 4
    ramp_seconds: float = 120.0
    min_completions: int = 8
    throughput_gain: float = 0.05
    recovery_seconds: float = 30.0

    def __post_init__(self):
        if (type(self.workers) is not int or self.workers < 1
                or type(self.state_copies) is not int or self.state_copies < 1
                or self.reserve_mib < 0 or self.worker_overhead_mib < 0
                or not 0 < self.memory_fraction <= 1
                or not 0 <= self.gpu_utilization_limit <= 100 or self.poll_seconds <= 0
                or type(self.initial_workers) is not int or self.initial_workers < 1
                or self.ramp_seconds <= 0 or self.min_completions < 1
                or self.throughput_gain < 0 or self.recovery_seconds < 0):
            raise ValueError("Use positive workers/state copies and nonnegative memory reserves.")


def estimate_statevector_bytes(n, *, state_copies=4, worker_overhead_mib=512):
    """Admission estimate: complex128 states plus per-worker CUDA/workspace allowance.

    Depth increases gate work; it does not change the 2**n state dimension.
    The adjustable workspace allowance is an estimate, not a measured GPU peak.
    """
    return state_copies * 16 * (1 << n) + worker_overhead_mib * MIB


class _Memory(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("total", "free", "used")]


class _Utilization(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint) for name in ("gpu", "memory")]


class NvmlMemory:
    """Read the allocated GPU or MIG instance, never its parent's pooled memory.

    NVML documents UUID lookup and memory queries for MIG device handles:
    https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html
    """

    def __init__(self, device=None):
        self.lib = ctypes.CDLL("nvml.dll" if os.name == "nt" else "libnvidia-ml.so.1")
        self.lib.nvmlErrorString.restype = ctypes.c_char_p
        self._call("nvmlInit_v2")
        try:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            allocated = os.environ.get("NVIDIA_VISIBLE_DEVICES", "")
            allocated = allocated.split(",") if allocated.startswith(("GPU-", "MIG-")) else []
            tokens = visible.split(",") if visible is not None else allocated
            # Containers often expose CUDA ordinal 0 but an allocated MIG UUID.
            tokens = [allocated[int(t)] if t.isdecimal() and int(t) < len(allocated) else t for t in tokens]
            if device is not None and tokens and device not in tokens:
                raise ValueError("gpu_device must be inside CUDA_VISIBLE_DEVICES; use the allocated UUID.")
            token = device or (tokens[0] if len(tokens) == 1 else None)
            if token is None and tokens:
                raise ValueError("Select one allocated GPU/MIG UUID with gpu_device.")
            self.handle = ctypes.c_void_p()
            if token and token.startswith(("GPU-", "MIG-")):
                self._call("nvmlDeviceGetHandleByUUID", token.encode(), ctypes.byref(self.handle))
            else:
                # Auto-selection is unambiguous only when one accessible CUDA device exists.
                handles = self._devices()
                if len(handles) != 1 or token not in (None, "0"):
                    raise ValueError("Set gpu_device to the allocated GPU/MIG UUID; numeric ordinals are ambiguous.")
                self.handle = handles[0]
            uuid = ctypes.create_string_buffer(96)
            self._call("nvmlDeviceGetUUID", self.handle, uuid, ctypes.c_uint(len(uuid)))
            self.uuid = uuid.value.decode()
            if self.uuid.startswith("GPU-"):
                current, pending = ctypes.c_uint(), ctypes.c_uint()
                code = self.lib.nvmlDeviceGetMigMode(self.handle, ctypes.byref(current), ctypes.byref(pending))
                if code == 0 and current.value:
                    raise ValueError("The selected GPU uses MIG; select its allocated MIG UUID for memory accounting.")
        except BaseException:
            self.close()
            raise

    def _call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            raise RuntimeError(f"{name}: {self.lib.nvmlErrorString(code).decode()}")

    def _devices(self):
        count, handles = ctypes.c_uint(), []
        self._call("nvmlDeviceGetCount_v2", ctypes.byref(count))
        for index in range(count.value):
            handle, current, pending = ctypes.c_void_p(), ctypes.c_uint(), ctypes.c_uint()
            self._call("nvmlDeviceGetHandleByIndex_v2", ctypes.c_uint(index), ctypes.byref(handle))
            code = self.lib.nvmlDeviceGetMigMode(handle, ctypes.byref(current), ctypes.byref(pending))
            if code not in (0, 3):  # NVML_ERROR_NOT_SUPPORTED on non-MIG hardware.
                raise RuntimeError(self.lib.nvmlErrorString(code).decode())
            if code == 3 or not current.value:
                handles.append(handle)
                continue
            maximum = ctypes.c_uint()
            self._call("nvmlDeviceGetMaxMigDeviceCount", handle, ctypes.byref(maximum))
            for slot in range(maximum.value):
                mig = ctypes.c_void_p()
                code = self.lib.nvmlDeviceGetMigDeviceHandleByIndex(handle, ctypes.c_uint(slot), ctypes.byref(mig))
                if code == 0:
                    handles.append(mig)
                elif code != 6:  # Empty MIG slot: NVML_ERROR_NOT_FOUND.
                    raise RuntimeError(self.lib.nvmlErrorString(code).decode())
        return handles

    def sample(self):
        memory = _Memory()
        self._call("nvmlDeviceGetMemoryInfo", self.handle, ctypes.byref(memory))
        utilization = _Utilization()
        code = self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(utilization))
        return {"gpu_uuid": self.uuid, "total_bytes": memory.total,
                "free_bytes": memory.free, "used_bytes": memory.used,
                "gpu_utilization_percent": None if code else utilization.gpu,
                "utilization_unavailable": self.lib.nvmlErrorString(code).decode() if code else None}

    def close(self):
        self.lib.nvmlShutdown()


def _worker_setup(device):
    if device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


@contextmanager
def _worker_environment(device, parallel):
    values = {"CUDA_VISIBLE_DEVICES": device} if device else {}
    if parallel:
        values.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _host_resources():
    """Read Linux container CPU quota and available RAM, when exposed."""
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    result = {"cpu_capacity": cpus, "host_available_bytes": None, "host_total_bytes": None}
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        amount, period = quota.read_text().split()
        if amount != "max":
            result["cpu_capacity"] = min(cpus, max(1, ceil(int(amount) / int(period))))
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()}
        result.update(host_available_bytes=values["MemAvailable"], host_total_bytes=values["MemTotal"])
        maximum, current = Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")
        if maximum.is_file() and current.is_file():
            ceiling = maximum.read_text().strip()
            if ceiling != "max":
                result["host_total_bytes"] = min(result["host_total_bytes"], int(ceiling))
                result["host_available_bytes"] = min(result["host_available_bytes"], max(0, int(ceiling) - int(current.read_text())))
    return result


class _AdaptiveLimit:
    """Ramp on comparable completed work; retain the smaller limit at a plateau."""

    def __init__(self, settings, maximum, now):
        self.settings, self.maximum = settings, maximum
        self.limit = min(settings.initial_workers, maximum)
        self.started, self.hold_until = now, now
        self.samples, self.reference, self.stalls = {}, None, 0
        self.tasks_per_second = 0.0

    def backoff(self, now):
        self.limit = max(1, int(self.limit * 0.75))
        self.samples, self.reference, self.stalls = {}, None, 0
        self.started, self.hold_until = now, now + self.settings.recovery_seconds

    def completed(self, key, seconds, now, can_grow=True):
        total, count = self.samples.get(key, (0.0, 0))
        self.samples[key] = total + max(seconds, 1e-6), count + 1
        count = sum(value[1] for value in self.samples.values())
        elapsed = now - self.started
        self.tasks_per_second = count / max(elapsed, 1e-6)
        if (now < self.hold_until or elapsed < self.settings.ramp_seconds
                or count < self.settings.min_completions):
            return self.limit
        if self.reference is not None:
            previous_limit, previous, previous_elapsed = self.reference
            common = self.samples.keys() & previous.keys()
            # A change of phase/size mix is not evidence of a concurrency gain.
            if (sum(self.samples[key][1] for key in common) < count / 2
                    or sum(previous[key][1] for key in common) < sum(v[1] for v in previous.values()) / 2):
                return self.limit
            # Weight both windows by the same per-class cost. This compares
            # completed work per wall second, including CPU/admission stalls.
            previous_work = sum(previous[key][0] for key in common)
            current_work = sum(self.samples[key][1] * previous[key][0] / previous[key][1] for key in common)
            gain = (current_work / elapsed) / (previous_work / previous_elapsed) - 1
            if gain < self.settings.throughput_gain:
                self.stalls += 1
                if self.stalls >= 2:
                    self.limit = previous_limit
                    self.reference, self.stalls = None, 0
                    self.hold_until = now + max(self.settings.ramp_seconds * 5, self.settings.recovery_seconds)
                self.samples, self.started = {}, now
                return self.limit
        if can_grow and self.limit < self.maximum:
            self.reference = self.limit, self.samples, elapsed
            self.limit = next(step for step in (4, 8, 12, 16, 24, 32, self.maximum)
                              if step > self.limit and step <= self.maximum)
            self.samples, self.started, self.stalls = {}, now, 0
        return self.limit


def run_tasks(worker, items, *, settings=None, backend="default.qubit", memory_bytes=None,
              should_stop=None, on_submit=None, on_resources=None, retry_queue=None,
              resource_history=None, workload_key=None, on_worker_error=None):
    """Yield completed ``(item, result)`` pairs; a stop drains admitted attempts.

    ``worker`` must be a top-level, picklable function. ``on_submit`` runs in the
    parent and can create the task's immutable start marker. Workers may save
    their own completed records before returning a small result. Exceptions are
    propagated after already admitted attempts finish. In adaptive mode a lost
    process is passed to ``on_worker_error(item, exception)``: the caller saves
    the fault and owns the same bounded retry queue used for allocation errors.

    Legacy admission subtracts all active estimates from current free memory.
    Adaptive admission uses the larger of measured use and predicted pool use,
    retaining headroom without counting observed allocations twice. Comparable
    ``workload_key(item)`` groups guide the adaptive throughput ramp.
    Busy devices wait between admission checks; impossible single-task memory
    estimates fail immediately. A result with ``resource_error=True`` lowers
    concurrency and raises estimates, then drains in-flight tasks. The caller
    owns bounded retry attempts. Reported durations are concurrent wall times,
    not isolated timing baselines. Utilization checks govern admission, not a
    hard throttle on kernels that are already running.
    """
    settings = settings or ParallelSettings()
    retry_queue = deque() if retry_queue is None else retry_queue
    cpus = (_host_resources()["cpu_capacity"] if settings.adaptive else
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
    # GPU workers alternate host work with device calls. CPU quota is telemetry,
    # while measured throughput decides whether more adaptive workers help.
    workers = settings.workers if settings.adaptive and backend == "lightning.gpu" else min(settings.workers, cpus)
    if backend == "lightning.gpu" and memory_bytes is None:
        raise ValueError("GPU admission requires a memory_bytes estimate for each task.")
    probe = NvmlMemory(settings.gpu_device) if backend == "lightning.gpu" else None
    device = probe.uuid if probe else None
    active, started = {}, perf_counter()
    ramp = _AdaptiveLimit(settings, workers, started) if settings.adaptive else None
    limit, estimate_multiplier, draining, restart_pool = ramp.limit if ramp else workers, 1, False, False
    retry_idle_headroom, memory_blocked, pool_restarts = None, False, 0
    baseline_used, last_sample, can_grow, cooldown_until = None, {}, True, started
    stop = should_stop or (lambda: False)

    def resources(estimate=0):
        nonlocal retry_idle_headroom, memory_blocked, baseline_used, last_sample, can_grow
        telemetry_error = None
        try:
            sample = probe.sample() if probe else {}
            last_sample = sample
        except (RuntimeError, OSError) as error:
            if not settings.adaptive:
                raise
            sample, telemetry_error = last_sample, str(error)
        host = _host_resources() if settings.adaptive else {}
        reserved = sum(entry[1] for entry in active.values())
        safety = max(settings.reserve_mib * MIB, int((1 - settings.memory_fraction) * sample.get("total_bytes", 0)))
        if probe and telemetry_error is None:
            if baseline_used is None:
                baseline_used = sample["used_bytes"]
            capacity = sample["total_bytes"] - safety
            base = estimate / estimate_multiplier
            if base > capacity:
                raise MemoryError("The next task's estimate exceeds the GPU memory cap; use a smaller graph, larger instance or measured lower estimate.")
            estimate = min(estimate, capacity)
            if estimate and estimate_multiplier > 1:
                available = sample["free_bytes"] - safety
                if retry_idle_headroom is None and not active and not draining and available >= base:
                    retry_idle_headroom = available
                if retry_idle_headroom is not None:
                    estimate = max(base, min(estimate, retry_idle_headroom))
        if probe and telemetry_error is not None:
            memory_ok = False
        elif probe and settings.adaptive:
            predicted = max(sample["used_bytes"], baseline_used + reserved)
            memory_ok = sample["total_bytes"] - predicted - safety >= estimate
        else:
            memory_ok = not probe or sample["free_bytes"] - safety - reserved >= estimate
        memory_blocked = not memory_ok
        utilization = sample.get("gpu_utilization_percent")
        utilization_ok = utilization is None or utilization <= settings.gpu_utilization_limit
        available, total = host.get("host_available_bytes"), host.get("host_total_bytes")
        host_ok = available is None or available >= max(1024 * MIB, int(total * .10)) + settings.worker_overhead_mib * MIB
        can_grow = memory_ok and utilization_ok and host_ok and telemetry_error is None
        cooling = perf_counter() < cooldown_until
        allowed = can_grow and not cooling
        reason = ("telemetry" if telemetry_error else "recovery" if cooling else "host_memory" if not host_ok
                  else "gpu_memory" if not memory_ok else "gpu_utilization" if not utilization_ok else None)
        if on_resources:
            on_resources({**sample, **host, "active_workers": len(active), "worker_limit": limit,
                "worker_maximum": workers, "requested_workers": settings.workers,
                "tasks_per_second": ramp.tasks_per_second if ramp else None,
                "wait_reason": reason, "telemetry_error": telemetry_error,
                "active_estimate_bytes": reserved, "next_estimate_bytes": estimate,
                "reserve_bytes": safety, "memory_fraction": settings.memory_fraction,
                "gpu_utilization_limit": settings.gpu_utilization_limit,
                "waiting_for_memory": not memory_ok, "waiting_for_utilization": not utilization_ok,
                "draining_after_resource_error": draining, "estimate_multiplier": estimate_multiplier,
                "retry_idle_headroom_bytes": retry_idle_headroom, "pool_restarts": pool_restarts,
                "elapsed_wall_seconds": perf_counter() - started,
                "memory_scope": "allocated_device_measured_or_predicted" if ramp else "allocated_device_free_minus_active_estimates",
                "timing_scope": "concurrent_wall_time" if workers > 1 else "serial_wall_time"})
        return allowed

    def estimate(item):
        nonlocal limit, estimate_multiplier, retry_idle_headroom
        prior_failures = resource_history(item) if resource_history else 0
        limit = min(limit, max(1, workers - prior_failures))
        if ramp:
            ramp.limit = limit
        if 2 ** prior_failures > estimate_multiplier:
            estimate_multiplier = 2 ** prior_failures
            retry_idle_headroom = None
        return estimate_multiplier * int(memory_bytes(item)) if probe else 0

    try:
        with _worker_environment(device, workers > 1):
            if workers == 1 and probe is None:
                iterator, exhausted = iter(items), False
                while retry_queue or not exhausted:
                    if stop():
                        break
                    item = retry_queue.popleft() if retry_queue else next(iterator, None)
                    if item is None:
                        exhausted = True
                        continue
                    while not resources(estimate(item)):
                        if stop():
                            return
                        sleep(settings.poll_seconds)
                    if stop():
                        break
                    if on_submit and on_submit(item) is False:
                        continue
                    result = worker(item)
                    if isinstance(result, dict) and result.get("resource_error"):
                        estimate_multiplier *= 2
                    yield item, result
                return
            def new_pool():
                return ProcessPoolExecutor(max_workers=workers if ramp else limit, mp_context=multiprocessing.get_context("spawn"),
                                           initializer=_worker_setup, initargs=(device,))

            pool, pool_has_work = new_pool(), False
            try:
                iterator, pending, exhausted = iter(items), None, False
                while active or pending is not None or retry_queue or not exhausted:
                    while len(active) < limit and (pending is not None or retry_queue or not exhausted) and not stop() and not draining:
                        if pending is None:
                            pending = retry_queue.popleft() if retry_queue else next(iterator, None)
                            if pending is None:
                                exhausted = True
                                break
                        needed = estimate(pending)
                        if len(active) >= limit:
                            break
                        if not resources(needed):
                            break
                        if on_submit and on_submit(pending) is False:
                            pending = None
                            continue
                        try:
                            future = pool.submit(worker, pending)
                        except BrokenProcessPool as error:
                            if not ramp or on_worker_error is None:
                                raise
                            future = Future()
                            future.set_exception(error)
                        active[future] = pending, needed, perf_counter()
                        pool_has_work = True
                        pending = None
                        resources()
                    if stop():
                        exhausted = True
                    if not active:
                        if stop() or (exhausted and pending is None and not retry_queue):
                            break
                        draining = False
                        # Idle CUDA contexts can exhaust a tight memory fraction.
                        # Reclaim once; a fresh pool then waits for external demand.
                        if memory_blocked and pool_has_work:
                            pool.shutdown(wait=True)
                            pool, pool_has_work = new_pool(), False
                            pool_restarts += 1
                            retry_idle_headroom, baseline_used = None, None
                        sleep(settings.poll_seconds)
                        continue
                    ready, _ = wait(active, timeout=1.0, return_when=FIRST_COMPLETED)
                    for future in ready:
                        item, _, submitted = active.pop(future)
                        broken = False
                        try:
                            result = future.result()
                        except BrokenProcessPool as error:
                            if not ramp or on_worker_error is None:
                                raise
                            result, broken = on_worker_error(item, error), True
                        failed = broken or isinstance(result, dict) and result.get("resource_error")
                        if failed:
                            if ramp and not restart_pool:
                                ramp.backoff(perf_counter())
                                limit, cooldown_until = ramp.limit, ramp.hold_until
                            elif not ramp:
                                limit = max(1, limit - 1)
                            if not restart_pool or not ramp:
                                estimate_multiplier *= 2
                            retry_idle_headroom = None
                            draining = bool(active)
                            restart_pool = True
                        elif ramp and not restart_pool:
                            key = workload_key(item) if workload_key else "all"
                            limit = ramp.completed(key, perf_counter() - submitted, perf_counter(), can_grow)
                        yield item, result
                    if not active:
                        draining = False
                        if restart_pool:
                            pool.shutdown(wait=True)
                            pool, pool_has_work = new_pool(), False
                            pool_restarts += 1
                            restart_pool = False
                            baseline_used = None
                    resources()
            finally:
                pool.shutdown(wait=True)
    finally:
        if probe:
            probe.close()
