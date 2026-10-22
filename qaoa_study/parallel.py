"""Bounded independent-task execution with optional GPU/MIG memory admission.

Workers run whole attempts and keep their own CUDA contexts. This module does
not split a statevector, alter task seeds, retry failures, or manage checkpoints.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from collections import deque
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import multiprocessing
import os
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

    def __post_init__(self):
        if (type(self.workers) is not int or self.workers < 1
                or type(self.state_copies) is not int or self.state_copies < 1
                or self.reserve_mib < 0 or self.worker_overhead_mib < 0
                or not 0 < self.memory_fraction <= 1
                or not 0 <= self.gpu_utilization_limit <= 100 or self.poll_seconds <= 0):
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


def run_tasks(worker, items, *, settings=None, backend="default.qubit", memory_bytes=None,
              should_stop=None, on_submit=None, on_resources=None, retry_queue=None,
              resource_history=None):
    """Yield completed ``(item, result)`` pairs; a stop drains admitted attempts.

    ``worker`` must be a top-level, picklable function. ``on_submit`` runs in the
    parent and can create the task's immutable start marker. Workers may save
    their own completed records before returning a small result. Exceptions are
    propagated after already admitted attempts finish; no task is resubmitted.

    GPU admission subtracts estimates for all active tasks from currently free
    memory. This conservative accounting can count already allocated memory
    twice. It accommodates tasks that have started but have not allocated yet.
    Busy devices wait between admission checks; impossible single-task memory
    estimates fail immediately. A result with ``resource_error=True`` lowers
    concurrency and raises estimates, then drains in-flight tasks. The caller
    owns bounded retry attempts. Reported durations are concurrent wall times,
    not isolated timing baselines. Utilization checks govern admission, not a
    hard throttle on kernels that are already running.
    """
    settings = settings or ParallelSettings()
    retry_queue = deque() if retry_queue is None else retry_queue
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    workers = min(settings.workers, cpus)
    if backend == "lightning.gpu" and memory_bytes is None:
        raise ValueError("GPU admission requires a memory_bytes estimate for each task.")
    probe = NvmlMemory(settings.gpu_device) if backend == "lightning.gpu" else None
    device = probe.uuid if probe else None
    active, started = {}, perf_counter()
    limit, estimate_multiplier, draining, restart_pool = workers, 1, False, False
    retry_idle_headroom, memory_blocked, pool_restarts = None, False, 0
    stop = should_stop or (lambda: False)

    def resources(estimate=0):
        nonlocal retry_idle_headroom, memory_blocked
        sample = probe.sample() if probe else {}
        reserved = sum(entry[1] for entry in active.values())
        safety = max(settings.reserve_mib * MIB, int((1 - settings.memory_fraction) * sample.get("total_bytes", 0)))
        if probe:
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
        memory_ok = not probe or sample["free_bytes"] - safety - reserved >= estimate
        memory_blocked = not memory_ok
        utilization = sample.get("gpu_utilization_percent")
        utilization_ok = utilization is None or utilization <= settings.gpu_utilization_limit
        allowed = memory_ok and utilization_ok
        if on_resources:
            on_resources({**sample, "active_workers": len(active), "worker_limit": limit,
                "active_estimate_bytes": reserved, "next_estimate_bytes": estimate,
                "reserve_bytes": safety, "memory_fraction": settings.memory_fraction,
                "gpu_utilization_limit": settings.gpu_utilization_limit,
                "waiting_for_memory": not memory_ok, "waiting_for_utilization": not utilization_ok,
                "draining_after_resource_error": draining, "estimate_multiplier": estimate_multiplier,
                "retry_idle_headroom_bytes": retry_idle_headroom, "pool_restarts": pool_restarts,
                "elapsed_wall_seconds": perf_counter() - started,
                "memory_scope": "allocated_device_free_minus_active_estimates",
                "timing_scope": "concurrent_wall_time" if workers > 1 else "serial_wall_time"})
        return allowed

    def estimate(item):
        nonlocal limit, estimate_multiplier, retry_idle_headroom
        prior_failures = resource_history(item) if resource_history else 0
        limit = min(limit, max(1, workers - prior_failures))
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
                return ProcessPoolExecutor(max_workers=limit, mp_context=multiprocessing.get_context("spawn"),
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
                        active[pool.submit(worker, pending)] = pending, needed
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
                            retry_idle_headroom = None
                        sleep(settings.poll_seconds)
                        continue
                    ready, _ = wait(active, timeout=1.0, return_when=FIRST_COMPLETED)
                    for future in ready:
                        item, _ = active.pop(future)
                        result = future.result()
                        if isinstance(result, dict) and result.get("resource_error"):
                            limit = max(1, limit - 1)
                            estimate_multiplier *= 2
                            retry_idle_headroom = None
                            draining = bool(active)
                            restart_pool = True
                        yield item, result
                    if not active:
                        draining = False
                        if restart_pool:
                            pool.shutdown(wait=True)
                            pool, pool_has_work = new_pool(), False
                            pool_restarts += 1
                            restart_pool = False
                    resources()
            finally:
                pool.shutdown(wait=True)
    finally:
        if probe:
            probe.close()
