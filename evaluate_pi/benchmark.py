"""Measure synchronous host-visible inference with input ready in RAM."""
import platform
import time
import numpy as np


def benchmark(adapter, signals, warmup=50, runs=1000):
    signals = np.asarray(signals)
    if (warmup < 0 or runs <= 0 or signals.ndim != 3 or len(signals) == 0
            or signals.shape[1] != 1 or signals.shape[2] not in (300, 400)):
        raise ValueError('Expected non-empty [N,1,300/400], warmup >=0 and runs >0.')
    # All array copies/conversions are outside the timer.
    inputs = [np.ascontiguousarray(signals[i:i+1], dtype=np.float32) for i in range(len(signals))]
    for i in range(warmup):
        adapter.predict(inputs[i % len(inputs)])
    elapsed = []
    for i in range(runs):
        x = inputs[i % len(inputs)]
        start = time.perf_counter_ns()
        adapter.predict(x)
        elapsed.append((time.perf_counter_ns()-start) / 1e6)
    values = np.asarray(elapsed)
    mean = float(values.mean())
    return dict(mean_ms_per_sample=mean, median_ms_per_sample=float(np.median(values)),
                p95_ms_per_sample=float(np.percentile(values, 95)), std_ms_per_sample=float(values.std()),
                samples_per_second=1000/mean if mean else None, batch_size=1, warmup=warmup, runs=runs,
                timer='perf_counter_ns', scope='adapter.predict: host-visible synchronous inference; no file I/O or metrics',
                environment=dict(platform=platform.platform(), machine=platform.machine(),
                                 processor=platform.processor(), python=platform.python_version())), elapsed
