"""Synthetic DCGM exporter.

dcgm-exporter needs an NVIDIA GPU. This emits the same metric names with the
same semantics so the dashboards, recording rules and alerts can be developed,
demonstrated and tested on any machine.

Three GPUs, three stories:

  gpu 0  the illusion     utilization pinned near 100%, SMs barely working
  gpu 1  bandwidth bound  SMs moderate, DRAM saturated   (LLM decode)
  gpu 2  genuinely busy   SMs and tensor cores both high (training)

Point Prometheus at :9400/metrics exactly as you would at the real thing.
"""

import math
import os
import random
import time

from prometheus_client import Gauge, start_http_server

LABELS = ("gpu", "UUID", "modelName", "Hostname")

GPU_UTIL = Gauge("DCGM_FI_DEV_GPU_UTIL", "GPU utilization (%)", LABELS)
SM_ACTIVE = Gauge("DCGM_FI_PROF_SM_ACTIVE", "Fraction of time at least one warp was resident on an SM", LABELS)
SM_OCCUPANCY = Gauge("DCGM_FI_PROF_SM_OCCUPANCY", "Fraction of resident warps against the maximum", LABELS)
TENSOR_ACTIVE = Gauge("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "Fraction of cycles the tensor pipe was active", LABELS)
DRAM_ACTIVE = Gauge("DCGM_FI_PROF_DRAM_ACTIVE", "Fraction of cycles the memory interface was active", LABELS)
FB_USED = Gauge("DCGM_FI_DEV_FB_USED", "Framebuffer memory used (MiB)", LABELS)
FB_FREE = Gauge("DCGM_FI_DEV_FB_FREE", "Framebuffer memory free (MiB)", LABELS)
POWER = Gauge("DCGM_FI_DEV_POWER_USAGE", "Power draw (W)", LABELS)
TEMP = Gauge("DCGM_FI_DEV_GPU_TEMP", "GPU temperature (C)", LABELS)
# Gauges, not Counters: prometheus_client appends _total to a Counter, and
# dcgm-exporter publishes these as bare cumulative gauges. The recording rules
# take rate() over them either way.
NVLINK_TX = Gauge("DCGM_FI_PROF_NVLINK_TX_BYTES", "NVLink bytes transmitted (cumulative)", LABELS)
NVLINK_RX = Gauge("DCGM_FI_PROF_NVLINK_RX_BYTES", "NVLink bytes received (cumulative)", LABELS)

HOST = os.environ.get("MOCK_HOSTNAME", "mock-node-0")
FB_TOTAL = 81920  # MiB, an 80GB card

# name, sm_active, sm_occupancy, tensor_active, dram_active, mem_ratio, power
PROFILES = [
    ("the illusion",    0.08, 0.06, 0.01, 0.12, 0.22, 95.0),
    ("bandwidth bound", 0.34, 0.29, 0.21, 0.88, 0.74, 290.0),
    ("genuinely busy",  0.86, 0.71, 0.63, 0.55, 0.91, 390.0),
]


def wobble(base, pct=0.12):
    """Jitter a value without letting it leave [0, 1]."""
    v = base * (1 + random.uniform(-pct, pct))
    return max(0.0, min(1.0, v))


def main():
    port = int(os.environ.get("MOCK_PORT", "9400"))
    start_http_server(port)
    print(f"mock dcgm-exporter on :{port}/metrics  ({len(PROFILES)} GPUs)", flush=True)

    t = 0.0
    while True:
        for idx, (name, sm, occ, tensor, dram, mem, watts) in enumerate(PROFILES):
            labels = (str(idx), f"GPU-mock-{idx:08d}", "Mock A100-SXM4-80GB", HOST)

            # A slow swell so the graphs move, rather than flat lines.
            swell = 1 + 0.08 * math.sin(t / 37 + idx)

            sm_a = wobble(sm * swell)
            SM_ACTIVE.labels(*labels).set(sm_a)
            SM_OCCUPANCY.labels(*labels).set(wobble(occ * swell))
            TENSOR_ACTIVE.labels(*labels).set(wobble(tensor * swell))
            DRAM_ACTIVE.labels(*labels).set(wobble(dram * swell))

            # The whole point: utilization.gpu is near 100% whenever any kernel
            # is resident, which is almost always true even when SMs are idle.
            util = 100.0 if sm_a > 0.02 else random.uniform(0, 8)
            GPU_UTIL.labels(*labels).set(min(100.0, util * random.uniform(0.97, 1.0)))

            used = FB_TOTAL * wobble(mem, 0.03)
            FB_USED.labels(*labels).set(round(used))
            FB_FREE.labels(*labels).set(round(FB_TOTAL - used))
            POWER.labels(*labels).set(round(watts * random.uniform(0.94, 1.06), 1))
            TEMP.labels(*labels).set(round(34 + 44 * sm_a + random.uniform(-2, 2), 1))

            # Only the multi-GPU training job moves real traffic over NVLink.
            nv = 1.8e9 if idx == 2 else 4.0e6
            NVLINK_TX.labels(*labels).inc(nv * random.uniform(0.8, 1.2))
            NVLINK_RX.labels(*labels).inc(nv * random.uniform(0.8, 1.2))

        t += 5
        time.sleep(5)


if __name__ == "__main__":
    main()
