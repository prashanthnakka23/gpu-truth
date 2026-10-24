# gpu-truth

A Prometheus and Grafana stack that answers a question `nvidia-smi` cannot: **how much of the GPU is actually working?**

Built by [Prashanth Nakka](https://www.linkedin.com/in/prashanthnakka) — 5 years running production infrastructure, now pointed at GPU fleets.

---

## The problem

`utilization.gpu` — the number on every GPU dashboard — is the share of time during which **at least one kernel was resident on the device**.

At least one. An A100 has 108 streaming multiprocessors. A single kernel occupying a single SM for the whole sampling window reports **100%**. The other 107 sit idle. The graph is identical either way.

This gets expensive in a specific, repeatable way:

```
utilization pinned at 100% all week
        │
        ▼
"we're out of capacity"  ──▶  purchase request  ──▶  new GPUs arrive
        │
        ▼
graph looks exactly the same, because the ceiling was memory bandwidth
```

It matters most for LLM serving, where decode is memory-bandwidth bound rather than compute bound — and `utilization.gpu` cannot tell those two states apart.

## What this does

Scrapes DCGM profiling fields, derives the signals that actually describe the card, and alerts on the gap between what the dashboard says and what the silicon is doing.

```
dcgm-exporter ──▶ Prometheus ──▶ recording rules ──▶ Grafana
      │                               │
   (or mock)                   alerting rules
```

### The signals

| Metric | What it answers |
| --- | --- |
| `gpu:sm_active:ratio` | How many multiprocessors are genuinely busy, not whether any of them is |
| `gpu:sm_occupancy:ratio` | Resident warps against the maximum — launch efficiency |
| `gpu:tensor_active:ratio` | Is the fast path being used, or are you paying for silicon nobody touches |
| `gpu:dram_active:ratio` | Memory bandwidth, which at decode time is usually the real wall |
| `gpu:nvlink_bytes:rate5m` | The interconnect, once the model stops fitting on one card |
| `gpu:illusion_gap:ratio` | `utilization.gpu` minus SM activity — how far the headline number is from reality |

### The alerts

The reason this is a project and not a dashboard screenshot.

| Alert | Fires when | Why you care |
| --- | --- | --- |
| `GPUBusyButIdle` | utilization > 90% **and** SM active < 30% | Somebody is about to buy hardware off a misleading graph |
| `GPUMemoryBandwidthBound` | DRAM > 70% **and** SM active < 40% | More compute will not help. Look at quantisation and batching |
| `TensorCoresUnused` | SM active > 50% **and** tensor pipe < 5% | Usually an fp32 code path or an unfused kernel |
| `GPUMetricsMissing` | profiling fields absent | A dashboard with no data looks identical to one with good data |

## Running it

**No GPU?** The mock exporter emits the real DCGM metric names with realistic shapes — three cards telling three different stories (the illusion, bandwidth bound, genuinely busy). Everything below works on a laptop.

```bash
docker compose --profile demo up -d
```

**Real hardware** — needs the NVIDIA container runtime on the host:

```bash
docker compose --profile gpu up -d
```

Grafana on `localhost:3000`, Prometheus on `localhost:9090`. The dashboard provisions itself; no import step.

> `dcgm-exporter` must run with the profiling field set. The default collector file does **not** include `DCGM_FI_PROF_*`, and without those fields every panel here is empty while the stack still looks healthy.

## Plan

Two weeks, not four. I would rather finish something small.

| Week | Dates | Goal |
| --- | --- | --- |
| 1 | Oct 24–31 | Stack running, rules firing, dashboard provisioned, mock exporter reproducing the illusion |
| 2 | Nov 1–7 | Run it against a real GPU, capture a live vLLM decode workload, write up what the numbers actually showed |

## Open questions

Things I expect to get wrong:

- What is a sensible `GPUBusyButIdle` threshold on a real serving fleet, rather than one I picked because it reads well?
- Does SM occupancy add anything over SM activity in practice, or is it a second number saying the same thing?
- During LLM decode, how close to the theoretical memory bandwidth ceiling does `DCGM_FI_PROF_DRAM_ACTIVE` actually get?
- Is `illusion_gap` a real diagnostic or just a nice chart?

Answers go in `docs/` as I find them.

## Status

Week 1. Stack and rules written, mock exporter verified, real-hardware run outstanding.
