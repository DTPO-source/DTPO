# DTPO

**Digital Twin for 5G AI-RAN: Sim-to-Real Gap Measurement and Optimization**

[Project Page](https://dtpo-source.github.io/DTPO/)

## Overview

DTPO studies and reduces the sim-to-real gap between a live 5G O-RAN deployment
and its digital twin. The digital twin couples map-based ray-tracing channels
from NVIDIA Aerial Omniverse Digital Twin (AODT) with a full-stack O-RAN system,
allowing channel behavior and end-to-end network KPIs to be compared under
aligned radio configurations and UE trajectories.

The project focuses on two questions:

1. How accurately does a cellular digital twin reproduce real channel and
   network behavior?
2. How can uncertain environmental and radio parameters be calibrated when each
   end-to-end digital-twin evaluation is expensive?

The paper shows that exact instantaneous channel reconstruction is difficult,
but a calibrated twin can closely reproduce network-level behavior and support
the transfer of AI models and control policies to the real deployment.

## DTPO Method

DTPO optimizes uncertain material, vegetation, receiver-noise, and RF parameters
through a closed loop:

1. **Collect real measurements.** Record UE position and time, channel
   coefficients, and network KPIs including RSRP, UL/DL throughput, CQI, PUSCH
   SINR, and UL/DL BLER.
2. **Generate the digital-twin response.** Apply a candidate parameter vector to
   AODT, generate site-specific CIR/CFR traces, and inject the channel into the
   O-RAN network stack.
3. **Measure context-aware fidelity.** Normalize channel and KPI dimensions with
   robust statistics computed from real measurements, build fixed local
   measurement groups, and use entropically regularized optimal transport to
   tolerate spatial and temporal misalignment.
4. **Update the parameters.** Use Sobol initialization followed by TuRBO with an
   ARD Matérn-5/2 Gaussian-process surrogate and Thompson sampling. Repeated
   end-to-end evaluations are averaged before the trust region is updated.

The transport objective jointly evaluates physical-layer channel fidelity and
network-level KPI fidelity while penalizing spatially or temporally implausible
matches. Stationary samples are grouped by location; mobile samples are grouped
into short contiguous trajectory windows.

## Source Code

| Path | Purpose |
| --- | --- |
| `source/DTPO.py` | Context-aware optimal-transport objective, Sinkhorn solver, Sobol initialization, TuRBO search, repeated evaluation, checkpointing, and result export. |
| `source/ebstr_calibration_pipeline.py` | Prepares EBSTR measurement inputs and AODT calibration assets from CSI, KPI, GPX, and scene data. |
| `source/ebstr_bo_calibration.yml` | Example site-specific parameter bounds, measurement paths, optimizer settings, and evaluation command. |
| `source/run_ebstr_experiment.sh` | Runs one isolated AODT-to-O-RAN evaluation and exports simulated channel/KPI metrics. |
| `data_outdoor/csi_aodt.py` | Receives, visualizes, and stores outdoor CSI snapshots. |
| `data_outdoor/` | Outdoor stationary/mobile GPX traces, CSI snapshots, and gNB metric logs used during data collection. |

## Optimization Workflow

```text
Real CSI/KPI + position/time
            │
            ▼
Fixed normalization and local groups
            │
            ▼
Sobol / TuRBO proposes normalized parameters
            │
            ▼
AODT generates site-specific CIR/CFR
            │
            ▼
O-RAN stack produces simulated CSI/KPIs
            │
            ▼
Context-aware Sinkhorn transport objective
            │
            └──────────── feedback to TuRBO
```

Each evaluation command must write a `simulation_metrics.csv` file containing
the channel, KPI, position, time, and grouping fields required by
`source/DTPO.py`. The real-measurement CSV provides the fixed normalization
statistics and reference groups.

## Running DTPO

Install the Python dependencies used by the optimizer and data pipeline, then
run:

```bash
python3 source/DTPO.py --config source/ebstr_bo_calibration.yml
```

Useful options:

```bash
python3 source/DTPO.py \
  --config source/ebstr_bo_calibration.yml \
  --output-dir data/bo_calibration/EBSTR/20mhz/stationary \
  --configurations 160
```

The optimizer writes iteration histories, repeated-run objectives, convergence
figures, the best simulated metrics, calibrated assets, and
`best_parameters.json` to the configured output directory.

The end-to-end experiment runner assumes an existing AODT/O-RAN environment,
radio configurations, prepared measurement CSVs, and the external modules
referenced by the YAML and shell scripts. Those deployment-specific components
are not bundled as a standalone environment in this repository.

## Project Webpage

The webpage contains the real-world 100 MHz channel measurement demo, the DTPO
evaluation demo, the system overview, the experimental area, and the measurement
map. Preview it locally with:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/`, or visit the hosted
[DTPO project page](https://dtpo-source.github.io/DTPO/).

## Repository Structure

```text
.
├── index.html                         # DTPO project webpage
├── assets/
│   ├── images/                        # System, deployment, and map figures
│   └── videos/                        # Browser-compatible project demos
├── data_outdoor/                      # Outdoor CSI, KPI, and GPX measurements
└── source/
    ├── DTPO.py                        # DTPO optimizer and fidelity objective
    ├── ebstr_calibration_pipeline.py  # Measurement/asset preparation
    ├── ebstr_bo_calibration.yml       # Example experiment configuration
    └── run_ebstr_experiment.sh        # One end-to-end candidate evaluation
```
