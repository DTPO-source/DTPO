# DTPO

**Digital Twin for 5G AI-RAN: Sim-to-Real Gap Measurement and Optimization**

DTPO measures and reduces the sim-to-real gap between a live 5G O-RAN
deployment and a digital twin that combines NVIDIA Aerial Omniverse Digital
Twin (AODT) ray-tracing channels with a full-stack O-RAN system. Although exact
instantaneous channel reconstruction remains difficult, the calibrated twin can
closely reproduce network behavior and support the transfer of AI models and
control policies to the real deployment.

This repository contains the project webpage, outdoor measurements, AODT
scenario inputs, and the reference implementation of the DTPO calibration and
optimization workflow.

## Repository Structure

```text
.
├── index.html                              # Project webpage
├── assets/
├── data_outdoor/
├── senario/
│   ├── scenario.yaml                       # AODT simulation configuration
│   ├── vegetation_dense.geojson            # Vegetation geometry
│   ├── waypoints_24_49.gpx                 # Source UE waypoints
│   └── ue_trajectory_1p5s.gpx              # Resampled UE trajectory
└── source_code/
    ├── DTPO.py                             # Optimizer and fidelity objective
    ├── ebstr_calibration_pipeline.py       # Measurement/asset preparation
    ├── ebstr_bo_calibration.yml            # Experiment configuration
    ├── run_ebstr_experiment.sh             # Candidate evaluation runner
    └── README.md                           # Source-code notes
```

### `source_code/`

`DTPO.py` implements the context-aware optimal-transport objective, Sinkhorn
solver, Sobol initialization, TuRBO search, repeated evaluation, checkpointing,
and result export. The calibration pipeline prepares EBSTR measurement inputs
and AODT assets from CSI, KPI, GPX, and scene data. The YAML file defines
site-specific parameter bounds and optimizer settings, while the shell script
runs one isolated AODT-to-O-RAN evaluation.

### `data_outdoor/`

`data_outdoor/` contains CSI snapshots, gNB metrics, GPX traces, and the CSI
collection script. The stationary and mobile folders pair radio observations
with route data, while `data_UE/` provides an additional UE capture set for the
real-data reference.

### `senario/`

The directory name follows the spelling currently used in the repository. Its
files define the site and UE motion used for AODT evaluation:

| File | Purpose |
| --- | --- |
| `scenario.yaml` | Configures database and Parquet export, GIS assets, DU/RU/UE placement, antenna panels, radio parameters, and the AODT simulation timeline. |
| `vegetation_dense.geojson` | Provides 38 visualization tree locations aligned with the electromagnetic vegetation layout. |
| `waypoints_24_49.gpx` | Stores the source UE waypoint sequence, including interpolated points where the original route record was incomplete. |
| `ue_trajectory_1p5s.gpx` | Densifies the waypoint route at 1.5-second intervals for the AODT UE trajectory. |

The paths inside `scenario.yaml` follow the target AODT deployment layout (for
example, `demo_gis/demo/`). The files in this directory are the repository-side
inputs that must be placed or mapped into that environment before simulation.

## DTPO Method

DTPO calibrates material, vegetation, receiver-noise, and RF parameters through
a closed loop:

1. Collect real CSI, KPIs, UE positions, and timestamps.
2. Apply candidate parameters in AODT and inject the generated CIR/CFR into the
   O-RAN stack.
3. Score channel and KPI fidelity with context-aware, entropically regularized
   optimal transport under spatial and temporal constraints.
4. Update the parameters using Sobol initialization followed by TuRBO, averaging
   repeated end-to-end evaluations.

```text
Real CSI/KPI + position/time
            │
            ▼
Fixed normalization and local groups
            │
            ▼
Sobol / TuRBO proposes parameters
            │
            ▼
AODT generates CIR/CFR
            │
            ▼
O-RAN produces simulated CSI/KPIs
            │
            ▼
Context-aware transport objective
            │
            └──────────── feedback to TuRBO
```

The real-measurement CSV defines fixed normalization statistics and reference
groups. Each candidate evaluation must write `simulation_metrics.csv` with the
channel, KPI, position, time, and grouping fields required by `DTPO.py`.
Stationary samples are grouped by location, while mobile samples use short
contiguous trajectory windows.

## Requirements

The optimizer requires:

```text
matplotlib
numpy
pandas
PyYAML
SciPy
```

The calibration pipeline additionally requires OmegaConf and the AODT client
modules referenced by `ebstr_calibration_pipeline.py`. The complete end-to-end
runner assumes an existing AODT/O-RAN environment and deployment-specific radio
configurations.

## Running DTPO

Run the optimizer with the example configuration:

```bash
python3 source_code/DTPO.py --config source_code/ebstr_bo_calibration.yml
```

To override the output directory or evaluation budget:

```bash
python3 source_code/DTPO.py \
  --config source_code/ebstr_bo_calibration.yml \
  --output-dir data/bo_calibration/EBSTR/20mhz/stationary \
  --configurations 160
```

The optimizer exports iteration histories, repeated-run objectives, convergence
figures, the best simulated metrics, calibrated assets, and
`best_parameters.json` to the configured output directory. Deployment-specific
AODT/O-RAN components are not bundled as a standalone environment in this
repository.

## Project Webpage

The webpage presents the real-world 100 MHz and AODT channel measurement demos,
the DTPO system overview, the experimental area, and the measurement map.
Preview it locally with:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000/`, or visit the hosted
[DTPO project page](https://dtpo-source.github.io/DTPO/).

## Citation

Citation details will be added upon publication.
