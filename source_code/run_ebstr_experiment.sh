#!/usr/bin/env bash
# One isolated physical/digital-twin evaluation. It uploads trial assets,
# generates a deterministic AODT CIR, then runs the current IPC radio chain.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${BO_ITERATION_DIR:?missing BO iteration workspace}"
: "${BO_REAL_METRICS_CSV:?missing real KPI reference}"
: "${BO_INPUT_BUILDING_MATERIALS_CALIBRATED:?missing patched building assets}"
: "${BO_INPUT_VEGETATION_VEG_MATERIALS_CALIBRATED:?missing patched vegetation assets}"
: "${BO_AODT_BANDWIDTH_MHZ:?missing BO_AODT_BANDWIDTH_MHZ}"

case "$BO_AODT_BANDWIDTH_MHZ" in
  20|20.0)
    export GNB_CONFIG_FILE="$PWD/calibration/radio/gnb_oai_20mhz.yaml"
    export UE_CONFIG_FILE="$PWD/calibration/radio/oaiue_20mhz.conf"
    export BRIDGE_SAMPLE_RATE_HZ=23040000
    ;;
  100|100.0)
    export GNB_CONFIG_FILE="$PWD/gnb_oai.yaml"
    export UE_CONFIG_FILE="$PWD/oaiue_zmq.conf"
    export BRIDGE_SAMPLE_RATE_HZ=122880000
    ;;
  *)
    echo "Unsupported BO radio bandwidth: $BO_AODT_BANDWIDTH_MHZ MHz" >&2
    exit 2
    ;;
esac
[[ -s "$GNB_CONFIG_FILE" ]] || { echo "Missing BO gNB config: $GNB_CONFIG_FILE" >&2; exit 2; }
[[ -s "$UE_CONFIG_FILE" ]] || { echo "Missing BO UE config: $UE_CONFIG_FILE" >&2; exit 2; }
export LOGDIR="$BO_ITERATION_DIR/logs"
echo "[bo] radio profile: ${BO_AODT_BANDWIDTH_MHZ} MHz; gNB=$GNB_CONFIG_FILE; UE=$UE_CONFIG_FILE; sample_rate=$BRIDGE_SAMPLE_RATE_HZ"

export BO_GNB_METRICS_FILE="$BO_ITERATION_DIR/gnb_metrics.jsonl"
BASE_YAML="${BO_AODT_BASE_YAML:-aodt_source/ebstr/scenario.yaml}"
export BRIDGE_CHANNEL_MODE=cir
export BRIDGE_CIR_FILE="$BO_ITERATION_DIR/channel_CIR.dat"
export BRIDGE_SAVE_APPLIED_TAPS="$BO_ITERATION_DIR/applied_cir_taps.npz"
# Dataset normalization intentionally removes absolute AODT gain so every
# material trial remains numerically usable by OAI. Re-introduce the BO RU
# power as a gain relative to the official calibrated baseline.
export BRIDGE_PATH_LOSS_DB
BRIDGE_PATH_LOSS_DB="$(python3 -c 'import sys; print(40.87192189948769-float(sys.argv[1]))' "$BO_RU_TRANSMIT_POWER_DBM")"
export BRIDGE_CIR_INTERVAL="${BO_CIR_INTERVAL:-3.75}"
export BRIDGE_STOP_WHEN_CIR_EXHAUSTED=off
export BRIDGE_STEP_TRIGGER_FILE="$BO_ITERATION_DIR/start_channel_variation"
rm -f "$BRIDGE_STEP_TRIGGER_FILE"
TRIAL_YAML="$BO_ITERATION_DIR/aodt_trial.yml"
gpx_args=()
if [[ -n "${BO_AODT_GPX_FILE:-}" ]]; then
  gpx_args+=(--gpx "$BO_AODT_GPX_FILE" \
    --timesteps "${BO_AODT_TIMESTEPS:?missing BO_AODT_TIMESTEPS}" \
    --step-seconds "$BRIDGE_CIR_INTERVAL")
fi
bandwidth_args=()
if [[ -n "${BO_AODT_BANDWIDTH_MHZ:-}" ]]; then
  bandwidth_args+=(--bandwidth-mhz "$BO_AODT_BANDWIDTH_MHZ")
  if [[ -n "${BO_AODT_FFT_SIZE:-}" ]]; then
    bandwidth_args+=(--fft-size "$BO_AODT_FFT_SIZE")
  fi
fi
ru_position_args=()
if [[ -n "${BO_AODT_RU_LAT:-}" || -n "${BO_AODT_RU_LON:-}" ]]; then
  : "${BO_AODT_RU_LAT:?missing BO_AODT_RU_LAT}"
  : "${BO_AODT_RU_LON:?missing BO_AODT_RU_LON}"
  ru_position_args+=(--ru-lat "$BO_AODT_RU_LAT" --ru-lon "$BO_AODT_RU_LON")
fi
python3 -m bo_calibration.prepare_aodt_trial --base-yaml "$BASE_YAML" \
  --building-materials "$BO_INPUT_BUILDING_MATERIALS_CALIBRATED" \
  --vegetation-materials "$BO_INPUT_VEGETATION_VEG_MATERIALS_CALIBRATED" \
  --ru-power-dbm "$BO_RU_TRANSMIT_POWER_DBM" --output-yaml "$TRIAL_YAML" \
  --prefix "demo_gis/EBSTR/bo/${BO_PHASE}/$(basename "$BO_ITERATION_DIR")" \
  "${gpx_args[@]}" "${bandwidth_args[@]}" "${ru_position_args[@]}"
PYTHONPATH=aodt_source/client/build:aodt_source/client/build/config \
  aodt_source/.venv/bin/python3 aodt_source/client/examples/ebstr_bo_export_cir.py \
  --yaml "$TRIAL_YAML" --cir "$BRIDGE_CIR_FILE" \
  --cfr "$BO_ITERATION_DIR/channel_CFR.dat" --log-file "$BO_ITERATION_DIR/aodt.log"

# Replay the DL and UL throughput observed during real full-buffer collection.
if [[ -n "${BO_RUN_SIM_COMMAND:-}" ]]; then
  eval "$BO_RUN_SIM_COMMAND"
else
  : "${BO_TRAFFIC_PROFILE_CSV:?missing BO_TRAFFIC_PROFILE_CSV}"
  GNB_METRICS_FILE="$BO_GNB_METRICS_FILE" \
    TRAFFIC_MODE=udp-replay TRAFFIC_DIRECTION=both \
    TRAFFIC_PROFILE_CSV="$BO_TRAFFIC_PROFILE_CSV" \
    UDP_REPLAY_DL_RECEIVER_JSON="$BO_ITERATION_DIR/udp_replay_dl_receiver.json" \
    UDP_REPLAY_UL_RECEIVER_JSON="$BO_ITERATION_DIR/udp_replay_ul_receiver.json" \
    UE_SETTLE_SECONDS=2 ./run_sim
fi
[[ -s "$BO_GNB_METRICS_FILE" ]] || { echo "No gNB metrics were captured" >&2; exit 2; }
aodt_source/.venv/bin/python3 -m bo_calibration.aggregate_gnb_metrics --input "$BO_GNB_METRICS_FILE" \
  --reference "$BO_REAL_METRICS_CSV" --output "$BO_ITERATION_DIR/simulation_metrics.csv" \
  --cfr "$BO_ITERATION_DIR/channel_CFR.dat"
