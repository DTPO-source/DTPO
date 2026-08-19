#!/usr/bin/env python3
"""End-to-end NVIDIA AODT v1.5 calibration pipeline for the EBSTR campaign.

The pipeline follows NVIDIA's ``example_calibration.py`` workflow: run a base
simulation with full ray-path export, load the same scenario with calibration
measurements, call ``run_calibration()``, and run the generated calibrated
scenario. All four 20/100 MHz x stationary/mobile datasets are supported. Every measured RSRP
row is paired one-to-one with a GPX point before it is sent to AODT.

Run from the client environment::

    PYTHONPATH=client/build:client/build/config .venv/bin/python3 \
      client/examples/ebstr_calibration_pipeline.py \
      --dataset 20mhz_stationary

Replace ``20mhz_stationary`` with ``20mhz_mobile``, ``100mhz_stationary``,
or ``100mhz_mobile`` for the other datasets.

Use ``--prepare-only`` to inspect the aligned GPX/CSV and field audit without
contacting the AODT server.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import dt_client
from _config import DiffusionModel, SimConfig

import aodt_one_ue_CIR as ebstr_cir


GPX_NS = "http://www.topografix.com/GPX/1/1"
NVIDIA_POWER_COLUMN = "Power_PCI_1"


def load_processing_module(project_root: Path, mobility: str):
    path = project_root / f"data/data_process/process_{mobility}_outdoor.py"
    spec = importlib.util.spec_from_file_location(f"process_{mobility}_outdoor", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scenario_definition(project_root: Path, dataset: str):
    import sys
    root = str(project_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from calibration.catalog import load_catalog
    return load_catalog()[dataset]


def real_data_dir(project_root: Path, dataset: str) -> Path:
    data_dir = scenario_definition(project_root, dataset).dataset_dir
    if not (data_dir / "gnb_metrics.jsonl").is_file():
        raise RuntimeError(f"{dataset} dataset not found: {data_dir}")
    return data_dir


def parse_gpx_with_elevation(path: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    root = ET.parse(path).getroot()
    ns = {"g": GPX_NS}
    for point in root.findall(".//g:trkpt", ns):
        time_element = point.find("g:time", ns)
        if time_element is None or not time_element.text:
            continue
        elevation = point.find("g:ele", ns)
        points.append(
            {
                "time": datetime.fromisoformat(time_element.text.replace("Z", "+00:00")).replace(tzinfo=None),
                "lat": float(point.attrib["lat"]),
                "lon": float(point.attrib["lon"]),
                "ele": float(elevation.text) if elevation is not None and elevation.text else 0.0,
            }
        )
    if len(points) < 2:
        raise RuntimeError(f"GPX needs at least two timed track points: {path}")
    return points


def interpolate_gpx(points: list[dict[str, Any]], when: datetime) -> tuple[float, float, float]:
    if when <= points[0]["time"]:
        p = points[0]
        return p["lat"], p["lon"], p["ele"]
    if when >= points[-1]["time"]:
        p = points[-1]
        return p["lat"], p["lon"], p["ele"]
    for left, right in zip(points, points[1:]):
        if left["time"] <= when <= right["time"]:
            span = (right["time"] - left["time"]).total_seconds()
            fraction = 0.0 if span <= 0 else (when - left["time"]).total_seconds() / span
            return tuple(
                float(left[name] + fraction * (right[name] - left[name]))
                for name in ("lat", "lon", "ele")
            )
    raise AssertionError("unreachable GPX interpolation state")


def prepare_stationary_measurements(project_root: Path, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use the stationary burst detector, then audit and align the raw sources."""
    processing = load_processing_module(project_root, "stationary")
    data_dir = real_data_dir(project_root, dataset)
    csi = processing.load_all_csi(data_dir)
    metrics = processing.load_gnb_metrics(data_dir / "gnb_metrics.jsonl")
    gpx_path = project_root / "data/GPX/Jul_19_213850.gpx"
    track, _ = processing.parse_gpx(gpx_path)
    points, skipped = processing.build_points(csi, metrics, track)
    source_track = parse_gpx_with_elevation(gpx_path)

    csi_start = csi["time"].min().round("us").to_pydatetime().replace(tzinfo=None)
    csi_end = csi["time"].max().round("us").to_pydatetime().replace(tzinfo=None)
    gps_start = source_track[0]["time"]
    gps_end = source_track[-1]["time"]
    csi_span = (csi_end - csi_start).total_seconds()
    gps_span = (gps_end - gps_start).total_seconds()

    rows: list[dict[str, Any]] = []
    for point in points:
        center = (point["t0"] + (point["t1"] - point["t0"]) / 2).to_pydatetime()
        fraction = min(max((center - csi_start).total_seconds() / csi_span, 0.0), 1.0)
        mapped_time = gps_start + timedelta(seconds=fraction * gps_span)
        lat, lon, elevation = interpolate_gpx(source_track, mapped_time)
        csi_mean_magnitude = float(np.nanmean(np.abs(point["csi_by_port"][0])))
        csi_relative_power_db = float(
            20.0 * np.log10(max(csi_mean_magnitude, np.finfo(float).tiny))
        )
        rows.append(
            {
                "point": point["point"],
                "measurement_start_utc": point["t0"].isoformat(),
                "measurement_end_utc": point["t1"].isoformat(),
                "measurement_center_utc": center.isoformat(),
                "mapped_gpx_time_utc": mapped_time.isoformat(),
                "clock_fraction": fraction,
                "lat": lat,
                "lon": lon,
                "elevation_m": elevation,
                "csi_mean_magnitude": csi_mean_magnitude,
                "csi_relative_power_db": csi_relative_power_db,
                "gnb_pusch_rsrp_db": point["rsrp"],
                "pusch_sinr_db": point["snr"],
                "cqi": point["cqi"],
                "dl_mcs": point["mcs"],
                "dl_bler_percent": point["dl_bler"],
                "ul_bler_percent": point["ul_bler"],
                "dl_throughput_mbps": point["dl_mbps"],
                "ul_throughput_mbps": point["ul_mbps"],
                "csi_samples": point["n_samples"],
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame["csi_relative_power_db"]).all():
        raise RuntimeError("No finite CSI-derived calibration samples were produced")

    actual_fields: set[str] = set()
    with (data_dir / "gnb_metrics.jsonl").open() as source:
        for line in source:
            for cell in json.loads(line).get("cells", []):
                for ue in cell.get("ue_list", []):
                    actual_fields.update(ue)
    audit = {
        "dataset": dataset,
        "real_data_dir": str(data_dir),
        "raw_formats": {
            "gnb_metrics": "JSONL: timestamp -> cells[] -> ue_list[]",
            "csi": "NPZ arrays: rx_port, tx_port, ta_us, csi[144 complex], timestamp[int64 ns]",
            "gpx": "GPX 1.1 trkpt: UTC time, lat, lon, elevation",
        },
        "actual_ue_fields": sorted(actual_fields),
        "calibration_target": "csi_relative_power_db",
        "calibration_target_formula": "20*log10(mean(abs(raw CSI samples and subcarriers)))",
        "calibration_target_reason": (
            "This is the exact dB transform of the real magnitude curve plotted in "
            "01_csi_mean_subcarrier. It preserves that curve's point-to-point trend. "
            "The gNB PUSCH RSRP is retained only as a diagnostic because it is a different metric."
        ),
        "clock_alignment": {
            "method": "affine session-time mapping from CSI/gNB clock to phone-GPX clock",
            "csi_start_utc": csi_start.isoformat(),
            "csi_end_utc": csi_end.isoformat(),
            "gpx_start_utc": gps_start.isoformat(),
            "gpx_end_utc": gps_end.isoformat(),
            "formula": "gps_t = gps_t0 + (measurement_t-csi_t0)/(csi_t1-csi_t0)*(gps_t1-gps_t0)",
            "reason": "independent logger clocks; absolute starts differ and direct matching loses early points",
        },
        "detected_bursts": len(points) + len(skipped),
        "calibration_rows_with_csi_coverage": len(points),
        "skipped_without_csi": len(skipped),
    }
    return frame, audit


def prepare_mobile_measurements(project_root: Path, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pair each processed mobile gNB report with one interpolated GPX point."""
    scenario = scenario_definition(project_root, dataset)
    data_dir = real_data_dir(project_root, dataset)
    metrics_path = scenario.processed_dir / "mobile_active_metrics.csv"
    if not metrics_path.is_file():
        raise RuntimeError(f"Processed mobile metrics not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path, parse_dates=["timestamp"])
    if metrics.empty:
        raise RuntimeError(f"No mobile metric rows were produced for {dataset}")

    processing = load_processing_module(project_root, "mobile")
    csi, _ = processing.load_all_csi(data_dir)
    port_mask = csi["rx_port"] == 0
    port_csi = csi["csi"][port_mask]
    port_times = pd.DatetimeIndex(csi["time"][port_mask])
    if not len(port_times):
        raise RuntimeError(f"RX port 0 has no mobile CSI for {dataset}")
    metric_times = pd.DatetimeIndex(metrics["timestamp"])
    boundaries = metric_times[:-1] + (metric_times[1:] - metric_times[:-1]) / 2
    edges = pd.DatetimeIndex([
        metric_times[0] - pd.Timedelta(seconds=0.5),
        *boundaries,
        metric_times[-1] + pd.Timedelta(seconds=0.5),
    ])
    time_ns = port_times.asi8
    csi_mean_magnitudes: list[float] = []
    for index, timestamp in enumerate(metric_times):
        left = int(np.searchsorted(time_ns, edges[index].value, side="left"))
        right = int(np.searchsorted(time_ns, edges[index + 1].value, side="right"))
        samples = port_csi[left:right]
        if not len(samples):
            nearest = int(np.argmin(np.abs(time_ns - timestamp.value)))
            samples = port_csi[nearest:nearest + 1]
        csi_mean_magnitudes.append(float(np.nanmean(np.abs(samples))))

    gpx_path = project_root / "data/GPX/Jul_19_213850.gpx"
    source_track = parse_gpx_with_elevation(gpx_path)
    route_start = source_track[0]["time"]
    route_end = source_track[-1]["time"]
    fractions = np.linspace(0.0, 1.0, len(metrics))
    rows: list[dict[str, Any]] = []
    for index, (fraction, (_, metric)) in enumerate(zip(fractions, metrics.iterrows())):
        mapped_time = route_start + timedelta(
            seconds=float(fraction) * (route_end - route_start).total_seconds()
        )
        lat, lon, elevation = interpolate_gpx(source_track, mapped_time)
        csi_mean_magnitude = csi_mean_magnitudes[index]
        csi_relative_power_db = float(
            20.0 * np.log10(max(csi_mean_magnitude, np.finfo(float).tiny))
        )
        rows.append(
            {
                "point": index + 1,
                "measurement_center_utc": pd.Timestamp(metric["timestamp"]).isoformat(),
                "mapped_gpx_time_utc": mapped_time.isoformat(),
                "clock_fraction": float(fraction),
                "lat": lat,
                "lon": lon,
                "elevation_m": elevation,
                "csi_mean_magnitude": csi_mean_magnitude,
                "csi_relative_power_db": csi_relative_power_db,
                "gnb_pusch_rsrp_db": float(metric["pusch_rsrp_db"]),
                "pusch_sinr_db": metric.get("pusch_snr_db", np.nan),
                "cqi": metric.get("cqi", np.nan),
                "dl_mcs": metric.get("dl_mcs", np.nan),
                "dl_bler_percent": metric.get("dl_bler", np.nan),
                "ul_bler_percent": metric.get("ul_bler", np.nan),
                "dl_throughput_mbps": metric.get("dl_tput_mbps", np.nan),
                "ul_throughput_mbps": metric.get("ul_tput_mbps", np.nan),
            }
        )

    frame = pd.DataFrame(rows)
    audit = {
        "dataset": dataset,
        "real_data_dir": str(data_dir),
        "processed_metrics": str(metrics_path),
        "gpx": str(gpx_path),
        "calibration_target": "csi_relative_power_db",
        "calibration_target_formula": "20*log10(mean(abs(raw CSI samples and subcarriers)))",
        "sample_count": len(frame),
        "alignment": {
            "method": "sequential one-to-one interpolation over the shared GPX route",
            "measurement_rows": len(frame),
            "generated_gpx_points": len(frame),
            "source_gpx_points": len(source_track),
            "route_start_utc": route_start.isoformat(),
            "route_end_utc": route_end.isoformat(),
            "formula": "gpx_t[i] = gpx_t0 + i/(N-1) * (gpx_t1-gpx_t0)",
        },
    }
    return frame, audit


def prepare_measurements(project_root: Path, dataset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    scenario = scenario_definition(project_root, dataset)
    if scenario.mobility == "stationary":
        return prepare_stationary_measurements(project_root, dataset)
    return prepare_mobile_measurements(project_root, dataset)


def write_calibration_gpx(frame: pd.DataFrame, path: Path) -> None:
    ET.register_namespace("", GPX_NS)
    root = ET.Element(f"{{{GPX_NS}}}gpx", version="1.1", creator="EBSTR AODT calibration pipeline")
    metadata = ET.SubElement(root, f"{{{GPX_NS}}}metadata")
    ET.SubElement(metadata, f"{{{GPX_NS}}}name").text = "EBSTR aligned calibration points"
    track = ET.SubElement(root, f"{{{GPX_NS}}}trk")
    ET.SubElement(track, f"{{{GPX_NS}}}name").text = "EBSTR calibration"
    segment = ET.SubElement(track, f"{{{GPX_NS}}}trkseg")
    synthetic_start = datetime(2026, 7, 16, tzinfo=timezone.utc)
    for index, row in frame.iterrows():
        point = ET.SubElement(segment, f"{{{GPX_NS}}}trkpt", lat=f"{row['lat']:.9f}", lon=f"{row['lon']:.9f}")
        ET.SubElement(point, f"{{{GPX_NS}}}ele").text = f"{row['elevation_m']:.3f}"
        stamp = synthetic_start + timedelta(seconds=int(index))
        ET.SubElement(point, f"{{{GPX_NS}}}time").text = stamp.isoformat().replace("+00:00", "Z")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def write_prepared_inputs(frame: pd.DataFrame, audit: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aligned_csv = output_dir / "aligned_measurements.csv"
    power_csv = output_dir / "ru1_ue1_power.csv"
    gpx_path = output_dir / "ebstr_calibration_points.gpx"
    audit_path = output_dir / "field_and_alignment_audit.json"
    frame.to_csv(aligned_csv, index=False)
    pd.DataFrame({
        "uniqtimestamp": [""] * len(frame),
        "time": np.arange(len(frame), dtype=int),
        NVIDIA_POWER_COLUMN: frame["csi_relative_power_db"].to_numpy(float),
    }).to_csv(power_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    write_calibration_gpx(frame, gpx_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    power_check = pd.read_csv(power_csv)
    gpx_count = len(ET.parse(gpx_path).findall(f".//{{{GPX_NS}}}trkpt"))
    if list(power_check.columns) != ["uniqtimestamp", "time", NVIDIA_POWER_COLUMN] or len(power_check) != gpx_count:
        raise RuntimeError("Calibration CSV and GPX are not strictly one-to-one")
    return {"aligned": aligned_csv, "power": power_csv, "gpx": gpx_path, "audit": audit_path}


def offset_latlon(lat: float, lon: float, east_m: float, north_m: float) -> tuple[float, float]:
    return lat + north_m / 110540.0, lon + east_m / (111320.0 * math.cos(math.radians(lat)))


def scenario_args(cli: argparse.Namespace, gpx_key: str, sim_id: str, ru_lat: float, ru_lon: float,
                  ru_power_dbm: float, sample_count: int) -> Namespace:
    return Namespace(
        scene="demo_gis/EBSTR", asset_config=str(Path(__file__).with_name("example_client_assets.yml")),
        sim_id=sim_id, db_host=cli.db_host, db_port=cli.db_port,
        s3_endpoint=cli.s3_endpoint, s3_bucket=cli.s3_bucket, s3_provider=cli.s3_provider,
        s3_access_key=cli.s3_access_key, s3_secret_key=cli.s3_secret_key,
        iceberg_uri=cli.iceberg_uri, iceberg_catalog_type=cli.iceberg_catalog_type,
        vegetation_geojson=cli.vegetation_geojson, duration=float(max(sample_count - 1, 0)), interval=1.0,
        seed=cli.seed, freq_mhz=cli.freq_mhz, scs_khz=cli.scs_khz,
        channel_bandwidth_mhz=cli.channel_bandwidth_mhz, du_fft_size=cli.du_fft_size,
        ru_lat=ru_lat, ru_lon=ru_lon, ru_alt=None, ru_height=cli.ru_height,
        gpx_s3_key=gpx_key, use_pathfinding=False, ru_power_dbm=ru_power_dbm,
        ru_azimuth=cli.ru_azimuth,
    )


def build_calibration_ready_yaml(args: Namespace, yaml_path: Path) -> str:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(ebstr_cir.build_scenario_yaml(args))
    config = SimConfig.from_yaml_file(str(yaml_path))
    config.set_ray_tracing_model(DiffusionModel.DIRECTIONAL)
    config.set_bldg_exterior_attr(
        activate_rf=True, activate_diffraction=True, activate_diffusion=True,
        activate_transmission=True, diffuse_surface_element_area=15.0,
    )
    config.get_ru(1).set_radiated_power(args.ru_power_dbm)
    content = OmegaConf.to_yaml(config.to_dict())
    yaml_path.write_text(content)
    return content


def cfr_relative_power_db(values: np.ndarray, delays: np.ndarray,
                          fft_size: int = 512, scs_hz: float = 30e3) -> float:
    """Return the exact dB transform of the AODT magnitude used by CSI plot 01."""
    cfr = np.asarray(ebstr_cir.cir_to_cfr_buffer(
        values, delays, samp_rate=fft_size * scs_hz, fft_size=fft_size,
        keep_absolute_delay=False, path_axis="auto", max_channel_taps=4096,
    ))
    # Match analyze_sim_to_real_gap.load_cfr(): coherently average any antenna
    # dimensions at each subcarrier, then average the resulting magnitudes.
    mean_cfr = np.nanmean(cfr.reshape(fft_size, -1), axis=1)
    mean_magnitude = float(np.nanmean(np.abs(mean_cfr)))
    return float(20.0 * np.log10(max(mean_magnitude, np.finfo(float).tiny)))


def collect_simulated_power(client: Any, status: dict[str, Any], count: int,
                            tx_power_dbm: float, cli: argparse.Namespace) -> np.ndarray:
    allocation = client.allocate_cirs_memory([0], [[0]], False)
    powers: list[float] = []
    try:
        for step in range(count):
            temporal = ebstr_cir.temporal_index(step, status["is_slot_symbol_mode"])
            client.get_cirs(allocation, batch_index=0, temporal_index=temporal)
            values = np.asarray(client.to_numpy(allocation, step, 0, "values")[0])
            delays = np.asarray(client.to_numpy(allocation, step, 0, "delays")[0])
            powers.append(cfr_relative_power_db(
                values, delays, fft_size=cli.du_fft_size,
                scs_hz=cli.scs_khz * 1e3,
            ))
    finally:
        try:
            client.deallocate_cirs_memory(allocation)
        except RuntimeError:
            pass
    return np.asarray(powers)


def export_bridge_channel(client: Any, count: int, cli: argparse.Namespace, stem: Path) -> None:
    export_args = Namespace(
        ru_index=0, ue_index=0, ru_id=1, ue_id=1, full_antenna_pair=False,
        samp_rate=cli.sample_rate_hz, cfr_fft_size=cli.du_fft_size, cfr_scs_khz=cli.scs_khz,
        keep_absolute_delay=False, max_channel_taps=4096, cir_path_axis="auto",
        freq_mhz=cli.freq_mhz, cir_file=str(stem.with_name(stem.name + "_CIR.dat")),
        cfr_file=str(stem.with_name(stem.name + "_CFR.dat")),
    )
    ebstr_cir.export_ue_cir(client, client.get_status(), count, export_args)


def run_simulation(client: Any, yaml_content: str, count: int, tx_power_dbm: float,
                   label: str, cli: argparse.Namespace) -> np.ndarray:
    print(f"\n=== {label} ===")
    if not client.start(yaml_content):
        raise RuntimeError(f"AODT rejected {label} YAML")
    status = client.get_status()
    actual = int(status["num_slots_or_timesteps_per_batch"])
    if actual != count:
        raise RuntimeError(
            f"{label}: AODT has {actual} time steps but GPX/measurement have {count}; "
            "calibration requires exact one-to-one alignment"
        )
    result = client.run_full_simulation()
    completed = int(result.get("time_steps_completed") or 0)
    if completed != count:
        raise RuntimeError(f"{label}: completed {completed}/{count} time steps")
    return collect_simulated_power(client, status, count, tx_power_dbm, cli)


def centered_rmse(measured: np.ndarray, simulated: np.ndarray) -> float:
    return float(np.sqrt(np.mean(((measured - measured.mean()) - (simulated - simulated.mean())) ** 2)))


def upload_inputs(cli: argparse.Namespace, paths: dict[str, Path]) -> tuple[str, str]:
    prefix = cli.s3_prefix.rstrip("/")
    gpx_key = f"{prefix}/input/{paths['gpx'].name}"
    power_key = f"{prefix}/input/{paths['power'].name}"
    io_args = Namespace(**vars(cli))
    io_args.s3_endpoint = cli.s3_client_endpoint
    ebstr_cir.upload_file_to_s3(io_args, paths["gpx"], gpx_key, "application/gpx+xml")
    ebstr_cir.upload_file_to_s3(io_args, paths["power"], power_key, "text/csv")
    return gpx_key, power_key


def create_calibration_yaml(base_yaml: Path, power_key: str, output_key: str,
                            count: int, destination: Path) -> str:
    config = SimConfig.from_yaml_file(str(base_yaml))
    config.set_calibration_targets(materials=True, veg_materials=True, rus=True, rus_beams=False, ues=False)
    config.add_calibration_measurement(ru_id=1, ue_id=1, measurement_file=power_key)
    config.set_calibration_timeline(start=0, step=1, end=count - 1)
    config.set_calibration_output(output_key)
    content = OmegaConf.to_yaml(config.to_dict())
    destination.write_text(content)
    return content


def list_s3_keys(cli: argparse.Namespace, prefix: str) -> list[str]:
    helper = """
import boto3, sys
endpoint, bucket, prefix, access, secret = sys.argv[1:6]
s3 = boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=access, aws_secret_access_key=secret)
token = None
while True:
    args = {'Bucket': bucket, 'Prefix': prefix}
    if token: args['ContinuationToken'] = token
    page = s3.list_objects_v2(**args)
    for item in page.get('Contents', []): print(item['Key'])
    if not page.get('IsTruncated'): break
    token = page['NextContinuationToken']
"""
    result = subprocess.run(
        ["/usr/bin/python3", "-c", helper, cli.s3_client_endpoint, cli.s3_bucket, prefix,
         cli.s3_access_key, cli.s3_secret_key], check=True, text=True, capture_output=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def download_s3(cli: argparse.Namespace, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/usr/bin/python3", "-c", ebstr_cir._S3_DOWNLOAD_HELPER, cli.s3_client_endpoint,
         cli.s3_bucket, key, cli.s3_access_key, cli.s3_secret_key, str(destination)], check=True,
    )


def find_and_download_calibrated_config(cli: argparse.Namespace, output_key: str,
                                        output_dir: Path) -> tuple[Path, list[str]]:
    keys = list_s3_keys(cli, output_key.rstrip("/") + "/")
    matches = [key for key in keys if key.endswith("/sim_config_calibrated.yml")]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one calibrated simulation config, found {matches}; S3 keys={keys}")
    destination = output_dir / "sim_config_calibrated.yml"
    download_s3(cli, matches[0], destination)
    return destination, keys


def disambiguate_calibrated_associations(cli: argparse.Namespace, config_path: Path) -> Path:
    """Avoid the v1.5 cache collision between two association.json files."""
    config = OmegaConf.load(config_path)
    building_source = str(config.sim.Materials.calibration.assignment[0])
    vegetation_source = str(config.sim.VegetationMaterials.calibration.assignment[0])
    building_target = str(Path(building_source).with_name("materials_association.json"))
    vegetation_target = str(Path(vegetation_source).with_name("vegetation_association.json"))
    helper = """
import boto3, sys
endpoint, bucket, access, secret, bsrc, bdst, vsrc, vdst = sys.argv[1:9]
s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access, aws_secret_access_key=secret)
s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": bsrc}, Key=bdst)
s3.copy_object(Bucket=bucket, CopySource={"Bucket": bucket, "Key": vsrc}, Key=vdst)
"""
    subprocess.run(
        ["/usr/bin/python3", "-c", helper, cli.s3_client_endpoint, cli.s3_bucket,
         cli.s3_access_key, cli.s3_secret_key, building_source, building_target,
         vegetation_source, vegetation_target],
        check=True,
    )
    config.sim.Materials.calibration.assignment = [building_target]
    config.sim.VegetationMaterials.calibration.assignment = [vegetation_target]
    for du_update in config.sim.DUs["update"]:
        du_update.attributes.aerial_du_num_antennas = 1
    compatible = config_path.with_name("sim_config_calibrated_ebstr.yml")
    OmegaConf.save(config, compatible)
    return compatible


def flatten_strings(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from flatten_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from flatten_strings(child)
    elif isinstance(value, str):
        yield value


def verify_calibrated_material_loading(config_path: Path) -> dict[str, list[str]]:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    strings = list(flatten_strings(config))
    building = [value for value in strings if value.endswith("/materials_calibrated.json")]
    vegetation = [value for value in strings if value.endswith("veg_materials_calibrated.json")]
    if not building or not vegetation:
        raise RuntimeError(
            "Generated post-calibration config does not reference both calibrated building and "
            f"vegetation material definitions: building={building}, vegetation={vegetation}"
        )
    return {"building_material_definitions": building, "vegetation_material_definitions": vegetation}


def read_calibrated_ru_orientation(config_path: Path) -> dict[str, float]:
    config = OmegaConf.load(config_path)
    ru_attributes = config.sim.RUs["update"][-1].attributes
    panel_attributes = config.sim.Panels["update"][-1].attributes
    return {
        "azimuth_deg": float(ru_attributes.aerial_gnb_mech_azimuth),
        "tilt_deg": float(ru_attributes.aerial_gnb_mech_tilt),
        "panel_roll_first_pol_deg": float(panel_attributes.antenna_roll_angle_first_polz_degree),
    }


def compare_results(frame: pd.DataFrame, baseline: np.ndarray, calibrated: np.ndarray,
                    output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    measured = frame["csi_relative_power_db"].to_numpy(float)
    baseline_offset = float(np.mean(measured - baseline))
    calibrated_offset = float(np.mean(measured - calibrated))
    baseline_aligned = baseline + baseline_offset
    calibrated_aligned = calibrated + calibrated_offset
    comparison = frame.copy()
    comparison["baseline_aodt_relative_power_db"] = baseline
    comparison["calibrated_aodt_relative_power_db"] = calibrated
    comparison["baseline_level_aligned_db"] = baseline_aligned
    comparison["calibrated_level_aligned_db"] = calibrated_aligned
    comparison["measured_centered_db"] = measured - measured.mean()
    comparison["baseline_centered_db"] = baseline - baseline.mean()
    comparison["calibrated_centered_db"] = calibrated - calibrated.mean()
    comparison.to_csv(output_dir / "measurement_baseline_calibrated.csv", index=False)
    summary = {
        **metadata, "sample_count": len(frame),
        "metrics": {
            "baseline_centered_rmse_db": centered_rmse(measured, baseline),
            "calibrated_centered_rmse_db": centered_rmse(measured, calibrated),
            "baseline_correlation": float(np.corrcoef(measured, baseline)[0, 1]),
            "calibrated_correlation": float(np.corrcoef(measured, calibrated)[0, 1]),
            "baseline_plot_level_offset_db": baseline_offset,
            "calibrated_plot_level_offset_db": calibrated_offset,
        },
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    # CSI has no absolute dBm reference. AODT calibration trains centered power,
    # so the simulated traces are shifted only by constant mean offsets for this
    # shape comparison; no per-point scaling or fitting is applied.
    x = np.arange(len(comparison))
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.plot(x, measured, label="Measured target", color="#1f77b4", linewidth=1.8)
    ax.plot(x, baseline_aligned, label="AODT baseline", color="#2ca02c", linewidth=1.8)
    ax.plot(x, calibrated_aligned, label="AODT calibrated", color="#ff7f0e", linewidth=1.8)
    ax.set_xlabel("time_indices")
    ax.set_ylabel("CSI-derived relative RSRP (dB)")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "measurement_baseline_calibrated.png", dpi=180)
    fig.savefig(output_dir / "measurement_baseline_calibrated.pdf")
    plt.close(fig)
    return summary


def run_pipeline(cli: argparse.Namespace, frame: pd.DataFrame, paths: dict[str, Path]) -> dict[str, Any]:
    output_dir = Path(cli.output_dir).resolve()
    measured = frame["csi_relative_power_db"].to_numpy(float)
    gpx_key, power_key = upload_inputs(cli, paths)
    base_ru_lat = cli.ru_lat
    base_ru_lon = cli.ru_lon
    radius = max(float(cli.ru_search_radius_m), 0.0)
    offsets = [(0.0, 0.0)]
    if radius > 0:
        offsets += [(radius, 0.0), (-radius, 0.0), (0.0, radius), (0.0, -radius)]
    client = dt_client.DigitalTwinClient(cli.server_address)
    client.start_server_log_streaming(str(output_dir / "dt_server_calibration.log"), "INFO")
    try:
        candidates: list[dict[str, Any]] = []
        if radius > 0:
            for index, (east, north) in enumerate(offsets):
                lat, lon = offset_latlon(base_ru_lat, base_ru_lon, east, north)
                args = scenario_args(
                    cli, gpx_key, f"EBSTR_{cli.dataset}_ru_search_{index}", lat, lon,
                    cli.initial_ru_power_dbm, len(frame),
                )
                yaml_path = output_dir / f"ru_candidate_{index}.yml"
                power = run_simulation(
                    client, build_calibration_ready_yaml(args, yaml_path), len(frame),
                    cli.initial_ru_power_dbm,
                    f"RU position candidate {index + 1}/{len(offsets)}", cli,
                )
                candidates.append({
                    "index": index, "east_m": east, "north_m": north,
                    "lat": lat, "lon": lon,
                    "centered_rmse_db": centered_rmse(measured, power), "power": power,
                })
            best = min(candidates, key=lambda item: item["centered_rmse_db"])
            diagnostic_offset = float(np.mean(measured - best["power"]))
        else:
            # With no search radius, the candidate would be identical to the
            # official baseline. Do not run the same full simulation twice.
            best = {
                "index": 0, "east_m": 0.0, "north_m": 0.0,
                "lat": base_ru_lat, "lon": base_ru_lon,
                "centered_rmse_db": None,
            }
            diagnostic_offset = None
        # CSI has no absolute dBm calibration. Keep RU radiated power fixed;
        # only centered curve shape is meaningful for this calibration target.
        optimized_power = float(cli.initial_ru_power_dbm)
        (output_dir / "ru_position_search.json").write_text(json.dumps({
            "objective": "centered RMSE against CSI-derived relative power",
            "note": "AODT v1.5 rus target trains angles only; position is an outer simulation search",
            "candidates": [{k: v for k, v in item.items() if k != "power"} for item in candidates],
            "selected_index": best["index"],
            "power_fit": {
                "method": "disabled for CSI-relative target; reported offset is diagnostic only",
                "diagnostic_mean_offset_db": diagnostic_offset,
                "initial_ru_power_dbm": cli.initial_ru_power_dbm,
                "optimized_ru_power_dbm": optimized_power, "limits_dbm": [0.0, 60.0],
            },
        }, indent=2) + "\n")

        baseline_args = scenario_args(
            cli, gpx_key, f"EBSTR_{cli.dataset}_calibration_baseline", best["lat"], best["lon"],
            optimized_power, len(frame),
        )
        baseline_path = output_dir / "ebstr_calibration_baseline.yml"
        baseline_power = run_simulation(
            client, build_calibration_ready_yaml(baseline_args, baseline_path), len(frame),
            optimized_power, "Official calibration baseline simulation", cli,
        )
        np.save(output_dir / "baseline_csi_relative_power_db.npy", baseline_power)
        export_bridge_channel(client, len(frame), cli, output_dir / "baseline")
        calibration_output_key = cli.s3_prefix.rstrip("/") + "/output"
        calibration_path = output_dir / "ebstr_calibration.yml"
        calibration_yaml = create_calibration_yaml(
            baseline_path, power_key, calibration_output_key, len(frame), calibration_path
        )
        print("\n=== Official AODT material/vegetation/RU-angle calibration ===")
        if not client.start(calibration_yaml):
            raise RuntimeError("AODT rejected calibration YAML")
        calibration_result = client.run_calibration()
        if calibration_result.get("stage") != "completed":
            raise RuntimeError(f"Calibration did not complete: {calibration_result}")
        calibrated_config_path, s3_outputs = find_and_download_calibrated_config(
            cli, calibration_output_key, output_dir
        )
        calibrated_config_path = disambiguate_calibrated_associations(cli, calibrated_config_path)
        loaded_materials = verify_calibrated_material_loading(calibrated_config_path)
        calibrated_power = run_simulation(
            client, calibrated_config_path.read_text(), len(frame), optimized_power,
            "Post-calibration simulation using generated material definitions", cli,
        )
        np.save(output_dir / "calibrated_csi_relative_power_db.npy", calibrated_power)
        export_bridge_channel(client, len(frame), cli, output_dir / "calibrated")
        return compare_results(frame, baseline_power, calibrated_power, output_dir, {
            "dataset": cli.dataset,
            "bandwidth_mhz": cli.channel_bandwidth_mhz,
            "calibration_target": "csi_relative_power_db",
            "calibration_target_formula": "20*log10(mean(abs(CSI)))",
            "calibration_api_result": calibration_result,
            "calibration_targets": {
                "Materials": True, "VegMaterials": True, "RUs": True,
                "RUsBeams": False, "UEs": False,
            },
            "selected_ru": {
                "lat": best["lat"], "lon": best["lon"], "radiated_power_dbm": optimized_power,
            },
            "verified_calibrated_material_inputs": loaded_materials,
            "calibrated_ru_orientation": read_calibrated_ru_orientation(calibrated_config_path),
            "calibration_s3_outputs": s3_outputs,
        })
    finally:
        client.stop_server_log_streaming()


def build_parser(project_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=("20mhz_stationary", "20mhz_mobile", "100mhz_stationary", "100mhz_mobile"),
        default="100mhz_stationary",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--server-address", default="localhost:50051")
    parser.add_argument("--s3-endpoint", default="http://minio:9000")
    parser.add_argument(
        "--s3-client-endpoint", default="http://localhost:9000",
        help="Host-visible endpoint used only to upload/download objects; the worker YAML uses --s3-endpoint",
    )
    parser.add_argument("--s3-bucket", default="aerial-data")
    parser.add_argument("--s3-provider", default="minio", choices=("minio", "aws"))
    parser.add_argument("--s3-access-key", default="minioadmin")
    parser.add_argument("--s3-secret-key", default="minioadmin")
    parser.add_argument("--s3-prefix", default=None)
    parser.add_argument("--iceberg-uri", default="http://nessie:19120/iceberg")
    parser.add_argument("--iceberg-catalog-type", default="rest", choices=("rest", "sql", "glue"))
    parser.add_argument("--db-host", default="clickhouse")
    parser.add_argument("--db-port", type=int, default=9000)
    parser.add_argument("--vegetation-geojson", default="demo_gis/EBSTR/sim/vegetation_ebstr.geojson")
    parser.add_argument("--freq-mhz", type=float, default=3489.42)
    parser.add_argument("--scs-khz", type=float, default=30.0)
    parser.add_argument("--channel-bandwidth-mhz", type=float)
    parser.add_argument("--du-fft-size", type=int)
    parser.add_argument("--sample-rate-hz", type=float)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ru-azimuth", type=float, default=0.0)
    parser.add_argument("--ru-lat", type=float, default=42.72399)
    parser.add_argument("--ru-lon", type=float, default=-84.48004)
    parser.add_argument("--ru-height", type=float, default=1.0)
    parser.add_argument("--initial-ru-power-dbm", type=float, default=43.0)
    parser.add_argument(
        "--ru-search-radius-m", type=float, default=0.0,
        help="Cross-search radius around the configured/first aligned point; 0 runs only the center",
    )
    return parser


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    args = build_parser(project_root).parse_args()
    scenario = scenario_definition(project_root, args.dataset)
    args.channel_bandwidth_mhz = args.channel_bandwidth_mhz or float(scenario.bandwidth_mhz)
    args.du_fft_size = args.du_fft_size or scenario.aodt_fft_size
    args.sample_rate_hz = args.sample_rate_hz or float(scenario.sample_rate_hz)
    bandwidth_dir = f"{scenario.bandwidth_mhz}mhz"
    if args.output_dir is None:
        args.output_dir = str(
            project_root / "data/calibration/EBSTR" / bandwidth_dir / "aodt_native" / args.dataset
        )
    if args.s3_prefix is None:
        args.s3_prefix = f"demo_gis/EBSTR/calibration/{bandwidth_dir}/native/{args.dataset}"
    output_dir = Path(args.output_dir).resolve()
    frame, audit = prepare_measurements(project_root, args.dataset)
    paths = write_prepared_inputs(frame, audit, output_dir)
    print(f"Prepared {len(frame)} one-to-one GPX/power samples in {output_dir}")
    print(f"Calibration target: {audit['calibration_target']} ({NVIDIA_POWER_COLUMN})")
    if args.prepare_only:
        return 0
    summary = run_pipeline(args, frame, paths)
    print(json.dumps(summary["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
