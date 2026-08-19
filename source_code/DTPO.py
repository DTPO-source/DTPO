#!/usr/bin/env python3
"""Context-aware optimal-transport calibration for DTPO."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import qmc

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


DEFAULT_KPIS = (
    "rsrp_db",
    "ul_throughput_mbps",
    "dl_throughput_mbps",
    "cqi",
    "sinr_db",
    "ul_bler_pct",
    "dl_bler_pct",
)


@dataclass(frozen=True)
class Parameter:
    name: str
    lower: float
    upper: float
    initial: float
    optimize: bool
    target: dict[str, Any]

    def __post_init__(self) -> None:
        if self.lower >= self.upper or not self.lower <= self.initial <= self.upper:
            raise ValueError(f"Invalid bounds/initial value for {self.name}")


def load_config(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ModuleNotFoundError("PyYAML is required to read DTPO configuration files")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    parent = config.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        base = load_config(parent_path.resolve())

        def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
            result = dict(left)
            for key, value in right.items():
                result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
            return result

        config = merge(base, config)
    return config


def parameters_from_config(config: dict[str, Any]) -> list[Parameter]:
    parameters = [
        Parameter(
            item["name"],
            float(item["lower_bound"]),
            float(item["upper_bound"]),
            float(item["initial_value"]),
            bool(item.get("optimize", True)),
            dict(item.get("target", {})),
        )
        for item in config.get("parameters", [])
    ]
    if not parameters or len({item.name for item in parameters}) != len(parameters):
        raise ValueError("Parameters must be non-empty and have unique names")
    return parameters


def _set_path(document: Any, dotted_path: str, value: float) -> None:
    current = document
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current[int(part)] if isinstance(current, list) else current[part]
    key = parts[-1]
    if isinstance(current, list):
        current[int(key)] = value
    elif isinstance(current[key], dict) and "value" in current[key]:
        current[key] = {**current[key], "value": value}
    else:
        current[key] = value


class WorkspaceBuilder:
    def __init__(self, parameters: list[Parameter], project_root: Path):
        self.parameters = parameters
        self.project_root = project_root

    def build(self, run_dir: Path, values: dict[str, float]) -> dict[str, str]:
        run_dir.mkdir(parents=True, exist_ok=True)
        documents: dict[str, tuple[Path, Any, str]] = {}
        environment: dict[str, str] = {}
        for parameter in self.parameters:
            target = parameter.target
            kind = target.get("kind", "environment")
            if kind == "environment":
                environment[target.get("name", parameter.name.upper())] = str(values[parameter.name])
                continue
            source = target.get("source")
            dotted_path = target.get("path")
            if kind not in {"json", "yaml"} or not source or not dotted_path:
                raise ValueError(f"{parameter.name}: invalid asset target")
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = self.project_root / source_path
            key = str(source_path.resolve())
            if key not in documents:
                destination = run_dir / source_path.name
                shutil.copy2(source_path, destination)
                with destination.open(encoding="utf-8") as stream:
                    if kind == "yaml" and yaml is None:
                        raise ModuleNotFoundError("PyYAML is required for YAML parameter assets")
                    document = json.load(stream) if kind == "json" else yaml.safe_load(stream)
                documents[key] = destination, document, kind
            _set_path(documents[key][1], dotted_path, values[parameter.name])
        for source, (destination, document, kind) in documents.items():
            with destination.open("w", encoding="utf-8") as stream:
                if kind == "json":
                    json.dump(document, stream, indent=2)
                    stream.write("\n")
                else:
                    if yaml is None:
                        raise ModuleNotFoundError("PyYAML is required for YAML parameter assets")
                    yaml.safe_dump(document, stream, sort_keys=False)
            name = Path(source).stem.upper().replace("-", "_")
            environment[f"BO_INPUT_{name}"] = str(destination.resolve())
        environment["BO_ITERATION_DIR"] = str(run_dir.resolve())
        environment["BO_PARAMETER_JSON"] = json.dumps(values, sort_keys=True)
        return environment


def _robust_scale(values: np.ndarray, epsilon: float) -> float:
    finite = np.asarray(values, float)[np.isfinite(values)]
    if not len(finite):
        raise ValueError("Cannot normalize an empty or non-finite measurement")
    scale = float(np.percentile(finite, 75) - np.percentile(finite, 25))
    if scale <= epsilon:
        scale = float(np.median(np.abs(finite - np.median(finite))) * 1.4826)
    return max(scale, epsilon)


def _complex_columns(frame: pd.DataFrame, settings: dict[str, Any]) -> np.ndarray:
    columns = settings.get("channel_columns")
    if columns:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing channel columns: {sorted(missing)}")
        return np.column_stack([frame[name].map(complex).to_numpy() for name in columns])
    prefixes = settings.get("channel_prefixes", ["cfr", "csi", "h"])
    for prefix in prefixes:
        real_columns = sorted(name for name in frame if name.startswith(f"{prefix}_real_"))
        imag_columns = [name.replace(f"{prefix}_real_", f"{prefix}_imag_", 1) for name in real_columns]
        if real_columns and all(name in frame for name in imag_columns):
            real = frame[real_columns].to_numpy(float)
            imag = frame[imag_columns].to_numpy(float)
            return real + 1j * imag
    if "csi_magnitude" in frame:
        return frame[["csi_magnitude"]].to_numpy(float).astype(complex)
    raise ValueError("CFR/CSI columns are required for the joint channel-KPI objective")


def _time_seconds(series: pd.Series, origin: float | None = None) -> tuple[np.ndarray, float]:
    if pd.api.types.is_numeric_dtype(series):
        values = series.to_numpy(float)
        magnitude = float(np.nanmedian(np.abs(values)))
        divisor = 1e9 if magnitude > 1e17 else 1e6 if magnitude > 1e14 else 1e3 if magnitude > 1e11 else 1.0
        values = values / divisor
    else:
        values = pd.to_datetime(series, utc=True).astype("int64").to_numpy(float) / 1e9
    reference = float(np.nanmin(values)) if origin is None else origin
    return values - reference, reference


def _positions(frame: pd.DataFrame, settings: dict[str, Any], origin: tuple[float, float] | None = None) -> tuple[np.ndarray, tuple[float, float]]:
    x_column = settings.get("x_column", "x_m")
    y_column = settings.get("y_column", "y_m")
    if x_column in frame and y_column in frame:
        position = frame[[x_column, y_column]].to_numpy(float)
        return position, (0.0, 0.0)
    lat_column = settings.get("latitude_column", "latitude")
    lon_column = settings.get("longitude_column", "longitude")
    if lat_column not in frame or lon_column not in frame:
        raise ValueError("Position columns x_m/y_m or latitude/longitude are required")
    latitude = frame[lat_column].to_numpy(float)
    longitude = frame[lon_column].to_numpy(float)
    lat0, lon0 = origin or (float(np.nanmedian(latitude)), float(np.nanmedian(longitude)))
    x = np.deg2rad(longitude - lon0) * 6371000.0 * math.cos(math.radians(lat0))
    y = np.deg2rad(latitude - lat0) * 6371000.0
    return np.column_stack([x, y]), (lat0, lon0)


def _sinkhorn(cost: np.ndarray, epsilon: float, iterations: int, tolerance: float) -> tuple[float, np.ndarray]:
    rows, columns = cost.shape
    log_kernel = -cost / epsilon
    log_a = np.full(rows, -math.log(rows))
    log_b = np.full(columns, -math.log(columns))
    log_u = np.zeros(rows)
    log_v = np.zeros(columns)
    for step in range(iterations):
        previous = log_u.copy()
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)
        if step % 10 == 0 and np.max(np.abs(log_u - previous)) < tolerance:
            break
    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    plan = np.exp(log_plan)
    plan /= plan.sum()
    entropy = np.sum(plan * (np.log(np.maximum(plan, np.finfo(float).tiny)) - 1.0))
    return float(np.sum(plan * cost) + epsilon * entropy), plan


class TransportObjective:
    def __init__(self, real: pd.DataFrame, config: dict[str, Any]):
        self.real = real.reset_index(drop=True)
        self.settings = config.get("fidelity", {})
        self.epsilon = float(self.settings.get("normalization_epsilon", 1e-6))
        configured_kpis = self.settings.get("kpi_metrics")
        self.kpis = list(configured_kpis or [name for name in DEFAULT_KPIS if name in self.real])
        if not self.kpis:
            raise ValueError("No DTPO KPI columns were found")
        missing_kpis = set(self.kpis) - set(self.real.columns)
        if missing_kpis:
            raise ValueError(f"Missing real KPI columns: {sorted(missing_kpis)}")
        self.channel = _complex_columns(self.real, self.settings)
        self.channel_scale = _robust_scale(np.abs(self.channel).ravel(), self.epsilon)
        self.kpi_median = {name: float(np.nanmedian(self.real[name])) for name in self.kpis}
        self.kpi_scale = {name: _robust_scale(self.real[name].to_numpy(float), self.epsilon) for name in self.kpis}
        self.positions, self.position_origin = _positions(self.real, self.settings)
        time_column = self.settings.get("time_column", "timestamp")
        if time_column not in self.real:
            raise ValueError(f"Missing time column: {time_column}")
        self.times, self.time_origin = _time_seconds(self.real[time_column])
        self.groups = self._groups(self.real, self.positions, self.times)

    def _groups(self, frame: pd.DataFrame, positions: np.ndarray, times: np.ndarray) -> np.ndarray:
        group_column = self.settings.get("group_column")
        mode = self.settings.get("scenario", "stationary").lower()
        if group_column:
            if group_column not in frame:
                raise ValueError(f"Missing group column: {group_column}")
            return frame[group_column].astype(str).to_numpy()
        if mode == "mobile":
            window = float(self.settings.get("mobile_window_seconds", 2.0))
            return np.floor(times / window).astype(int).astype(str)
        if "point" in frame:
            return frame["point"].astype(str).to_numpy()
        spatial_scale = float(self.settings.get("spatial_scale_m", 5.0))
        cells = np.rint(positions / spatial_scale).astype(int)
        return np.array([f"{x}:{y}" for x, y in cells])

    def score(self, simulated: pd.DataFrame) -> tuple[float, dict[str, float]]:
        simulated = simulated.reset_index(drop=True)
        missing = set(self.kpis) - set(simulated.columns)
        if missing:
            raise ValueError(f"Missing simulated KPI columns: {sorted(missing)}")
        channel = _complex_columns(simulated, self.settings)
        if channel.shape[1] != self.channel.shape[1]:
            raise ValueError("Real and simulated channel dimensions do not match")
        positions, _ = _positions(simulated, self.settings, self.position_origin)
        time_column = self.settings.get("time_column", "timestamp")
        if time_column not in simulated:
            raise ValueError(f"Missing simulated time column: {time_column}")
        times, _ = _time_seconds(simulated[time_column], self.time_origin)
        groups = self._groups(simulated, positions, times)
        common = sorted(set(self.groups) & set(groups))
        if set(self.groups) != set(groups):
            raise ValueError("Simulated measurements must cover every fixed real-data group")

        alpha = float(self.settings.get("channel_weight", 0.5))
        beta = float(self.settings.get("network_weight", 0.5))
        sigma_x = float(self.settings.get("spatial_scale_m", 5.0))
        sigma_t = float(self.settings.get("temporal_scale_s", 0.5))
        lambda_x = float(self.settings.get("spatial_penalty", 0.1))
        lambda_t = float(self.settings.get("temporal_penalty", 0.1))
        entropy = float(self.settings.get("sinkhorn_epsilon", 0.05))
        sinkhorn_iterations = int(self.settings.get("sinkhorn_iterations", 200))
        sinkhorn_tolerance = float(self.settings.get("sinkhorn_tolerance", 1e-6))
        kpi_weights = self.settings.get("kpi_weights", {})
        default_kpi_weight = 1.0 / len(self.kpis)
        totals = {"channel": 0.0, "network": 0.0, "spatial": 0.0, "temporal": 0.0}
        objectives: list[float] = []

        normalized_real_channel = self.channel / self.channel_scale
        normalized_simulated_channel = channel / self.channel_scale
        for group in common:
            real_index = np.flatnonzero(self.groups == group)
            simulated_index = np.flatnonzero(groups == group)
            channel_cost = np.mean(
                np.abs(
                    normalized_simulated_channel[simulated_index, None, :]
                    - normalized_real_channel[None, real_index, :]
                ) ** 2,
                axis=2,
            )
            network_cost = np.zeros_like(channel_cost)
            for name in self.kpis:
                real_values = (self.real[name].to_numpy(float)[real_index] - self.kpi_median[name]) / self.kpi_scale[name]
                simulated_values = (simulated[name].to_numpy(float)[simulated_index] - self.kpi_median[name]) / self.kpi_scale[name]
                weight = float(kpi_weights.get(name, default_kpi_weight))
                network_cost += weight * (simulated_values[:, None] - real_values[None, :]) ** 2
            spatial_cost = np.sum(
                (positions[simulated_index, None, :] - self.positions[None, real_index, :]) ** 2,
                axis=2,
            ) / sigma_x**2
            temporal_cost = (
                (times[simulated_index, None] - self.times[None, real_index]) / sigma_t
            ) ** 2
            cost = alpha * channel_cost + beta * network_cost + lambda_x * spatial_cost + lambda_t * temporal_cost
            objective, plan = _sinkhorn(cost, entropy, sinkhorn_iterations, sinkhorn_tolerance)
            objectives.append(objective)
            totals["channel"] += float(np.sum(plan * channel_cost))
            totals["network"] += float(np.sum(plan * network_cost))
            totals["spatial"] += float(np.sum(plan * spatial_cost))
            totals["temporal"] += float(np.sum(plan * temporal_cost))
        count = len(common)
        return float(np.mean(objectives)), {name: value / count for name, value in totals.items()}


class GaussianProcess:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.x = np.asarray(x, float)
        self.y_mean = float(np.mean(y))
        self.y_scale = max(float(np.std(y)), 1e-6)
        self.y = (np.asarray(y, float) - self.y_mean) / self.y_scale
        dimension = self.x.shape[1]

        def objective(log_lengthscale: np.ndarray) -> float:
            covariance = self._kernel(self.x, self.x, np.exp(log_lengthscale))
            covariance += np.eye(len(self.x)) * 1e-5
            try:
                factor = cho_factor(covariance, lower=True, check_finite=False)
            except np.linalg.LinAlgError:
                return 1e12
            alpha = cho_solve(factor, self.y, check_finite=False)
            return float(0.5 * self.y @ alpha + np.sum(np.log(np.diag(factor[0]))) + 0.5 * len(self.x) * math.log(2 * math.pi))

        result = minimize(objective, np.full(dimension, math.log(0.3)), method="L-BFGS-B", bounds=[(math.log(0.02), math.log(2.0))] * dimension)
        self.lengthscale = np.exp(result.x)
        covariance = self._kernel(self.x, self.x, self.lengthscale) + np.eye(len(self.x)) * 1e-5
        self.factor = cho_factor(covariance, lower=True, check_finite=False)
        self.alpha = cho_solve(self.factor, self.y, check_finite=False)
        return self

    @staticmethod
    def _kernel(left: np.ndarray, right: np.ndarray, lengthscale: np.ndarray) -> np.ndarray:
        distance = np.sqrt(np.sum(((left[:, None, :] - right[None, :, :]) / lengthscale) ** 2, axis=2))
        root5 = math.sqrt(5.0)
        return (1.0 + root5 * distance + 5.0 * distance**2 / 3.0) * np.exp(-root5 * distance)

    def thompson_sample(self, candidates: np.ndarray, rng: np.random.Generator, features: int = 512) -> np.ndarray:
        omega = rng.standard_t(5, size=(features, self.x.shape[1])) / self.lengthscale
        phase = rng.uniform(0.0, 2.0 * math.pi, features)
        weights = rng.normal(size=features)
        scale = math.sqrt(2.0 / features)
        train_features = scale * np.cos(self.x @ omega.T + phase)
        candidate_features = scale * np.cos(candidates @ omega.T + phase)
        prior_train = train_features @ weights
        prior_candidates = candidate_features @ weights
        residual = self.y - prior_train - rng.normal(0.0, math.sqrt(1e-5), len(self.x))
        correction = self._kernel(candidates, self.x, self.lengthscale) @ cho_solve(self.factor, residual, check_finite=False)
        return self.y_mean + self.y_scale * (prior_candidates + correction)


@dataclass
class TrustRegionState:
    initial: float = 0.8
    length: float = 0.8
    minimum: float = 2**-7
    maximum: float = 1.6
    success_tolerance: int = 10
    failure_tolerance: int = 16
    successes: int = 0
    failures: int = 0
    best: float = math.inf
    restart: bool = False

    def update(self, value: float) -> None:
        improved = not math.isfinite(self.best) or value < self.best - 1e-3 * max(1.0, abs(self.best))
        if improved:
            self.best = value
            self.successes += 1
            self.failures = 0
        else:
            self.successes = 0
            self.failures += 1
        if self.successes >= self.success_tolerance:
            self.length = min(2.0 * self.length, self.maximum)
            self.successes = 0
        elif self.failures >= self.failure_tolerance:
            self.length *= 0.5
            self.failures = 0
        if self.length < self.minimum:
            self.length = self.initial
            self.restart = True


class TurboOptimizer:
    def __init__(self, dimension: int, settings: dict[str, Any]):
        self.dimension = dimension
        self.seed = int(settings.get("seed", 20260718))
        self.rng = np.random.default_rng(self.seed)
        self.initial_design = int(settings.get("initial_design", 32))
        self.candidate_count = int(settings.get("candidate_count", 3200))
        self.restart_index = 0
        exponent = math.ceil(math.log2(max(1, self.initial_design)))
        self.sobol_initial = qmc.Sobol(dimension, scramble=True, seed=self.seed).random_base2(exponent)[: self.initial_design]
        initial_length = float(settings.get("trust_region_initial", 0.8))
        self.state = TrustRegionState(
            initial=initial_length,
            length=initial_length,
            minimum=float(settings.get("trust_region_min", 2**-7)),
            maximum=float(settings.get("trust_region_max", 1.6)),
            success_tolerance=int(settings.get("success_tolerance", 10)),
            failure_tolerance=int(settings.get("failure_tolerance", 16)),
        )

    def suggest(self, x: np.ndarray, objective: np.ndarray, iteration: int) -> tuple[np.ndarray, str]:
        if iteration < self.initial_design:
            return self.sobol_initial[iteration], "sobol"
        if self.state.restart:
            self.state.restart = False
            self.state.best = math.inf
            self.state.successes = 0
            self.state.failures = 0
            self.restart_index = len(x)
            return qmc.Sobol(self.dimension, scramble=True, seed=int(self.rng.integers(2**31))).random(1)[0], "turbo_restart"
        local_x = x[self.restart_index :]
        local_objective = objective[self.restart_index :]
        if len(local_x) < 2:
            return qmc.Sobol(self.dimension, scramble=True, seed=int(self.rng.integers(2**31))).random(1)[0], "sobol_recovery"
        gp = GaussianProcess().fit(local_x, local_objective)
        center = local_x[int(np.argmin(local_objective))]
        weights = gp.lengthscale / np.exp(np.mean(np.log(gp.lengthscale)))
        lower = np.clip(center - 0.5 * self.state.length * weights, 0.0, 1.0)
        upper = np.clip(center + 0.5 * self.state.length * weights, 0.0, 1.0)
        sobol = qmc.Sobol(self.dimension, scramble=True, seed=int(self.rng.integers(2**31)))
        exponent = math.ceil(math.log2(max(1, self.candidate_count)))
        candidates = lower + (upper - lower) * sobol.random_base2(exponent)[: self.candidate_count]
        probability = min(20.0 / self.dimension, 1.0)
        mask = self.rng.random((self.candidate_count, self.dimension)) <= probability
        empty = np.flatnonzero(~mask.any(axis=1))
        if len(empty):
            mask[empty, self.rng.integers(self.dimension, size=len(empty))] = True
        candidates = np.where(mask, candidates, center)
        sample = gp.thompson_sample(candidates, self.rng)
        return candidates[int(np.argmin(sample))], "turbo_thompson"


class CommandEvaluator:
    def __init__(self, config: dict[str, Any], builder: WorkspaceBuilder, output_dir: Path, root: Path):
        self.config = config
        self.builder = builder
        self.output_dir = output_dir
        self.root = root
        self.real_path = root / config["measurements"]["real_metrics_csv"]
        self.objective = TransportObjective(pd.read_csv(self.real_path), config)

    def evaluate(self, iteration: int, values: dict[str, float], phase: str, repetitions: int) -> dict[str, Any]:
        objectives: list[float] = []
        components: list[dict[str, float]] = []
        metrics_files: list[str] = []
        started = time.monotonic()
        for repetition in range(repetitions):
            run_dir = self.output_dir / "iterations" / f"iteration_{iteration:03d}_repetition_{repetition + 1:02d}"
            environment = os.environ.copy()
            environment.update(self.builder.build(run_dir, values))
            environment["BO_REAL_METRICS_CSV"] = str(self.real_path.resolve())
            environment["BO_PHASE"] = phase
            environment["BO_REPETITION"] = str(repetition + 1)
            try:
                result = subprocess.run(
                    self.config["evaluation"]["command"],
                    shell=True,
                    cwd=self.root,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=self.config["evaluation"].get("timeout_seconds", 7200),
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                (run_dir / "evaluation.stdout.log").write_text(error.stdout or "")
                (run_dir / "evaluation.stderr.log").write_text(error.stderr or "")
                raise RuntimeError(f"Iteration {iteration}, repetition {repetition + 1} timed out") from error
            (run_dir / "evaluation.stdout.log").write_text(result.stdout)
            (run_dir / "evaluation.stderr.log").write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"Iteration {iteration}, repetition {repetition + 1} failed; see {run_dir}")
            metrics_path = run_dir / self.config["evaluation"].get("simulated_metrics_file", "simulation_metrics.csv")
            value, detail = self.objective.score(pd.read_csv(metrics_path))
            objectives.append(value)
            components.append(detail)
            metrics_files.append(str(metrics_path))
        average_components = {name: float(np.mean([item[name] for item in components])) for name in components[0]}
        return {
            "iteration": iteration,
            "phase": phase,
            "parameters": values,
            "objective": float(np.mean(objectives)),
            "repeat_objectives": objectives,
            "transport_components": average_components,
            "runtime_seconds": time.monotonic() - started,
            "metrics_files": metrics_files,
        }


def _plot_history(history: pd.DataFrame, parameters: list[Parameter], output_dir: Path, bandwidth_mhz: int) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history.iteration, history.objective, "o-", label="transport objective")
    axis.plot(history.iteration, history.objective.cummin(), "o-", label="best so far")
    axis.set(xlabel="Configuration", ylabel="Context-aware OT objective", title=f"DTPO convergence - {bandwidth_mhz} MHz")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "dtpo_convergence.png", dpi=160)
    plt.close(figure)
    active = [parameter for parameter in parameters if parameter.optimize]
    figure, axes = plt.subplots(len(active), 1, figsize=(10, max(3, 2.3 * len(active))), sharex=True, squeeze=False)
    for axis, parameter in zip(axes[:, 0], active):
        axis.plot(history.iteration, history[parameter.name], "o-")
        axis.axhline(parameter.initial, color="gray", linestyle="--", linewidth=1)
        axis.set_ylabel(parameter.name)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Configuration")
    figure.tight_layout()
    figure.savefig(output_dir / "parameter_evolution.png", dpi=160)
    plt.close(figure)


def run(config_path: Path, output_override: str | None = None, configurations_override: int | None = None) -> Path:
    config = load_config(config_path.resolve())
    root = Path(config.get("project_root", ".")).expanduser().resolve()
    output_dir = (root / (output_override or config["output_dir"])).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = parameters_from_config(config)
    active = [parameter for parameter in parameters if parameter.optimize]
    if not active:
        raise ValueError("At least one parameter must be optimized")
    evaluator = CommandEvaluator(config, WorkspaceBuilder(parameters, root), output_dir, root)
    settings = config.get("optimizer", {})
    repetitions = int(settings.get("repetitions", 3))
    budget = int(settings.get("evaluation_budget", 480))
    configurations = configurations_override or int(settings.get("configurations", 160))
    if repetitions < 1 or budget < repetitions or configurations < 1:
        raise ValueError("Invalid repetitions, evaluation budget, or configuration count")
    configurations = min(configurations, budget // repetitions)
    optimizer = TurboOptimizer(len(active), settings)
    records: list[dict[str, Any]] = []
    history_path = output_dir / "iteration_history.json"
    if history_path.exists():
        records = json.loads(history_path.read_text(encoding="utf-8"))
    x_values = [
        np.array([(record["parameters"][item.name] - item.lower) / (item.upper - item.lower) for item in active])
        for record in records
    ]
    objectives = [float(record["objective"]) for record in records]
    penalty = float(config.get("objective", {}).get("failure_penalty", 1_000_000.0))
    restart_iterations = [record["iteration"] for record in records if record.get("phase") == "turbo_restart"]
    if restart_iterations:
        last_restart = max(restart_iterations)
        optimizer.restart_index = sum(
            record["iteration"] < last_restart and record["objective"] < penalty for record in records
        )
    else:
        last_restart = 0
    for record in records:
        value = float(record["objective"])
        if record["iteration"] < last_restart:
            continue
        if value < penalty:
            optimizer.state.update(value)
    for iteration in range(len(records), configurations):
        valid = np.array(objectives) < penalty
        fit_x = np.asarray(x_values)[valid] if valid.any() else np.empty((0, len(active)))
        fit_y = np.asarray(objectives)[valid]
        normalized, phase = optimizer.suggest(fit_x, fit_y, iteration)
        values = {parameter.name: parameter.initial for parameter in parameters}
        values.update(
            {
                parameter.name: parameter.lower + normalized[index] * (parameter.upper - parameter.lower)
                for index, parameter in enumerate(active)
            }
        )
        try:
            record = evaluator.evaluate(iteration, values, phase, repetitions)
        except Exception as error:
            record = {
                "iteration": iteration,
                "phase": phase,
                "parameters": values,
                "objective": penalty,
                "repeat_objectives": [],
                "transport_components": {},
                "runtime_seconds": 0.0,
                "metrics_files": [],
                "error": str(error),
            }
        records.append(record)
        x_values.append(normalized)
        objectives.append(float(record["objective"]))
        if record["objective"] < penalty:
            optimizer.state.update(float(record["objective"]))
        flat = [
            {
                **entry["parameters"],
                "iteration": entry["iteration"],
                "phase": entry["phase"],
                "objective": entry["objective"],
                "runtime_seconds": entry["runtime_seconds"],
            }
            for entry in records
        ]
        pd.DataFrame(flat).to_csv(output_dir / "iteration_history.csv", index=False)
        history_path.write_text(json.dumps(records, indent=2) + "\n")
    successful = [index for index, record in enumerate(records) if record["objective"] < penalty]
    if not successful:
        raise RuntimeError("No DTPO configuration completed successfully")
    best_index = min(successful, key=lambda index: records[index]["objective"])
    best = records[best_index]
    best_metrics = Path(best["metrics_files"][0])
    shutil.copy2(best_metrics, output_dir / "best_simulation_metrics.csv")
    material_dir = output_dir / "best_calibrated_materials"
    material_dir.mkdir(exist_ok=True)
    copied: list[str] = []
    best_run_dir = output_dir / "iterations" / f"iteration_{best_index:03d}_repetition_01"
    for parameter in parameters:
        source = parameter.target.get("source")
        candidate = best_run_dir / Path(source).name if source else None
        if candidate and candidate.exists() and candidate.name not in copied:
            shutil.copy2(candidate, material_dir / candidate.name)
            copied.append(candidate.name)
    history = pd.read_csv(output_dir / "iteration_history.csv")
    bandwidth_mhz = int(config.get("bandwidth_mhz", 20))
    _plot_history(history, parameters, output_dir, bandwidth_mhz)
    summary = {
        "algorithm": "dtpo_context_aware_ot_turbo_thompson",
        "bandwidth_mhz": bandwidth_mhz,
        "best_iteration": best_index,
        "best_parameters": best["parameters"],
        "best_objective": best["objective"],
        "transport_components": best["transport_components"],
        "configurations": configurations,
        "repetitions": repetitions,
        "pipeline_executions": configurations * repetitions,
        "best_calibrated_materials": [str(material_dir / name) for name in copied],
    }
    (output_dir / "best_parameters.json").write_text(json.dumps(summary, indent=2) + "\n")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("ebstr_bo_calibration.yml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--configurations", type=int)
    arguments = parser.parse_args()
    print(run(arguments.config, arguments.output_dir, arguments.configurations))


if __name__ == "__main__":
    main()
