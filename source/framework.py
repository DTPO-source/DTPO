#!/usr/bin/env python3
"""Two-order, noise-aware Bayesian black-box calibration for EBSTR."""
from __future__ import annotations

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
from calibration.metrics import (FIRST_ORDER_METRICS as FIRST_ORDER,
                                 KPI_METRICS as METRICS,
                                 SECOND_ORDER_METRICS as SECOND_ORDER)
from calibration.plotting import plot_three_way_comparison
import yaml
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm, qmc



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
    result = [Parameter(item["name"], float(item["lower_bound"]), float(item["upper_bound"]),
                        float(item["initial_value"]), bool(item.get("optimize", True)),
                        dict(item.get("target", {}))) for item in config.get("parameters", [])]
    if not result or len({parameter.name for parameter in result}) != len(result):
        raise ValueError("Parameters must be non-empty and have unique names")
    return result


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
    """Build immutable material/vegetation assets for one black-box trial."""

    def __init__(self, parameters: list[Parameter], project_root: Path):
        self.parameters, self.project_root = parameters, project_root

    def build(self, iteration_dir: Path, values: dict[str, float]) -> dict[str, str]:
        iteration_dir.mkdir(parents=True, exist_ok=True)
        documents: dict[str, tuple[Path, Any, str]] = {}
        environment: dict[str, str] = {}
        for parameter in self.parameters:
            target = parameter.target
            kind = target.get("kind", "environment")
            if kind == "environment":
                environment[target.get("name", parameter.name.upper())] = str(values[parameter.name])
                continue
            source, dotted_path = target.get("source"), target.get("path")
            if kind not in {"json", "yaml"} or not source or not dotted_path:
                raise ValueError(f"{parameter.name}: invalid asset target")
            source_path = Path(source)
            if not source_path.is_absolute():
                source_path = self.project_root / source_path
            key = str(source_path.resolve())
            if key not in documents:
                destination = iteration_dir / source_path.name
                shutil.copy2(source_path, destination)
                with destination.open(encoding="utf-8") as stream:
                    document = json.load(stream) if kind == "json" else yaml.safe_load(stream)
                documents[key] = (destination, document, kind)
            _set_path(documents[key][1], dotted_path, values[parameter.name])
        for source, (destination, document, kind) in documents.items():
            with destination.open("w", encoding="utf-8") as stream:
                if kind == "json":
                    json.dump(document, stream, indent=2)
                    stream.write("\n")
                else:
                    yaml.safe_dump(document, stream, sort_keys=False)
            environment[f"BO_INPUT_{Path(source).stem.upper().replace('-', '_')}"] = str(destination.resolve())
        environment.update({"BO_ITERATION_DIR": str(iteration_dir.resolve()),
                            "BO_PARAMETER_JSON": json.dumps(values, sort_keys=True)})
        return environment


def _rmse(real: np.ndarray, simulated: np.ndarray) -> float:
    valid = np.isfinite(real) & np.isfinite(simulated)
    if not valid.any():
        raise ValueError("No finite matching measurements")
    return float(np.sqrt(np.mean((real[valid] - simulated[valid]) ** 2)))


def score_metrics(real: pd.DataFrame, simulated: pd.DataFrame, weights: dict[str, float]) -> tuple[float, float, float, dict[str, float]]:
    merged = real.merge(simulated, on="point", suffixes=("_real", "_sim"), validate="one_to_one")
    if len(merged) != len(real):
        raise ValueError("Simulation points do not exactly cover real measurement points")
    losses = {metric: _rmse(merged[f"{metric}_real"].to_numpy(float), merged[f"{metric}_sim"].to_numpy(float)) for metric in METRICS}
    first = sum(weights.get(metric, 1.0) * losses[metric] for metric in FIRST_ORDER)
    second = sum(weights.get(metric, 1.0) * losses[metric] for metric in SECOND_ORDER)
    return first + second, first, second, losses


class GaussianProcess:
    """Matérn-5/2 ARD GP with a fixed relative observation-noise floor."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.x = np.asarray(x, float)
        self.mean = float(np.mean(y))
        self.scale = max(float(np.std(y)), 1e-6)
        self.y = (np.asarray(y, float) - self.mean) / self.scale
        pairwise = np.abs(self.x[:, None, :] - self.x[None, :, :])
        self.lengthscale = np.clip(np.median(pairwise + np.eye(len(self.x))[:, :, None], axis=(0, 1)), .08, 1.5)
        covariance = self.kernel(self.x, self.x) + np.eye(len(self.x)) * 1e-5
        self.factor = cho_factor(covariance, lower=True, check_finite=False)
        self.alpha = cho_solve(self.factor, self.y, check_finite=False)
        return self

    def kernel(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        distance = np.sqrt(np.sum(((left[:, None, :] - right[None, :, :]) / self.lengthscale) ** 2, axis=2))
        root5 = math.sqrt(5.0)
        return (1 + root5 * distance + 5 * distance**2 / 3) * np.exp(-root5 * distance)

    def predict(self, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cross = self.kernel(np.asarray(candidates, float), self.x)
        mean = cross @ self.alpha
        solved = cho_solve(self.factor, cross.T, check_finite=False)
        variance = np.maximum(1 - np.einsum("ij,ji->i", cross, solved), 1e-12)
        return self.mean + self.scale * mean, self.scale * np.sqrt(variance)


class TwoOrderBayesianOptimizer:
    """First minimize amplitude loss, then constrained noisy EI for dynamics.

    A Matérn-5/2 ARD GP is a good fit for the small, continuous, expensive and
    noisy Sim-to-Real search.  The second stage uses constrained EI: candidates
    improve the full loss only when their predicted first-order loss remains
    within a configurable tolerance of the best amplitude calibration.
    """

    def __init__(self, seed: int, candidates: int, initial_design: int, first_order_iterations: int, tolerance: float):
        self.rng = np.random.default_rng(seed)
        self.candidates, self.initial_design = candidates, initial_design
        self.first_order_iterations, self.tolerance = first_order_iterations, tolerance

    def suggest(self, x: np.ndarray, objective: np.ndarray, first: np.ndarray, initial: np.ndarray,
                iteration: int, forbidden: np.ndarray | None = None) -> tuple[np.ndarray, str]:
        if iteration == 0:
            return initial, "official_baseline"
        candidates = qmc.LatinHypercube(x.shape[1], seed=self.rng).random(self.candidates)
        feasibility = np.ones(len(candidates))
        if forbidden is not None and len(forbidden):
            nearest = np.sqrt(np.min(np.sum((candidates[:, None, :] - forbidden[None, :, :]) ** 2, axis=2), axis=1))
            feasibility = 1.0 - np.exp(-0.5 * (nearest / 0.15) ** 2)
        if iteration < self.initial_design:
            return candidates[0], "space_filling_initial_design"
        if iteration < self.first_order_iterations:
            gp = GaussianProcess().fit(x, first)
            mean, std = gp.predict(candidates)
            best = float(np.min(first))
            improvement = best - mean - .01 * max(np.std(first), 1e-6)
            z = improvement / np.maximum(std, 1e-12)
            acquisition = (improvement * norm.cdf(z) + std * norm.pdf(z)) * feasibility
            return candidates[int(np.argmax(acquisition))], "first_order_noisy_ei"
        objective_gp, first_gp = GaussianProcess().fit(x, objective), GaussianProcess().fit(x, first)
        mean, std = objective_gp.predict(candidates)
        best = float(np.min(objective))
        improvement = best - mean - .01 * max(np.std(objective), 1e-6)
        z = improvement / np.maximum(std, 1e-12)
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        first_mean, first_std = first_gp.predict(candidates)
        threshold = float(np.min(first)) * self.tolerance
        amplitude_feasibility = norm.cdf((threshold - first_mean) / np.maximum(first_std, 1e-12))
        return candidates[int(np.argmax(ei * amplitude_feasibility * feasibility))], "second_order_constrained_noisy_ei"


class CommandEvaluator:
    def __init__(self, config: dict[str, Any], builder: WorkspaceBuilder, output_dir: Path, root: Path):
        self.config, self.builder, self.output_dir, self.root = config, builder, output_dir, root
        real_path = root / config["measurements"]["real_metrics_csv"]
        self.real = pd.read_csv(real_path)
        self._check_columns(self.real, "real metrics")

    @staticmethod
    def _check_columns(frame: pd.DataFrame, label: str) -> None:
        missing = {"point", *METRICS} - set(frame.columns)
        if missing or frame.point.duplicated().any():
            raise ValueError(f"{label} missing columns or duplicate points: {sorted(missing)}")

    def evaluate(self, iteration: int, values: dict[str, float], phase: str) -> dict[str, Any]:
        directory = self.output_dir / "iterations" / f"iteration_{iteration:03d}"
        environment = os.environ.copy()
        environment.update(self.builder.build(directory, values))
        environment["BO_REAL_METRICS_CSV"] = str((self.root / self.config["measurements"]["real_metrics_csv"]).resolve())
        environment["BO_PHASE"] = phase
        started = time.monotonic()
        try:
            result = subprocess.run(self.config["evaluation"]["command"], shell=True, cwd=self.root, env=environment,
                                    text=True, capture_output=True, timeout=self.config["evaluation"].get("timeout_seconds", 7200), check=False)
        except subprocess.TimeoutExpired as error:
            (directory / "evaluation.stdout.log").write_text(error.stdout or "")
            (directory / "evaluation.stderr.log").write_text(error.stderr or "")
            raise RuntimeError(f"Iteration {iteration} timed out") from error
        (directory / "evaluation.stdout.log").write_text(result.stdout)
        (directory / "evaluation.stderr.log").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"Iteration {iteration} command failed ({result.returncode}); see {directory}")
        simulated_path = directory / self.config["evaluation"].get("simulated_metrics_file", "simulation_metrics.csv")
        simulated = pd.read_csv(simulated_path)
        self._check_columns(simulated, "simulated metrics")
        objective, first, second, losses = score_metrics(self.real, simulated, self.config.get("objective", {}).get("weights", {}))
        return {"iteration": iteration, "phase": phase, "parameters": values, "objective": objective,
                "first_order_loss": first, "second_order_loss": second, "metric_losses": losses,
                "runtime_seconds": time.monotonic() - started, "metrics_file": str(simulated_path)}


def _plot_history(history: pd.DataFrame, parameters: list[Parameter], output_dir: Path, bandwidth_mhz: int) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history.iteration, history.objective, "o-", label="objective")
    axis.plot(history.iteration, history.objective.cummin(), "o-", label="best so far")
    axis.set(xlabel="Iteration", ylabel="Weighted loss", title=f"Two-order BO convergence · {bandwidth_mhz} MHz"); axis.grid(alpha=.25); axis.legend()
    figure.tight_layout(); figure.savefig(output_dir / "bo_convergence.png", dpi=160); plt.close(figure)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(history.iteration, history.first_order_loss, "o-", label="first-order")
    axis.plot(history.iteration, history.second_order_loss, "o-", label="second-order")
    axis.plot(history.iteration, history.objective, "o-", label="total objective")
    axis.set(xlabel="Iteration", ylabel="Weighted loss", title=f"Objective evolution · {bandwidth_mhz} MHz")
    axis.grid(alpha=.25); axis.legend(); figure.tight_layout()
    figure.savefig(output_dir / "objective_evolution.png", dpi=160); plt.close(figure)
    active = [parameter for parameter in parameters if parameter.optimize]
    figure, axes = plt.subplots(len(active), 1, figsize=(10, max(3, 2.3 * len(active))), sharex=True, squeeze=False)
    for axis, parameter in zip(axes[:, 0], active):
        axis.plot(history.iteration, history[parameter.name], "o-"); axis.axhline(parameter.initial, color="gray", linestyle="--", linewidth=1)
        axis.set_ylabel(parameter.name); axis.grid(alpha=.25)
    figure.suptitle(f"Parameter evolution · {bandwidth_mhz} MHz")
    axes[-1, 0].set_xlabel("Iteration"); figure.tight_layout(); figure.savefig(output_dir / "parameter_evolution.png", dpi=160); plt.close(figure)



def run(config_path: Path, output_override: str | None = None, iterations_override: int | None = None) -> Path:
    config, config_path = load_config(config_path), config_path.resolve()
    root = Path(config.get("project_root", ".")).expanduser().resolve()
    output_dir = (root / (output_override or config["output_dir"])).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    parameters = parameters_from_config(config); active = [parameter for parameter in parameters if parameter.optimize]
    evaluator = CommandEvaluator(config, WorkspaceBuilder(parameters, root), output_dir, root)
    initial = np.array([(item.initial - item.lower) / (item.upper - item.lower) for item in active])
    settings = config.get("optimizer", {})
    optimizer = TwoOrderBayesianOptimizer(int(settings.get("seed", 20260718)), int(settings.get("candidate_count", 4096)),
                                           int(settings.get("initial_design", max(4, 2 * len(active) + 1))),
                                           int(settings.get("first_order_iterations", 10)), float(settings.get("first_order_tolerance", 1.10)))
    records: list[dict[str, Any]] = []; x_values: list[np.ndarray] = []; objective: list[float] = []; first: list[float] = []
    history_json = output_dir / "iteration_history.json"
    if history_json.exists():
        records = json.loads(history_json.read_text(encoding="utf-8"))
        for record in records:
            x_values.append(np.array([(record["parameters"][item.name] - item.lower) / (item.upper - item.lower) for item in active]))
            objective.append(float(record["objective"])); first.append(float(record["first_order_loss"]))
    max_iterations = iterations_override if iterations_override is not None else int(settings.get("iterations", 24))
    if max_iterations < 1:
        raise ValueError("iterations must be positive")
    for iteration in range(len(records), max_iterations):
        all_x = np.asarray(x_values) if x_values else initial.reshape(1, -1)
        objective_array, first_array = np.asarray(objective), np.asarray(first)
        penalty = float(config.get("objective", {}).get("failure_penalty", 1_000_000.0))
        valid = objective_array < penalty
        fit_x = all_x[valid] if valid.any() else initial.reshape(1, -1)
        fit_objective = objective_array[valid] if valid.any() else np.array([])
        fit_first = first_array[valid] if valid.any() else np.array([])
        forbidden = all_x[~valid] if len(all_x) == len(valid) else None
        normalized, phase = optimizer.suggest(fit_x, fit_objective, fit_first, initial, iteration, forbidden)
        values = {parameter.name: parameter.initial for parameter in parameters}
        values.update({parameter.name: parameter.lower + normalized[index] * (parameter.upper - parameter.lower) for index, parameter in enumerate(active)})
        try:
            record = evaluator.evaluate(iteration, values, phase)
        except Exception as error:
            penalty = float(config.get("objective", {}).get("failure_penalty", 1_000_000.0))
            record = {"iteration": iteration, "phase": phase, "parameters": values,
                      "objective": penalty, "first_order_loss": penalty,
                      "second_order_loss": penalty, "metric_losses": {},
                      "runtime_seconds": 0.0, "metrics_file": "", "error": str(error)}
        records.append(record); x_values.append(normalized); objective.append(record["objective"]); first.append(record["first_order_loss"])
        flat = [{**entry["parameters"], **{key: entry[key] for key in ("iteration", "phase", "objective", "first_order_loss", "second_order_loss", "runtime_seconds")}} for entry in records]
        pd.DataFrame(flat).to_csv(output_dir / "iteration_history.csv", index=False)
        (output_dir / "iteration_history.json").write_text(json.dumps(records, indent=2) + "\n")
    history = pd.read_csv(output_dir / "iteration_history.csv"); best_index = int(history.objective.idxmin()); best = records[best_index]
    shutil.copy2(best["metrics_file"], output_dir / "best_simulation_metrics.csv")
    material_dir = output_dir / "best_calibrated_materials"; material_dir.mkdir(exist_ok=True)
    copied: list[str] = []
    for parameter in parameters:
        source = parameter.target.get("source")
        candidate = output_dir / "iterations" / f"iteration_{best_index:03d}" / Path(source).name if source else None
        if candidate and candidate.exists() and candidate.name not in copied:
            shutil.copy2(candidate, material_dir / candidate.name); copied.append(candidate.name)
    baseline_path = config.get("baseline_metrics_csv")
    if baseline_path and (root / baseline_path).exists():
        baseline = pd.read_csv(root / baseline_path)
    elif records and records[0].get("metrics_file") and Path(records[0]["metrics_file"]).exists():
        baseline = pd.read_csv(records[0]["metrics_file"])
        baseline.to_csv(output_dir / "baseline_simulation_metrics.csv", index=False)
    else:
        baseline = None
    bandwidth_mhz = int(config.get("bandwidth_mhz", 20))
    _plot_history(history, parameters, output_dir, bandwidth_mhz)
    calibrated = pd.read_csv(best["metrics_file"])
    if baseline is None:
        raise ValueError("AODT baseline metrics are required for the three-way comparison")
    # The AODT trial CFR/CIR export has no absolute scale shared across runs, so
    # csi_magnitude straight out of a BO trial isn't comparable to Real/AODT.
    # Rescale it onto the AODT baseline's own (already-aligned) magnitude using
    # the trial-internal dB delta between the best iteration and iteration 0
    # (phase "official_baseline", the unperturbed starting point) — mirroring
    # the same fix applied to the official-calibration comparison.
    reference_metrics_file = next(
        (
            record["metrics_file"] for record in records
            if record.get("metrics_file") and Path(record["metrics_file"]).exists()
        ),
        None,
    )
    if (
        "csi_magnitude" in calibrated
        and "csi_magnitude" in baseline
        and reference_metrics_file
    ):
        reference_magnitude = pd.read_csv(reference_metrics_file)["csi_magnitude"].to_numpy(float)
        best_magnitude = calibrated["csi_magnitude"].to_numpy(float)
        delta_db = 20.0 * np.log10(
            np.clip(best_magnitude, 1e-30, None) / np.clip(reference_magnitude, 1e-30, None)
        )
        calibrated = calibrated.copy()
        calibrated["csi_magnitude"] = (
            baseline["csi_magnitude"].to_numpy(float) * np.power(10.0, delta_db / 20.0)
        )
    plot_three_way_comparison(
        evaluator.real, baseline, calibrated,
        output_dir / "figures", bandwidth_mhz,
        calibrated_label="BO",
    )
    best_summary = {"algorithm": "two_order_constrained_noisy_expected_improvement_matern52_ard", "bandwidth_mhz": bandwidth_mhz, "best_iteration": best_index,
                    "best_parameters": best["parameters"], "best_objective": best["objective"], "best_first_order_loss": best["first_order_loss"],
                    "best_second_order_loss": best["second_order_loss"], "best_calibrated_materials": [str(material_dir / name) for name in copied],
                    "best_bridge_noise": {item.name: best["parameters"][item.name] for item in parameters if item.target.get("name") == "BRIDGE_NOISE_POWER_DBFS"}}
    (output_dir / "best_parameters.json").write_text(json.dumps(best_summary, indent=2) + "\n")
    return output_dir
