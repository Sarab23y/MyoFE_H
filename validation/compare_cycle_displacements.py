#!/usr/bin/env python3
"""Compare existing healthy-cycle displacement fields without altering them.

The script reads DOLFIN's temporal solution XDMF/HDF5 through meshio, selects
the saved snapshot nearest each requested cycle phase, removes the best-fit
rigid motion, compares affine deformation gradients on tetrahedra, and maps
requested Quadrature IDs by their saved reference coordinates (never by mesh
node number).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_CYCLES = ((37.484, 38.316),
                  (38.316, 39.147),
                  (39.147, 39.978))
DEFAULT_PHASES = (0.10, 0.30, 0.50, 0.70)
DEFAULT_HOTSPOTS = (1643, 1694, 2404, 4751, 4844, 5004)


def best_fit_rigid(reference: np.ndarray,
                   target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return proper rotation R and translation t for reference @ R + t."""
    ref_center = reference.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (reference - ref_center).T @ (target - target_center)
    left, unused, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    translation = target_center - ref_center @ rotation
    return rotation, translation


def affine_tetra_gradient(points: np.ndarray, displacement: np.ndarray,
                          tetrahedra: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return F=I+grad(u) and validity for linear tetra vertex data."""
    gradients = np.full((len(tetrahedra), 3, 3), np.nan)
    valid = np.zeros(len(tetrahedra), dtype=bool)
    for index, tet in enumerate(tetrahedra):
        vertices = np.asarray(tet[:4], dtype=int)
        x = points[vertices]
        u = displacement[vertices]
        reference_edges = np.column_stack((x[1] - x[0],
                                           x[2] - x[0],
                                           x[3] - x[0]))
        if abs(np.linalg.det(reference_edges)) < 1.0e-14:
            continue
        displacement_edges = np.column_stack((u[1] - u[0],
                                              u[2] - u[0],
                                              u[3] - u[0]))
        gradients[index] = np.eye(3) + displacement_edges @ np.linalg.inv(
            reference_edges)
        valid[index] = True
    return gradients, valid


def containing_tetra(point: np.ndarray, points: np.ndarray,
                     tetrahedra: np.ndarray, tolerance: float = 1.0e-8):
    """Find a containing tetrahedron using barycentric coordinates."""
    for index, tet in enumerate(tetrahedra):
        ids = np.asarray(tet[:4], dtype=int)
        x = points[ids]
        matrix = np.column_stack((x[1] - x[0], x[2] - x[0], x[3] - x[0]))
        try:
            tail = np.linalg.solve(matrix, point - x[0])
        except np.linalg.LinAlgError:
            continue
        barycentric = np.concatenate(([1.0 - tail.sum()], tail))
        if np.all(barycentric >= -tolerance) and np.all(
                barycentric <= 1.0 + tolerance):
            return int(index), [float(value) for value in barycentric]
    return None, None


def nearest_index(times: np.ndarray, wanted: float) -> int:
    return int(np.argmin(np.abs(times - wanted)))


def select_displacement(point_data: Dict[str, np.ndarray], requested: str):
    if requested:
        if requested not in point_data:
            raise KeyError("displacement field %r not found; available: %s" %
                           (requested, sorted(point_data)))
        return requested, np.asarray(point_data[requested], dtype=float)
    candidates = []
    for name, values in point_data.items():
        array = np.asarray(values)
        if array.ndim == 2 and array.shape[1] in (2, 3):
            score = 0 if "displacement" in name.lower() else 1
            candidates.append((score, name, array))
    if not candidates:
        raise ValueError("no vector point field found; available: %s" %
                         sorted(point_data))
    unused, name, array = sorted(candidates, key=lambda item: item[:2])[0]
    if array.shape[1] == 2:
        array = np.column_stack((array, np.zeros(len(array))))
    return name, np.asarray(array, dtype=float)


def read_time_series(path: Path, field: str):
    import meshio
    with meshio.xdmf.TimeSeriesReader(str(path)) as reader:
        points, cells = reader.read_points_cells()
        frames = []
        chosen_name = None
        for step in range(reader.num_steps):
            time, point_data, unused_cell_data = reader.read_data(step)
            name, displacement = select_displacement(point_data, field)
            if chosen_name is None:
                chosen_name = name
            elif name != chosen_name:
                raise ValueError("displacement field changed across steps")
            if len(displacement) != len(points):
                raise ValueError("displacement/mesh point counts differ")
            frames.append((float(time), displacement.copy()))
    tetrahedra = None
    for block in cells:
        if block.type in ("tetra", "tetra10"):
            tetrahedra = np.asarray(block.data, dtype=int)
            break
    if tetrahedra is None:
        raise ValueError("XDMF contains no tetra/tetra10 cells")
    return np.asarray(points, dtype=float), tetrahedra, chosen_name, frames


def load_hotspots(path: Path, ids: Sequence[int]):
    coordinates = np.load(str(path))
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 2 or coordinates.shape[1] < 3:
        raise ValueError("quadrature coordinate NPY must have shape (N, >=3)")
    result = {}
    for global_id in ids:
        if global_id < 0 or global_id >= len(coordinates):
            raise ValueError("hotspot DOF %d is outside coordinate array" %
                             global_id)
        result[int(global_id)] = coordinates[global_id, :3].astype(float)
    return result


def summary_metrics(first: np.ndarray, second: np.ndarray):
    difference = second - first
    norm_first = float(np.linalg.norm(first))
    return {
        "absolute_l2": float(np.linalg.norm(difference)),
        "relative_l2": float(np.linalg.norm(difference) /
                             max(norm_first, np.finfo(float).eps)),
        "maximum_point_difference": float(np.linalg.norm(
            difference, axis=1).max()),
        "mean_point_difference": float(np.linalg.norm(
            difference, axis=1).mean())}


def analyze(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    points, tetrahedra, field_name, frames = read_time_series(
        Path(args.xdmf), args.displacement_field)
    times = np.asarray([item[0] for item in frames])
    if len(np.unique(times)) != len(times):
        raise ValueError("XDMF contains duplicate timestamps")
    order = np.argsort(times)
    times = times[order]
    displacements = [frames[index][1] for index in order]
    finite = bool(np.isfinite(points).all() and
                  all(np.isfinite(value).all() for value in displacements))
    if not finite:
        raise ValueError("mesh or displacement contains nonfinite values")

    hotspot_coordinates = {}
    if args.quadrature_coordinates:
        hotspot_coordinates = load_hotspots(
            Path(args.quadrature_coordinates), args.hotspot_dofs)
    hotspot_elements = {}
    for global_id, coordinate in hotspot_coordinates.items():
        cell, barycentric = containing_tetra(coordinate, points, tetrahedra)
        hotspot_elements[global_id] = {
            "coordinate": coordinate.tolist(),
            "containing_tetrahedron": cell,
            "barycentric_coordinates": barycentric}

    rows = []
    comparisons = []
    figures = []
    for phase in args.phases:
        selected = []
        for cycle_index, (start, stop) in enumerate(args.cycles):
            wanted = start + phase
            index = nearest_index(times, wanted)
            actual = float(times[index])
            phase_error = actual - wanted
            selected.append((index, actual, displacements[index]))
            rows.append({"phase": phase, "cycle": cycle_index + 1,
                         "wanted_time": wanted, "saved_time": actual,
                         "phase_error": phase_error,
                         "within_cycle": bool(start <= actual <= stop)})
        for first_cycle in range(2):
            second_cycle = first_cycle + 1
            first_index, first_time, first_u = selected[first_cycle]
            second_index, second_time, second_u = selected[second_cycle]
            first_x = points + first_u
            second_x = points + second_u
            rotation, translation = best_fit_rigid(first_x, second_x)
            aligned = first_x @ rotation + translation
            rigid_residual = second_x - aligned
            first_f, first_valid = affine_tetra_gradient(
                points, first_u, tetrahedra)
            second_f, second_valid = affine_tetra_gradient(
                points, second_u, tetrahedra)
            valid = first_valid & second_valid
            f_difference = np.linalg.norm(second_f[valid] - first_f[valid],
                                          axis=(1, 2))
            j_first = np.linalg.det(first_f[valid])
            j_second = np.linalg.det(second_f[valid])
            item = {
                "phase": float(phase),
                "cycle_pair": [first_cycle + 1, second_cycle + 1],
                "times": [first_time, second_time],
                "phase_errors": [rows[-3 + first_cycle]["phase_error"],
                                  rows[-3 + second_cycle]["phase_error"]],
                "displacement": summary_metrics(first_u, second_u),
                "best_fit_rigid": {
                    "rotation": rotation.tolist(),
                    "translation": translation.tolist(),
                    "residual_l2": float(np.linalg.norm(rigid_residual)),
                    "residual_maximum": float(np.linalg.norm(
                        rigid_residual, axis=1).max())},
                "tetrahedral_deformation": {
                    "valid_cell_count": int(valid.sum()),
                    "F_difference_l2": float(np.linalg.norm(f_difference)),
                    "F_difference_maximum": float(f_difference.max()),
                    "minimum_J_first": float(j_first.min()),
                    "minimum_J_second": float(j_second.min())},
                "hotspots": {}}
            for global_id, metadata in hotspot_elements.items():
                cell = metadata["containing_tetrahedron"]
                hotspot = dict(metadata)
                if cell is not None and valid[cell]:
                    hotspot.update({
                        "F_difference": float(np.linalg.norm(
                            second_f[cell] - first_f[cell])),
                        "J_first": float(np.linalg.det(first_f[cell])),
                        "J_second": float(np.linalg.det(second_f[cell]))})
                item["hotspots"][str(global_id)] = hotspot
            comparisons.append(item)

            if not args.no_plots:
                import matplotlib.pyplot as plt
                magnitude = np.linalg.norm(second_u - first_u, axis=1)
                figure = plt.figure(figsize=(10, 7))
                axis = figure.add_subplot(111, projection="3d")
                image = axis.scatter(points[:, 0], points[:, 1], points[:, 2],
                                     c=magnitude, s=5, cmap="viridis")
                figure.colorbar(image, ax=axis,
                                label="|u(second)-u(first)|")
                axis.set_title("Phase %.2f s, cycles %d-%d" %
                               (phase, first_cycle + 1, second_cycle + 1))
                name = "displacement_phase_%.2f_cycles_%d_%d.png" % (
                    phase, first_cycle + 1, second_cycle + 1)
                figure.tight_layout()
                figure.savefig(str(output / name), dpi=180)
                plt.close(figure)
                figures.append(name)

    maximum_allowed = args.save_frequency / 2.0 + args.time_tolerance
    phase_selection_passed = all(
        abs(row["phase_error"]) <= maximum_allowed and row["within_cycle"]
        for row in rows)
    report = {
        "test": "existing_cycle_displacement_comparison",
        "source_xdmf": str(Path(args.xdmf).resolve()),
        "source_files_modified": False,
        "displacement_field": field_name,
        "mesh_point_count": int(len(points)),
        "tetrahedron_count": int(len(tetrahedra)),
        "saved_time_count": int(len(times)),
        "saved_time_range": [float(times.min()), float(times.max())],
        "declared_save_frequency_seconds": args.save_frequency,
        "maximum_allowed_phase_error_seconds": maximum_allowed,
        "finite": finite,
        "phase_selection_passed": phase_selection_passed,
        "passed": bool(finite and phase_selection_passed),
        "pass_scope": ("I/O, correspondence, finiteness, and phase selection; "
                       "mechanical differences are reported, not thresholded"),
        "selected_snapshots": rows,
        "comparisons": comparisons,
        "plots": figures}
    with open(output / "displacement_cycle_comparison.json", "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    with open(output / "selected_snapshots.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Displacement comparison: %s" %
          ("PASS" if report["passed"] else "FAIL"))
    print("Report: %s" %
          (output / "displacement_cycle_comparison.json"))
    return 0 if report["passed"] else 1


def parse_cycle(value: str):
    start, stop = value.split(":", 1)
    start, stop = float(start), float(stop)
    if stop <= start:
        raise argparse.ArgumentTypeError("cycle stop must exceed start")
    return start, stop


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--xdmf", required=True)
    parser.add_argument("--output", required=True,
                        help="new directory; existing paths are refused")
    parser.add_argument("--displacement-field", default="")
    parser.add_argument("--cycle", dest="cycles", action="append",
                        type=parse_cycle)
    parser.add_argument("--phase", dest="phases", action="append", type=float)
    parser.add_argument("--save-frequency", type=float, default=0.1)
    parser.add_argument("--time-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--quadrature-coordinates",
                        help="quadrature_dof.npy ordered by global q-DOF")
    parser.add_argument("--hotspot-dof", dest="hotspot_dofs", action="append",
                        type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    args.cycles = tuple(args.cycles or DEFAULT_CYCLES)
    args.phases = tuple(args.phases or DEFAULT_PHASES)
    args.hotspot_dofs = tuple(args.hotspot_dofs or DEFAULT_HOTSPOTS)
    if len(args.cycles) != 3:
        parser.error("exactly three --cycle values are required")
    if args.save_frequency <= 0.0:
        parser.error("--save-frequency must be positive")
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
