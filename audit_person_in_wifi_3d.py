import argparse
import os
import shlex
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SPLITS = {"train": "train_data", "test": "test_data"}
ARTIFACTS = {
    "raw CSI": ("csi", ".mat"),
    "csi_ap": ("csi_ap", ".npy"),
    "amplitude": ("csi_amplitude", ".npy"),
    "phase": ("csi_phase", ".npy"),
    "keypoint": ("keypoint", ".npy"),
}
METADATA_SUFFIXES = {".md", ".txt", ".json", ".yaml", ".yml", ".csv"}
METADATA_TERMS = (
    "unit",
    "meter",
    "metre",
    "millimet",
    "centimet",
    "coordinate",
    "joint",
    "skeleton",
    "keypoint",
    "x-axis",
    "y-axis",
    "z-axis",
)


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only Person-in-WiFi-3D structure, file-pairing, and bounded "
            "array-content audit. Only the requested report is written."
        )
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max_samples",
        type=positive_int,
        default=6,
        help="maximum IDs sampled per split for array-content checks (default: 6)",
    )
    return parser.parse_args()


def md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def code(value):
    return "`" + str(value).replace("`", "\\`") + "`"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(md_cell(cell) for cell in row) + " |" for row in rows)
    return lines


def examples(values, limit=8):
    values = list(values)
    if not values:
        return "none"
    shown = ", ".join(code(value) for value in values[:limit])
    if len(values) > limit:
        shown += f" (+{len(values) - limit} more)"
    return shown


def number(value):
    if value is None:
        return "unknown"
    value = np.asarray(value).item()
    if isinstance(value, (float, np.floating)):
        return format(value, ".6g")
    return str(value)


def shape_text(shape):
    if shape is None:
        return "unknown"
    return code(tuple(shape))


def filename_info(sample_id):
    parts = sample_id.split("_")
    person_count = None
    if parts and len(parts[0]) >= 3:
        try:
            person_count = int(parts[0][2])
        except ValueError:
            pass
    frame_index = None
    if len(parts) >= 3:
        try:
            frame_index = int(parts[2])
        except ValueError:
            pass
    sequence_key = "_".join(parts[:2]) if len(parts) >= 2 else None
    return {
        "parts": parts,
        "person_count": person_count,
        "frame_index": frame_index,
        "sequence_key": sequence_key,
    }


def safe_sample_id(sample_id):
    if not sample_id or "\0" in sample_id:
        return False
    path = Path(sample_id)
    return not path.is_absolute() and ".." not in path.parts


def artifact_path(split_dir, artifact, sample_id):
    directory, suffix = ARTIFACTS[artifact]
    return split_dir / directory / f"{sample_id}{suffix}"


def inspect_inventory(root):
    result = {
        "directories": [],
        "total_files": 0,
        "extensions": Counter(),
        "metadata_files": [],
        "errors": [],
    }
    if not root.is_dir():
        return result

    def onerror(error):
        result["errors"].append(str(error))

    for current, dirnames, filenames in os.walk(root, onerror=onerror):
        dirnames.sort()
        filenames.sort()
        current_path = Path(current)
        relative_dir = current_path.relative_to(root)
        extensions = Counter(
            (Path(filename).suffix.lower() or "[no extension]") for filename in filenames
        )
        examples_in_dir = [
            str(relative_dir / filename) if str(relative_dir) != "." else filename
            for filename in filenames[:3]
        ]
        result["directories"].append(
            {
                "path": str(relative_dir),
                "file_count": len(filenames),
                "extensions": dict(sorted(extensions.items())),
                "examples": examples_in_dir,
            }
        )
        result["total_files"] += len(filenames)
        result["extensions"].update(extensions)
        for filename in filenames:
            path = current_path / filename
            if path.suffix.lower() not in METADATA_SUFFIXES or filename.endswith(
                "_data_list.txt"
            ):
                continue
            try:
                if path.stat().st_size <= 1024 * 1024:
                    relative = path.relative_to(root)
                    result["metadata_files"].append(str(relative))
            except OSError as error:
                result["errors"].append(str(error))
    return result


def load_sample_list(split_dir, split):
    path = split_dir / f"{split}_data_list.txt"
    result = {
        "path": path,
        "exists": path.is_file(),
        "error": None,
        "line_count": 0,
        "ids": [],
        "unique_ids": [],
        "blank_lines": [],
        "extra_token_lines": [],
        "duplicate_count": 0,
        "unsafe_ids": [],
        "malformed_ids": [],
        "filename_counts": Counter(),
    }
    if not result["exists"]:
        return result
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as error:
        result["error"] = str(error)
        return result
    result["line_count"] = len(lines)
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if not fields:
            result["blank_lines"].append(line_number)
            continue
        result["ids"].append(fields[0])
        if len(fields) > 1:
            result["extra_token_lines"].append(line_number)
    result["unique_ids"] = list(dict.fromkeys(result["ids"]))
    result["duplicate_count"] = len(result["ids"]) - len(result["unique_ids"])
    result["unsafe_ids"] = [
        sample_id for sample_id in result["unique_ids"] if not safe_sample_id(sample_id)
    ]
    for sample_id in result["unique_ids"]:
        info = filename_info(sample_id)
        result["filename_counts"][info["person_count"]] += 1
        if info["frame_index"] is None or info["sequence_key"] is None:
            result["malformed_ids"].append(sample_id)
    return result


def inspect_coverage(split_dir, sample_list):
    listed_ids = [
        sample_id for sample_id in sample_list["unique_ids"] if safe_sample_id(sample_id)
    ]
    listed_set = set(listed_ids)
    result = {}
    for artifact in ARTIFACTS:
        paths = {sample_id: artifact_path(split_dir, artifact, sample_id) for sample_id in listed_ids}
        missing = [sample_id for sample_id in listed_ids if not paths[sample_id].is_file()]
        directory = split_dir / ARTIFACTS[artifact][0]
        actual_stems = set()
        if directory.is_dir():
            try:
                actual_stems = {
                    path.stem
                    for path in directory.iterdir()
                    if path.is_file() and path.suffix.lower() == ARTIFACTS[artifact][1]
                }
            except OSError:
                actual_stems = set()
        result[artifact] = {
            "directory_exists": directory.is_dir(),
            "expected": len(listed_ids),
            "present": len(listed_ids) - len(missing),
            "missing": missing,
            "unlisted": sorted(actual_stems - listed_set),
        }
    result["listed_ids_missing_pose"] = list(result["keypoint"]["missing"])
    result["listed_csi_without_pose"] = [
        sample_id
        for sample_id in listed_ids
        if (
            artifact_path(split_dir, "raw CSI", sample_id).is_file()
            or artifact_path(split_dir, "csi_ap", sample_id).is_file()
        )
        and not artifact_path(split_dir, "keypoint", sample_id).is_file()
    ]
    return result


def select_samples(sample_ids, limit):
    if len(sample_ids) <= limit:
        return list(sample_ids)
    selected = []
    first_by_count = {}
    for sample_id in sample_ids:
        count = filename_info(sample_id)["person_count"]
        first_by_count.setdefault(count, sample_id)
    for count in sorted(first_by_count, key=lambda value: (value is None, str(value))):
        if len(selected) < limit:
            selected.append(first_by_count[count])
    indices = {
        int(round(position * (len(sample_ids) - 1) / (limit - 1)))
        if limit > 1
        else 0
        for position in range(limit)
    }
    for index in sorted(indices):
        sample_id = sample_ids[index]
        if sample_id not in selected and len(selected) < limit:
            selected.append(sample_id)
    for sample_id in sample_ids:
        if sample_id not in selected and len(selected) < limit:
            selected.append(sample_id)
    order = {sample_id: index for index, sample_id in enumerate(sample_ids)}
    return sorted(selected, key=order.get)


def empty_stats():
    return {
        "shape": None,
        "dtype": "unknown",
        "size": None,
        "nan": "unknown",
        "inf": "unknown",
        "range": "unknown",
    }


def array_stats(array):
    stats = {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "nan": 0,
        "inf": 0,
        "range": "not numeric",
    }
    if not np.issubdtype(array.dtype, np.number):
        return stats
    if np.issubdtype(array.dtype, np.inexact):
        stats["nan"] = int(np.count_nonzero(np.isnan(array)))
        stats["inf"] = int(np.count_nonzero(np.isinf(array)))
    finite = np.isfinite(array)
    if not np.any(finite):
        stats["range"] = "no finite values"
        return stats
    values = array[finite]
    if np.issubdtype(array.dtype, np.complexfloating):
        magnitude = np.abs(values)
        finite_magnitude = magnitude[np.isfinite(magnitude)]
        magnitude_range = (
            f"[{number(finite_magnitude.min())}, {number(finite_magnitude.max())}]"
            if finite_magnitude.size
            else "no finite magnitudes"
        )
        stats["range"] = (
            f"real [{number(values.real.min())}, {number(values.real.max())}]; "
            f"imag [{number(values.imag.min())}, {number(values.imag.max())}]; "
            f"magnitude {magnitude_range}"
        )
    else:
        stats["range"] = f"[{number(values.min())}, {number(values.max())}]"
    return stats


def base_inspection(sample_id, path):
    return {
        "sample_id": sample_id,
        "path": str(path),
        "stored_shape": None,
        "stored_dtype": "unknown",
        "interpreted_shape": None,
        "stats": empty_stats(),
        "attributes": {},
        "error": None,
    }


def inspect_npy(sample_id, path):
    row = base_inspection(sample_id, path)
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.ndarray):
            close = getattr(array, "close", None)
            if close is not None:
                close()
            raise ValueError("file did not contain a single NumPy array")
        row["stats"] = array_stats(array)
        row["stored_shape"] = row["stats"]["shape"]
        row["stored_dtype"] = row["stats"]["dtype"]
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def inspect_raw_csi(sample_id, path):
    row = base_inspection(sample_id, path)
    try:
        import h5py
    except ImportError:
        row["error"] = "h5py is not installed; raw HDF5 content was not inspected"
        return row
    try:
        with h5py.File(path, "r") as handle:
            if "csi_out" not in handle:
                row["error"] = f"csi_out not found; top-level keys: {list(handle.keys())}"
                return row
            dataset = handle["csi_out"]
            if not isinstance(dataset, h5py.Dataset):
                row["error"] = "csi_out is not an HDF5 dataset"
                return row
            row["stored_shape"] = tuple(dataset.shape)
            row["stored_dtype"] = str(dataset.dtype)
            row["attributes"] = {
                str(key): repr(value)[:300] for key, value in dataset.attrs.items()
            }
            stored = dataset[()]
            names = dataset.dtype.names or ()
            if "real" not in names or "imag" not in names:
                row["stats"] = array_stats(stored)
                row["error"] = "csi_out lacks the compound real/imag fields used by process.py"
                return row
            complex_data = stored["real"] + stored["imag"] * 1j
            if complex_data.ndim != 4:
                row["stats"] = array_stats(complex_data)
                row["error"] = "csi_out is not four-dimensional; processor transpose not applied"
                return row
            interpreted = np.array(complex_data).transpose(3, 2, 1, 0).astype(np.complex128)
            row["interpreted_shape"] = tuple(interpreted.shape)
            row["stats"] = array_stats(interpreted)
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def inspect_normalization(sample_id, path):
    row = {
        "sample_id": sample_id,
        "shape": None,
        "amplitude_range": "unknown",
        "amplitude_delta": "unknown",
        "phase_range": "unknown",
        "phase_delta": "unknown",
        "constant": "unknown",
        "notes": "",
    }
    try:
        array = np.load(path, allow_pickle=False)
        row["shape"] = tuple(array.shape)
        if array.ndim < 2 or array.shape[1] <= 90:
            row["notes"] = "cannot split axis 1 into 90 amplitude and remaining phase channels"
            return row
        if not np.issubdtype(array.dtype, np.number):
            row["notes"] = "non-numeric array"
            return row
        parts = {"amplitude": array[:, :90, ...], "phase": array[:, 90:, ...]}
        constant = []
        notes = []
        for name, part in parts.items():
            minimum = np.min(part)
            maximum = np.max(part)
            delta = maximum - minimum
            row[f"{name}_range"] = f"[{number(minimum)}, {number(maximum)}]"
            row[f"{name}_delta"] = number(delta)
            if bool(np.asarray(delta == 0)):
                constant.append(name)
            if not bool(np.all(np.isfinite(part))):
                notes.append(f"{name} contains NaN/Inf")
        row["constant"] = ", ".join(constant) if constant else "none"
        row["notes"] = "; ".join(notes) if notes else "none"
    except Exception as error:
        row["notes"] = f"{type(error).__name__}: {error}"
    return row


def inspect_process_compatibility(sample_id, split_dir):
    row = {
        "sample_id": sample_id,
        "combined_shape": None,
        "csi_ap_shape": None,
        "layout": "unknown",
        "exact": "unknown",
        "allclose": "unknown",
        "error": None,
    }
    try:
        amplitude = np.load(
            artifact_path(split_dir, "amplitude", sample_id), allow_pickle=False
        )
        phase = np.load(artifact_path(split_dir, "phase", sample_id), allow_pickle=False)
        csi_ap = np.load(artifact_path(split_dir, "csi_ap", sample_id), allow_pickle=False)
        combined = np.concatenate((amplitude, phase), axis=1)
        row["combined_shape"] = tuple(combined.shape)
        row["csi_ap_shape"] = tuple(csi_ap.shape)
        row["layout"] = "yes" if combined.shape == (3, 180, 20) else "no"
        if combined.shape != csi_ap.shape:
            row["exact"] = "no"
            row["allclose"] = "no"
            row["error"] = "shape mismatch"
            return row
        try:
            row["exact"] = "yes" if np.array_equal(combined, csi_ap, equal_nan=True) else "no"
        except TypeError:
            row["exact"] = "yes" if np.array_equal(combined, csi_ap) else "no"
        try:
            row["allclose"] = (
                "yes"
                if np.allclose(combined, csi_ap, rtol=1e-5, atol=1e-8, equal_nan=True)
                else "no"
            )
        except (TypeError, ValueError) as error:
            row["allclose"] = "unavailable"
            row["error"] = str(error)
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def sequence_analysis(sample_ids):
    groups = defaultdict(list)
    malformed = []
    for sample_id in sample_ids:
        info = filename_info(sample_id)
        if info["sequence_key"] is None or info["frame_index"] is None:
            malformed.append(sample_id)
            continue
        groups[info["sequence_key"]].append(info["frame_index"])
    gap_examples = []
    groups_with_gaps = 0
    missing_indices = 0
    duplicate_indices = 0
    for key, indices in sorted(groups.items()):
        unique = sorted(set(indices))
        duplicate_indices += len(indices) - len(unique)
        if not unique:
            continue
        missing = sorted(set(range(unique[0], unique[-1] + 1)) - set(unique))
        if missing:
            groups_with_gaps += 1
            missing_indices += len(missing)
            gap_examples.append(f"{key}: {missing[:10]}")
    return {
        "groups": groups,
        "malformed": malformed,
        "gap_examples": gap_examples,
        "groups_with_gaps": groups_with_gaps,
        "missing_indices": missing_indices,
        "duplicate_indices": duplicate_indices,
    }


def training_without_next(sample_ids):
    listed = set(sample_ids)
    excluded = []
    for sample_id in sample_ids:
        info = filename_info(sample_id)
        if info["sequence_key"] is None or info["frame_index"] is None:
            continue
        next_id = f"{info['sequence_key']}_{info['frame_index'] + 1}"
        if next_id not in listed:
            excluded.append(sample_id)
    return excluded


def metadata_evidence(root, inventory):
    evidence = []
    for relative in inventory["metadata_files"]:
        path = root / relative
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            evidence.append((str(relative), 0, f"read error: {error}"))
            continue
        for line_number, line in enumerate(lines, start=1):
            if any(term in line.lower() for term in METADATA_TERMS):
                evidence.append((str(relative), line_number, line.strip()[:400]))
                if len(evidence) >= 60:
                    return evidence
    return evidence


def audit_split(root, split, max_samples):
    split_dir = root / SPLITS[split]
    sample_list = load_sample_list(split_dir, split)
    result = {
        "directory": split_dir,
        "list": sample_list,
        "coverage": None,
        "sample_ids": [],
        "inspections": {},
        "normalization": [],
        "compatibility": [],
        "sequence": None,
        "without_next": [],
    }
    if not sample_list["exists"] or sample_list["error"]:
        return result
    result["coverage"] = inspect_coverage(split_dir, sample_list)
    safe_ids = [sample_id for sample_id in sample_list["unique_ids"] if safe_sample_id(sample_id)]
    result["sample_ids"] = select_samples(safe_ids, max_samples)
    result["sequence"] = sequence_analysis(safe_ids)
    if split == "train":
        result["without_next"] = training_without_next(safe_ids)
    for artifact in ARTIFACTS:
        rows = []
        for sample_id in result["sample_ids"]:
            path = artifact_path(split_dir, artifact, sample_id)
            if not path.is_file():
                continue
            if artifact == "raw CSI":
                rows.append(inspect_raw_csi(sample_id, path))
            else:
                rows.append(inspect_npy(sample_id, path))
        result["inspections"][artifact] = rows
    for sample_id in result["sample_ids"]:
        csi_ap_path = artifact_path(split_dir, "csi_ap", sample_id)
        if csi_ap_path.is_file():
            result["normalization"].append(inspect_normalization(sample_id, csi_ap_path))
        required = [
            artifact_path(split_dir, artifact, sample_id).is_file()
            for artifact in ("amplitude", "phase", "csi_ap")
        ]
        if all(required):
            result["compatibility"].append(
                inspect_process_compatibility(sample_id, split_dir)
            )
    return result


def add_inventory(lines, inventory):
    lines.extend(["## Actual directory inventory", ""])
    lines.append(f"- Files found: **{inventory['total_files']}**.")
    extension_text = ", ".join(
        f"{code(extension)}: {count}"
        for extension, count in sorted(inventory["extensions"].items())
    )
    lines.append(f"- Extensions: {extension_text or 'none'}.")
    if inventory["errors"]:
        lines.append(f"- Traversal errors: {examples(inventory['errors'], 3)}")
    lines.append("")
    rows = []
    for directory in inventory["directories"][:100]:
        extensions = ", ".join(
            f"{extension}={count}"
            for extension, count in directory["extensions"].items()
        )
        rows.append(
            [
                code(directory["path"]),
                directory["file_count"],
                extensions or "none",
                examples(directory["examples"], 3),
            ]
        )
    if rows:
        lines.extend(table(["Directory", "Files", "Extensions", "Example files"], rows))
        if len(inventory["directories"]) > 100:
            lines.append("")
            lines.append(
                f"Directory table truncated after 100 of {len(inventory['directories'])} directories."
            )
    else:
        lines.append("No readable directories or files were found.")


def add_code_findings(lines):
    lines.extend(["## Code-derived findings (not real-data measurements)", ""])
    lines.extend(
        [
            "- `feeder/person_in_wifi_3d.py:58-59` expects `csi_ap/<ID>.npy` and `keypoint/<ID>.npy` beneath each split.",
            "- `data/person_in_wifi_3d/process.py:77-87` reads HDF5 `csi/<ID>.mat` variable `csi_out`, constructs amplitude and sanitized phase, and concatenates them on axis 1.",
            "- `data/person_in_wifi_3d/process.py:89-102` writes `csi_amplitude` and `csi_phase`; its `csi_ap` save block is commented out. The shipped processor therefore does not create the feeder's required `csi_ap` files.",
            "- The processor's hard-coded reshape implies a code-level `(3, 90, 20)` amplitude/phase layout and `(3, 180, 20)` combined layout. These are expectations, not verified on-disk shapes.",
            "- `feeder/person_in_wifi_3d.py:116-120` normalizes the first 90 and remaining channels independently with no zero-denominator guard.",
            "- `feeder/person_in_wifi_3d.py:62-73` keeps a training item only when its immediately following ID is in the filtered list; `feeder/person_in_wifi_3d.py:99-111` does not apply that rule to test items.",
            "- `feeder/person_in_wifi_3d.py:151-157` pads poses to three people, 14 joints, and three coordinates. This code expectation does not establish actual pose schema, joint ordering, or units.",
            "- Filename-derived person counts in this report use the feeder's third-character convention and are audit-only. This script is not imported by any training or unlabelled loader.",
        ]
    )


def absent_report(root, output, inventory, max_samples):
    lines = [
        "# Person-in-WiFi-3D Dataset Audit",
        "",
        f"- Data root: {code(root)}",
        f"- Requested report: {code(output)}",
        "- Status: **dataset content absent; real-data audit not performed**.",
        "- No dataset files were downloaded, created, modified, preprocessed, or used for training.",
        "",
        "## Dataset placement and rerun",
        "",
        "Place an existing Person-in-WiFi-3D package under the supplied data root, or pass its actual location with `--data_root`. The expected top-level split names are `train_data` and `test_data`. No files need to be generated by this audit.",
        "",
        "Rerun:",
        "",
        "```bash",
        f"python audit_person_in_wifi_3d.py --data_root {shlex.quote(str(root))} --output {shlex.quote(str(output))} --max_samples {max_samples}",
        "```",
        "",
    ]
    add_inventory(lines, inventory)
    lines.extend(["", "## Requested real-data findings", ""])
    rows = [
        ["Train/test list contents and sample counts", "not measured; split lists are absent"],
        ["Raw CSI, csi_ap, amplitude, phase, and keypoint existence", "not measured; no split content is present"],
        ["CSI/pose shapes, dtypes, ranges, NaN/Inf, and normalization inputs", "not measured; no real arrays were loaded"],
        ["Listed CSI-to-pose pairing", "not measured; no listed samples are available"],
        ["Recording/sequence identifiers and frame continuity", "unknown; no sample IDs are available"],
        ["Pose joint count, coordinate dimensions, units, and conventions", "unknown; no pose arrays or unit metadata are available"],
        ["Empirical process.py output versus feeder input", "not measured; no processed arrays are available"],
    ]
    lines.extend(table(["Finding", "Status"], rows))
    lines.append("")
    add_code_findings(lines)
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "Only the directory inventory and source-code compatibility statements above are factual findings for the current local state. All real dataset schema and value questions remain unverified.",
        ]
    )
    return "\n".join(lines) + "\n"


def add_split_list_and_coverage(lines, split, result):
    sample_list = result["list"]
    lines.extend(["", f"### {split}", ""])
    lines.append(f"- Split directory: {code(result['directory'])}")
    lines.append(f"- List path: {code(sample_list['path'])}")
    if not sample_list["exists"]:
        lines.append("- List status: **missing**; sample and pairing counts are unavailable.")
        return
    if sample_list["error"]:
        lines.append(f"- List read error: {sample_list['error']}")
        return
    lines.extend(
        [
            f"- Physical lines: {sample_list['line_count']}; listed entries: **{len(sample_list['ids'])}**; unique IDs: **{len(sample_list['unique_ids'])}**; duplicate entries: {sample_list['duplicate_count']}.",
            f"- Blank lines: {examples(sample_list['blank_lines'])}; lines with extra tokens: {examples(sample_list['extra_token_lines'])}.",
            f"- Representative listed IDs: {examples(sample_list['unique_ids'], 6)}",
            f"- IDs unsafe to resolve beneath the split: {examples(sample_list['unsafe_ids'])}.",
            f"- IDs not matching the feeder's sequence/frame convention: {examples(sample_list['malformed_ids'])}.",
        ]
    )
    counts = []
    for count, amount in sorted(
        sample_list["filename_counts"].items(),
        key=lambda item: (item[0] is None, str(item[0])),
    ):
        label = "unparseable" if count is None else str(count)
        counts.append(f"{label}: {amount}")
    lines.append(f"- Filename-derived person-count distribution (unique IDs): {', '.join(counts) or 'none'}.")
    coverage = result["coverage"]
    rows = []
    for artifact in ARTIFACTS:
        item = coverage[artifact]
        rows.append(
            [
                artifact,
                "yes" if item["directory_exists"] else "no",
                item["expected"],
                item["present"],
                len(item["missing"]),
                len(item["unlisted"]),
                examples(item["missing"], 4),
            ]
        )
    lines.append("")
    lines.extend(
        table(
            ["Artifact", "Directory", "Expected", "Present", "Missing", "Unlisted", "Missing examples"],
            rows,
        )
    )
    lines.extend(
        [
            "",
            f"- Listed IDs missing keypoint files: **{len(coverage['listed_ids_missing_pose'])}**; examples: {examples(coverage['listed_ids_missing_pose'], 5)}.",
            f"- Listed IDs with raw CSI or csi_ap but no keypoint: **{len(coverage['listed_csi_without_pose'])}**; examples: {examples(coverage['listed_csi_without_pose'], 5)}.",
        ]
    )


def add_array_content(lines, split, result):
    lines.extend(["", f"### {split} sampled arrays", ""])
    lines.append(
        f"Selected {len(result['sample_ids'])} unique IDs: {examples(result['sample_ids'], 10)}. Statistics below apply only to successfully read sampled files."
    )
    lines.append("")
    rows = []
    for artifact in ARTIFACTS:
        for row in result["inspections"].get(artifact, []):
            stats = row["stats"]
            notes = row["error"] or "none"
            if row["attributes"]:
                notes += "; csi_out attrs: " + repr(row["attributes"])
            rows.append(
                [
                    artifact,
                    code(row["sample_id"]),
                    shape_text(row["stored_shape"]),
                    code(row["stored_dtype"]),
                    shape_text(row["interpreted_shape"]),
                    stats["range"],
                    stats["nan"],
                    stats["inf"],
                    notes,
                ]
            )
    if rows:
        lines.extend(
            table(
                ["Artifact", "ID", "Stored shape", "Stored dtype", "Interpreted shape", "Finite value range", "NaN", "Inf", "Notes"],
                rows,
            )
        )
    else:
        lines.append("No sampled arrays were available to read.")


def add_normalization(lines, split, result):
    lines.extend(["", f"### {split} feeder normalization inputs", ""])
    rows = []
    for row in result["normalization"]:
        rows.append(
            [
                code(row["sample_id"]),
                shape_text(row["shape"]),
                row["amplitude_range"],
                row["amplitude_delta"],
                row["phase_range"],
                row["phase_delta"],
                row["constant"],
                row["notes"],
            ]
        )
    if rows:
        lines.extend(
            table(
                ["ID", "csi_ap / normalized shape", "Amplitude range", "Amplitude delta", "Phase range", "Phase delta", "Constant inputs", "Notes"],
                rows,
            )
        )
        lines.append("")
        lines.append("A zero delta identifies a constant normalization input. The audit does not divide by it or save a normalized array.")
    else:
        lines.append("No sampled `csi_ap` arrays were available for normalization checks.")


def add_pose_findings(lines, split, result):
    lines.extend(["", f"### {split} pose structure", ""])
    rows = []
    joint_counts = Counter()
    coordinate_counts = Counter()
    for row in result["inspections"].get("keypoint", []):
        shape = row["stats"]["shape"]
        persons = joints = coordinates = "unknown"
        agreement = "unknown"
        if shape is not None and len(shape) == 3:
            persons, joints, coordinates = shape
            joint_counts[joints] += 1
            coordinate_counts[coordinates] += 1
            filename_count = filename_info(row["sample_id"])["person_count"]
            if filename_count is not None:
                agreement = "yes" if persons == filename_count else "no"
        rows.append(
            [
                code(row["sample_id"]),
                shape_text(shape),
                persons,
                joints,
                coordinates,
                agreement,
                row["error"] or "none",
            ]
        )
    if rows:
        lines.extend(
            table(
                ["ID", "Pose shape", "People", "Joints", "Coordinates", "Matches filename count", "Notes"],
                rows,
            )
        )
        lines.append("")
        lines.append(f"- Sampled joint-count distribution: {dict(sorted(joint_counts.items())) or 'unknown'}.")
        lines.append(f"- Sampled coordinate-dimension distribution: {dict(sorted(coordinate_counts.items())) or 'unknown'}.")
    else:
        lines.append("No sampled pose arrays were available; pose structure is unknown.")
    lines.extend(
        [
            "- Coordinate units: **unknown** unless independently established by validated dataset documentation.",
            "- Joint ordering, axis directions, coordinate origin, and world/camera/body convention: **unknown** unless independently established by validated dataset documentation.",
        ]
    )


def add_sequence_findings(lines, results):
    lines.extend(["", "## Filename-derived sequence identifiers and frame continuity", ""])
    lines.append("Candidate sequence keys are the first two underscore-separated filename fields, matching the feeder's next-ID construction. Their recording, subject, and environment semantics are **unknown** without explicit metadata.")
    lines.append("")
    rows = []
    for split, result in results.items():
        sequence = result["sequence"]
        if sequence is None:
            rows.append([split, "unavailable", "unavailable", "unavailable", "unavailable", "unavailable", "unavailable"])
            continue
        group_sizes = [len(indices) for indices in sequence["groups"].values()]
        rows.append(
            [
                split,
                len(group_sizes),
                f"{min(group_sizes)}–{max(group_sizes)}" if group_sizes else "none",
                len(sequence["malformed"]),
                sequence["duplicate_indices"],
                sequence["groups_with_gaps"],
                sequence["missing_indices"],
            ]
        )
    lines.extend(
        table(
            ["Split", "Candidate groups", "Group size range", "Unparseable IDs", "Duplicate frame indices", "Groups with gaps", "Missing interior indices"],
            rows,
        )
    )
    for split, result in results.items():
        sequence = result["sequence"]
        if sequence is None:
            continue
        lines.extend(
            [
                "",
                f"- {split} candidate keys: {examples(sorted(sequence['groups']), 8)}.",
                f"- {split} gap examples: {examples(sequence['gap_examples'], 5)}.",
            ]
        )
        if split == "train":
            lines.append(
                f"- Training entries excluded by the feeder's immediate-next-ID rule on the unfiltered list: **{len(result['without_next'])}**; examples: {examples(result['without_next'], 6)}. Experiment-specific person-count filtering can change this count."
            )
        else:
            lines.append("- Test entries are not removed by an immediate-next-ID requirement in the current feeder.")
    train = results["train"]["sequence"]
    test = results["test"]["sequence"]
    if train is not None and test is not None:
        overlap = sorted(set(train["groups"]) & set(test["groups"]))
        lines.extend(
            [
                "",
                f"Candidate sequence-key overlap between train and test: **{len(overlap)}**; examples: {examples(overlap, 8)}.",
            ]
        )
    train_ids = set(results["train"]["list"]["unique_ids"])
    test_ids = set(results["test"]["list"]["unique_ids"])
    id_overlap = sorted(train_ids & test_ids)
    lines.append(f"Exact listed-ID overlap between train and test: **{len(id_overlap)}**; examples: {examples(id_overlap, 8)}.")


def add_compatibility(lines, results):
    lines.extend(["", "## Empirical process.py output versus feeder input", ""])
    lines.append("For sampled IDs with amplitude, phase, and csi_ap all present, the audit concatenates amplitude and phase in memory on axis 1 and compares them with csi_ap. `allclose` uses `rtol=1e-5`, `atol=1e-8`, and equal NaN positions. No processor or feeder code is executed.")
    lines.append("")
    rows = []
    for split, result in results.items():
        for row in result["compatibility"]:
            rows.append(
                [
                    split,
                    code(row["sample_id"]),
                    shape_text(row["combined_shape"]),
                    shape_text(row["csi_ap_shape"]),
                    row["layout"],
                    row["exact"],
                    row["allclose"],
                    row["error"] or "none",
                ]
            )
    if rows:
        lines.extend(
            table(
                ["Split", "ID", "Combined shape", "csi_ap shape", "Code-expected (3,180,20)", "Exact", "Allclose", "Notes"],
                rows,
            )
        )
    else:
        lines.append("No complete sampled amplitude/phase/csi_ap triplets were available; empirical compatibility is **not measured**. The static incompatibility documented above remains confirmed.")


def full_report(root, output, inventory, results, max_samples):
    lines = [
        "# Person-in-WiFi-3D Dataset Audit",
        "",
        f"- Data root: {code(root)}",
        f"- Requested report: {code(output)}",
        "- Status: **read-only audit completed; per-file findings and errors are reported below**.",
        f"- Array content was bounded to at most {max_samples} selected IDs per split. List parsing and expected-file existence checks cover every unique listed ID; value statistics are not dataset-wide.",
        "- No dataset files were downloaded, created, modified, preprocessed, or used for training.",
        "",
    ]
    add_inventory(lines, inventory)
    lines.extend(["", "## Train/test lists and expected-file coverage", ""])
    for split, result in results.items():
        add_split_list_and_coverage(lines, split, result)
    lines.extend(["", "## Sampled CSI and pose array content", ""])
    lines.append("NaN/Inf counts apply to the stored NumPy arrays and to the processor-interpreted complex raw arrays, as identified by each row. Ranges exclude NaN/Inf values.")
    for split, result in results.items():
        add_array_content(lines, split, result)
    lines.extend(["", "## Feeder normalization checks", ""])
    for split, result in results.items():
        add_normalization(lines, split, result)
    lines.extend(["", "## Pose array structure and conventions", ""])
    for split, result in results.items():
        add_pose_findings(lines, split, result)
    evidence = metadata_evidence(root, inventory)
    lines.extend(["", "### Potential metadata evidence", ""])
    if evidence:
        lines.append("These exact snippets are leads requiring manual validation; their presence alone does not establish units or conventions.")
        lines.append("")
        lines.extend(
            table(
                ["File", "Line", "Snippet"],
                [[code(path), line_number, snippet] for path, line_number, snippet in evidence],
            )
        )
    else:
        lines.append("No readable metadata snippets matching unit, coordinate, joint, or skeleton terms were found. Units and conventions remain **unknown**.")
    add_sequence_findings(lines, results)
    add_compatibility(lines, results)
    lines.append("")
    add_code_findings(lines)
    return "\n".join(lines) + "\n"


def dataset_content_present(root):
    return any((root / split_dir).exists() for split_dir in SPLITS.values()) or any(
        (root / split_dir / f"{split}_data_list.txt").is_file()
        for split, split_dir in SPLITS.items()
    )


def main():
    args = parse_args()
    root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(root):
        raise SystemExit("Refusing to write the report inside --data_root; choose an output outside the dataset.")
    inventory = inspect_inventory(root)
    if dataset_content_present(root):
        results = {
            split: audit_split(root, split, args.max_samples) for split in SPLITS
        }
        report = full_report(root, output, inventory, results, args.max_samples)
    else:
        report = absent_report(root, output, inventory, args.max_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Audit report written to: {output}")
    if not dataset_content_present(root):
        print(f"Person-in-WiFi-3D dataset content was not found at: {root}")
        print("Place the existing dataset under train_data/ and test_data/, or pass its actual root with --data_root.")
        print("No dataset files were created or modified. Rerun the same audit command after supplying the dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
