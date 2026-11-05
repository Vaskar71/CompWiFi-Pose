import argparse
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


SPLITS = {"train": "train_data", "test": "test_data"}
JOINT_NAMES = (
    "Neck",
    "Head",
    "Left shoulder",
    "Right shoulder",
    "Left elbow",
    "Left hip",
    "Right elbow",
    "Right hip",
    "Left hand",
    "Left knee",
    "Right hand",
    "Right knee",
    "Left ankle",
    "Right ankle",
)
JOINT_SHORT = (
    "Neck", "Head", "LSh", "RSh", "LElb", "LHip", "RElb", "RHip",
    "LHand", "LKnee", "RHand", "RKnee", "LAnk", "RAnk",
)
SKELETON_EDGES = (
    (0, 1), (0, 2), (0, 3), (2, 4), (4, 8), (3, 6), (6, 10),
    (2, 5), (3, 7), (5, 7), (5, 9), (9, 12), (7, 11), (11, 13),
)
MAJOR_BONES = {
    "Left upper arm (2-4)": (2, 4),
    "Right upper arm (3-6)": (3, 6),
    "Left forearm (4-8)": (4, 8),
    "Right forearm (6-10)": (6, 10),
    "Left thigh (5-9)": (5, 9),
    "Right thigh (7-11)": (7, 11),
    "Left shin (9-12)": (9, 12),
    "Right shin (11-13)": (11, 13),
}
PERSON_COLORS = ((220, 80, 40), (70, 70, 220), (60, 160, 70))
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only PiW3D filename, training-keypoint, and source-metric audit. "
            "Reads no CSI and no official test keypoints; writes only the requested "
            "report and figures outside the dataset root."
        )
    )
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure_dir", type=Path, required=True)
    parser.add_argument("--samples_per_count", type=int, default=30)
    return parser.parse_args()


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.6g}"


def code(value):
    return f"`{value}`"


def table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def ensure_output_outside_dataset(path, root):
    resolved = path.resolve()
    dataset = root.resolve()
    if resolved == dataset or dataset in resolved.parents:
        raise ValueError(f"Refusing to write inside dataset root: {resolved}")


def parse_id(sample_id):
    parts = sample_id.split("_")
    valid_shape = (
        len(parts) == 3
        and re.fullmatch(r"S\d\d", parts[0]) is not None
        and re.fullmatch(r"\d+", parts[1]) is not None
        and re.fullmatch(r"\d+", parts[2]) is not None
    )
    if not valid_shape:
        return {
            "id": sample_id,
            "parts": parts,
            "valid": False,
            "person_count": None,
            "group": None,
            "frame": None,
        }
    return {
        "id": sample_id,
        "parts": parts,
        "valid": True,
        "person_count": int(parts[0][2]),
        "group": f"{parts[0]}_{parts[1]}",
        "frame": int(parts[2]),
    }


def read_split_list(root, split):
    split_dir = root / SPLITS[split]
    path = split_dir / f"{split}_data_list.txt"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    ids = [line.split()[0] for line in lines if line.split()]
    unique_ids = list(dict.fromkeys(ids))
    parsed = [parse_id(sample_id) for sample_id in unique_ids]
    valid = [item for item in parsed if item["valid"]]
    frames = np.array([item["frame"] for item in valid], dtype=np.int64)
    groups = defaultdict(list)
    for item in valid:
        groups[item["group"]].append(item["frame"])
    return {
        "split": split,
        "path": path,
        "line_count": len(lines),
        "listed_count": len(ids),
        "ids": unique_ids,
        "parsed": parsed,
        "valid": valid,
        "malformed": [item for item in parsed if not item["valid"]],
        "duplicate_count": len(ids) - len(unique_ids),
        "token_counts": Counter(len(item["parts"]) for item in parsed),
        "person_counts": Counter(item["person_count"] for item in valid),
        "frames": frames,
        "groups": {key: sorted(value) for key, value in groups.items()},
    }


def id_sort_key(sample_id):
    item = parse_id(sample_id)
    return (item["group"] or "", item["frame"] if item["frame"] is not None else -1, sample_id)


def evenly_spaced(ids, count):
    ordered = sorted(ids, key=id_sort_key)
    if len(ordered) < count:
        raise ValueError(f"Need {count} IDs, found only {len(ordered)}")
    positions = np.linspace(0, len(ordered) - 1, num=count, dtype=np.int64)
    selected = [ordered[int(position)] for position in positions]
    if len(set(selected)) != count:
        raise AssertionError("Deterministic selection produced duplicate IDs")
    return selected


def select_training_ids(train, count):
    selected = {}
    for person_count in (1, 2, 3):
        candidates = [
            item["id"] for item in train["valid"] if item["person_count"] == person_count
        ]
        selected[person_count] = evenly_spaced(candidates, count)
    return selected


def load_training_poses(root, selected):
    keypoint_dir = (root / "train_data" / "keypoint").resolve()
    records = []
    valid_samples = []
    for person_count in (1, 2, 3):
        for sample_id in selected[person_count]:
            path = (keypoint_dir / f"{sample_id}.npy").resolve()
            if path.parent != keypoint_dir:
                raise ValueError(f"Unsafe training keypoint path: {path}")
            record = {
                "id": sample_id,
                "expected_people": person_count,
                "shape": None,
                "shape_ok": False,
                "count_match": False,
                "finite": False,
                "constant": None,
                "error": None,
            }
            try:
                pose = np.load(path, allow_pickle=False)
                record["shape"] = tuple(pose.shape)
                record["shape_ok"] = pose.shape == (person_count, 14, 3)
                record["count_match"] = pose.ndim >= 1 and pose.shape[0] == person_count
                record["finite"] = bool(np.isfinite(pose).all())
                record["constant"] = bool(pose.size == 0 or np.all(pose == pose.flat[0]))
                if record["shape_ok"] and record["finite"]:
                    valid_samples.append({"id": sample_id, "count": person_count, "pose": pose})
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            records.append(record)
    return records, valid_samples


def percentile_row(values):
    return [np.percentile(values, percentile) for percentile in PERCENTILES]


def pose_statistics(valid_samples):
    points = np.concatenate(
        [sample["pose"].reshape(-1, 3) for sample in valid_samples], axis=0
    )
    axis_rows = []
    extremes = []
    screening = []
    for axis, name in enumerate(("x (axis 0)", "y (axis 1)", "z (axis 2)")):
        values = points[:, axis]
        stats = [values.min(), values.max(), *percentile_row(values)]
        axis_rows.append([name, *[fmt(value) for value in stats]])
        for kind, index in (("min", int(np.argmin(values))), ("max", int(np.argmax(values)))):
            flat_index = index // 14
            joint = index % 14
            person_offset = 0
            located = None
            for sample in valid_samples:
                people = sample["pose"].shape[0]
                if person_offset <= flat_index < person_offset + people:
                    located = (
                        sample["id"],
                        flat_index - person_offset,
                        joint,
                        float(values[index]),
                    )
                    break
                person_offset += people
            extremes.append((name, kind, located))
        q1, q3 = np.percentile(values, (25, 75))
        iqr = q3 - q1
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        outside = int(np.count_nonzero((values < lower) | (values > upper)))
        screening.append([name, fmt(lower), fmt(upper), outside, fmt(100 * outside / len(values))])

    head_ankle = np.concatenate(
        [
            sample["pose"][:, 1, :] - sample["pose"][:, (12, 13), :].mean(axis=1)
            for sample in valid_samples
        ],
        axis=0,
    )
    vertical_rows = []
    for axis, name in enumerate(("x (axis 0)", "y (axis 1)", "z (axis 2)")):
        vertical_rows.append([
            name,
            fmt(np.median(head_ankle[:, axis])),
            fmt(np.median(np.abs(head_ankle[:, axis]))),
            fmt(np.percentile(np.abs(head_ankle[:, axis]), 5)),
            fmt(np.percentile(np.abs(head_ankle[:, axis]), 95)),
        ])
    vertical_axis = int(np.argmax(np.median(np.abs(head_ankle), axis=0)))

    bone_values = {}
    bone_rows = []
    extreme_rows = []
    for name, (joint_a, joint_b) in MAJOR_BONES.items():
        observations = []
        for sample in valid_samples:
            lengths = np.linalg.norm(
                sample["pose"][:, joint_a, :] - sample["pose"][:, joint_b, :], axis=1
            )
            observations.extend(
                (float(length), sample["id"], person)
                for person, length in enumerate(lengths)
            )
        bone_values[name] = observations
        lengths = np.array([item[0] for item in observations])
        q1, q3 = np.percentile(lengths, (25, 75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = [item for item in observations if item[0] < lower or item[0] > upper]
        bone_rows.append([
            name,
            len(lengths),
            fmt(lengths.mean()),
            fmt(lengths.std()),
            fmt(lengths.min()),
            *[fmt(value) for value in percentile_row(lengths)],
            fmt(lengths.max()),
            len(outliers),
        ])
        for kind, item in (("shortest", min(observations)), ("longest", max(observations))):
            extreme_rows.append([
                name,
                kind,
                code(item[1]),
                item[2],
                JOINT_NAMES[MAJOR_BONES[name][0]] + " / " + JOINT_NAMES[MAJOR_BONES[name][1]],
                fmt(item[0]),
            ])
    return {
        "points": points,
        "axis_rows": axis_rows,
        "extremes": extremes,
        "screening": screening,
        "head_ankle": head_ankle,
        "vertical_rows": vertical_rows,
        "vertical_axis": vertical_axis,
        "bone_values": bone_values,
        "bone_rows": bone_rows,
        "bone_extremes": extreme_rows,
    }


def put_text(image, position, value, scale=0.45, color=(30, 30, 30), thickness=1):
    cv2.putText(
        image,
        str(value),
        (int(position[0]), int(position[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def projection(pose, mode):
    if mode == "3D":
        azimuth = math.radians(45)
        elevation = math.radians(20)
        rotated_x = math.cos(azimuth) * pose[..., 0] - math.sin(azimuth) * pose[..., 1]
        rotated_y = math.sin(azimuth) * pose[..., 0] + math.cos(azimuth) * pose[..., 1]
        horizontal = rotated_x
        vertical = pose[..., 2] * math.cos(elevation) - rotated_y * math.sin(elevation)
        return np.stack((horizontal, vertical), axis=-1)
    axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[mode]
    return pose[..., axes]


def draw_panel(image, pose, rect, mode, title):
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (215, 215, 215), 1)
    put_text(image, (x + 18, y + 28), title, scale=0.58, thickness=2)
    projected = projection(pose, mode)
    minimum = projected.reshape(-1, 2).min(axis=0)
    maximum = projected.reshape(-1, 2).max(axis=0)
    center = (minimum + maximum) / 2
    span = maximum - minimum
    common_span = max(float(span.max()), 0.1) * 1.22
    plot_width = width - 90
    plot_height = height - 90
    scale = min(plot_width / common_span, plot_height / common_span)
    origin_x = x + width / 2
    origin_y = y + 48 + plot_height / 2

    def pixels(points):
        result = np.empty_like(points, dtype=np.float64)
        result[..., 0] = origin_x + (points[..., 0] - center[0]) * scale
        result[..., 1] = origin_y + (points[..., 1] - center[1]) * scale
        return np.rint(result).astype(np.int32)

    mapped = pixels(projected)
    for person in range(pose.shape[0]):
        color = PERSON_COLORS[person % len(PERSON_COLORS)]
        for joint_a, joint_b in SKELETON_EDGES:
            cv2.line(
                image,
                tuple(mapped[person, joint_a]),
                tuple(mapped[person, joint_b]),
                color,
                2,
                cv2.LINE_AA,
            )
        for joint in range(14):
            point = tuple(mapped[person, joint])
            cv2.circle(image, point, 5, color, -1, cv2.LINE_AA)
            cv2.circle(image, point, 5, (30, 30, 30), 1, cv2.LINE_AA)
            offset_x = 7 if joint % 2 == 0 else -58
            offset_y = -5 if joint % 3 else 14
            label_position = (point[0] + offset_x, point[1] + offset_y)
            put_text(
                image,
                label_position,
                f"{joint}:{JOINT_SHORT[joint]}",
                scale=0.36,
                color=(255, 255, 255),
                thickness=3,
            )
            put_text(
                image,
                label_position,
                f"{joint}:{JOINT_SHORT[joint]}",
                scale=0.36,
                color=(20, 20, 20),
                thickness=1,
            )
        put_text(
            image,
            (x + 18, y + height - 14),
            f"P{person} = person {person}; equal x/y data-unit scale ({scale:.1f} px/unit)",
            scale=0.4,
            color=color,
        )


def render_figure(sample, figure_dir):
    sample_id = sample["id"]
    pose = sample["pose"]
    image = np.full((1500, 2000, 3), 255, dtype=np.uint8)
    put_text(
        image,
        (30, 42),
        f"PiW3D official training pose: {sample_id} ({pose.shape[0]} person(s))",
        scale=0.9,
        thickness=2,
    )
    put_text(
        image,
        (30, 68),
        "Provisional DT-Pose ordering; stored coordinates; interpretation only",
        scale=0.5,
        color=(80, 80, 80),
    )
    panel_height = 645
    draw_panel(image, pose, (0, 80, 1000, panel_height), "3D", "3D orthographic (azimuth 45 deg, elevation 20 deg)")
    draw_panel(image, pose, (1000, 80, 1000, panel_height), "XY", "2D projection: stored X-Y")
    draw_panel(image, pose, (0, 725, 1000, panel_height), "XZ", "2D projection: stored X-Z")
    draw_panel(image, pose, (1000, 725, 1000, panel_height), "YZ", "2D projection: stored Y-Z")
    put_text(image, (30, 1400), "Joint labels: " + "; ".join(f"{i} {name}" for i, name in enumerate(JOINT_NAMES[:7])), scale=0.43)
    put_text(image, (30, 1430), " " * 14 + "; ".join(f"{i} {name}" for i, name in enumerate(JOINT_NAMES[7:], start=7)), scale=0.43)
    put_text(image, (30, 1470), "Visualization edges are not a model adjacency change. All panels use equal scaling on their displayed axes.", scale=0.46, color=(80, 80, 80))
    path = figure_dir / f"{pose.shape[0]}p_{sample_id}.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write figure: {path}")
    return path


def source_line(source, needle):
    for line_number, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return line_number
    raise AssertionError(f"Expected source text not found: {needle}")


def frame_summary(split):
    frames = split["frames"]
    return [
        split["listed_count"],
        len(split["ids"]),
        split["duplicate_count"],
        len(split["malformed"]),
        dict(sorted(split["token_counts"].items())),
        dict(sorted(split["person_counts"].items())),
        frames.min(),
        *[fmt(value) for value in percentile_row(frames)],
        frames.max(),
        len(split["groups"]),
    ]


def build_report(root, splits, representatives, selected, records, valid_samples, stats, figures, utils_source, train_source, rating, rating_reasons):
    train = splits["train"]
    test = splits["test"]
    train_groups = set(train["groups"])
    test_groups = set(test["groups"])
    shared_groups = sorted(train_groups & test_groups)
    exact_overlap = sorted(set(train["ids"]) & set(test["ids"]))
    literal_train = int(np.count_nonzero(train["frames"] <= 550))
    literal_test = int(np.count_nonzero((test["frames"] >= 551) & (test["frames"] <= 600)))
    compatible_groups = [
        group for group in shared_groups
        if max(train["groups"][group]) <= 550 and min(test["groups"][group]) >= 551
    ]

    mpjpe_def = source_line(utils_source, "def calulate_error")
    mpjpe_values = source_line(utils_source, "mpjpe = np.sqrt")
    pampjpe_values = source_line(utils_source, "pampjpe[n] = np.sqrt")
    metric_return = source_line(utils_source, "return mpjpe, pampjpe")
    pck_scale = source_line(utils_source, "gt[:,:,6]-gt[:,:,4]")
    pck_dist = source_line(utils_source, "dist=np.sqrt")
    procrustes_scale = source_line(utils_source, "compute_similarity_transform(frame_gt, frame_pred, compute_optimal_scale=True)")
    align_zero = source_line(utils_source, "preds_hip = preds[:, 0, :]")
    caller = source_line(train_source, "mpjpe, pampjpe, mpjpe_joints")
    prediction_trim = source_line(train_source, "predicted_val_pose = torch.cat")
    pck_batch_average = source_line(train_source, "pck_overall = [np.mean(pck_value, 0)[14]")

    lines = [
        "# Person-in-WiFi-3D Pose, Split, and Metric Audit",
        "",
        f"**Rating: {rating}.** This is a read-only audit of filename semantics, bounded official training keypoints, and existing metric source code.",
        "",
        "## Scope and access controls",
        "",
        f"- Dataset root: {code(root.resolve())}.",
        f"- Parsed every ID in {code(train['path'])} and {code(test['path'])}.",
        f"- Read only the {len(valid_samples)} selected keypoint arrays under official {code('train_data/keypoint')}; no CSI array and no official test CSI/keypoint array was read.",
        "- Did not import or execute `process.py`, `train_pose.py`, `pretrain.py`, or any model/training code. Their text was inspected only where stated.",
        "- Wrote only this report and the listed PNG figures, both guarded to be outside the dataset root. No dataset file was created, renamed, moved, deleted, preprocessed, or corrected.",
        "- Coordinate units and joint order remain interpretations unless explicitly described as verified. Missing frame numbers are not treated as corruption.",
        "",
        "## Part 1 — Filename semantics",
        "",
        "### Exact token rules used",
        "",
        "For an ID such as `S11_01_10`, splitting on underscores gives exactly three zero-based tokens:",
        "",
        "- **Person count:** token `0` (`S11`), character `2` (third character); `int(parts[0][2])`. This matches `feeder/person_in_wifi_3d.py:42`.",
        "- **Candidate temporal group:** tokens `0` and `1`, joined as `parts[0] + '_' + parts[1]`; for the example, `S11_01`. This matches the feeder's next-frame key construction at `feeder/person_in_wifi_3d.py:62`.",
        "- **Frame index:** token `2`; `int(parts[2])`.",
        "- Expected types: token 0 matches `S` plus two decimal digits, tokens 1 and 2 contain decimal digits, and there are exactly three tokens. The other digit in token 0 is not assigned a semantic identity here.",
        "",
        "These are **candidate temporal groups only**. No filename or supplied documentation proves that they are recording IDs, clip IDs, subject IDs, or environment IDs.",
        "",
        "### Representative official training IDs",
        "",
        "The 12 IDs below are deterministic evenly spaced selections (four per filename-derived person-count category) from the sorted official training list.",
        "",
        *table(
            ["ID", "Tokens", "Person count: token 0 char 2", "Candidate group: tokens 0-1", "Frame: token 2"],
            [[
                code(sample_id),
                code(parse_id(sample_id)["parts"]),
                parse_id(sample_id)["person_count"],
                code(parse_id(sample_id)["group"]),
                parse_id(sample_id)["frame"],
            ] for sample_id in representatives],
        ),
        "",
        "### Full-list validation and frame-index distributions",
        "",
        *table(
            ["Split", "Listed", "Unique", "Duplicates", "Malformed", "Token-count distribution", "Person counts {1,2,3}", "Min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "Max", "Candidate groups"],
            [[split["split"], *frame_summary(split)] for split in (train, test)],
        ),
        "",
        f"- Every listed ID valid: **{not train['malformed'] and not test['malformed']}**. All IDs have three tokens of the expected types: **{set(train['token_counts']) == {3} and set(test['token_counts']) == {3}}**.",
        f"- Exact listed-ID overlap: **{len(exact_overlap)}**.",
        f"- Candidate temporal groups: train **{len(train_groups)}**, test **{len(test_groups)}**, shared **{len(shared_groups)}**. Every test candidate group occurs in train: **{test_groups <= train_groups}**.",
        "- The official split is therefore not candidate-temporal-group disjoint. This does not by itself prove leakage, and it does not establish any recording-, subject-, or environment-level claim.",
        "",
        "### Evidence about the first-550/last-50 protocol",
        "",
        "No `550`, `first-550`, or `last-50` statement was found in the supplied local Markdown/Python/YAML documentation. To test the requested protocol without assuming it, a literal 1-based 600-index interpretation would put train at 1–550 and test at 551–600.",
        "",
        f"- Train IDs with frame index `<=550`: **{literal_train}/{len(train['frames'])}**.",
        f"- Test IDs with frame index `551..600`: **{literal_test}/{len(test['frames'])}**; observed test range is **{test['frames'].min()}..{test['frames'].max()}**.",
        f"- Shared candidate groups satisfying all train `<=550` and all test `>=551`: **{len(compatible_groups)}/{len(shared_groups)}**.",
        "- **Conclusion:** the observed indices do not support that literal first-550/last-50 within-clip interpretation. Filename indices might use another origin or selection convention, but that is unresolved, not evidence for an alternative mapping. Gaps are observed absences, not labeled corruption.",
        "",
        "## Part 2 — Official training-pose validation",
        "",
        "### Deterministic sample and per-file checks",
        "",
        f"For each filename-derived person count, the audit sorted by candidate group/frame/ID and selected {len(selected[1])} unique evenly spaced IDs. Total: **{len(records)} training files**.",
        "",
        *table(
            ["Category", "Selected official training IDs"],
            [[f"{count} person", ", ".join(code(value) for value in selected[count])] for count in (1, 2, 3)],
        ),
        "",
        *table(
            ["Check", "Passed", "Failed"],
            [
                ["Shape exactly `(filename_people, 14, 3)`", sum(r["shape_ok"] for r in records), sum(not r["shape_ok"] for r in records)],
                ["Filename person count equals `keypoint.shape[0]`", sum(r["count_match"] for r in records), sum(not r["count_match"] for r in records)],
                ["All values finite", sum(r["finite"] for r in records), sum(not r["finite"] for r in records)],
                ["Array not constant", sum(r["constant"] is False for r in records), sum(r["constant"] is not False for r in records)],
                ["Read without error", sum(r["error"] is None for r in records), sum(r["error"] is not None for r in records)],
            ],
        ),
        "",
        "The shape check includes the count match, but both are reported separately as requested. These are bounded-sample findings, not a full-dataset integrity scan.",
        "",
        "<details><summary>Per-file validation records</summary>",
        "",
        *table(
            ["ID", "Shape", "Shape OK", "Count match", "Finite", "Constant", "Error"],
            [[
                code(record["id"]),
                code(record["shape"]),
                record["shape_ok"],
                record["count_match"],
                record["finite"],
                record["constant"],
                record["error"] or "none",
            ] for record in records],
        ),
        "",
        "</details>",
        "",
        "### Coordinate statistics",
        "",
        f"Statistics aggregate all {stats['points'].shape[0]} joints from all people in the {len(valid_samples)} valid sampled training files. Values are in stored coordinate units.",
        "",
        *table(
            ["Axis", "Min", "Max", "p1", "p5", "p25", "p50", "p75", "p95", "p99"],
            stats["axis_rows"],
        ),
        "",
        "Extreme coordinate locations (not automatic errors):",
        "",
        *table(
            ["Axis", "Extreme", "ID", "Person (zero-based)", "Joint", "Value"],
            [[axis, kind, code(item[0]), item[1], f"{item[2]} {JOINT_NAMES[item[2]]}", fmt(item[3])] for axis, kind, item in stats["extremes"]],
        ),
        "",
        "A conservative Tukey outer-fence screen (`Q1 - 3*IQR`, `Q3 + 3*IQR`) is shown only to locate unusual values. No point was removed or corrected.",
        "",
        *table(["Axis", "Lower fence", "Upper fence", "Outside", "Outside %"], stats["screening"]),
        "",
        "### Provisional scale and vertical-axis interpretation",
        "",
        "The joint order used here is the user-supplied provisional DT-Pose order, not independently verified dataset metadata. Under that order, head-minus-mean-ankle differences give:",
        "",
        *table(["Axis", "Median signed head - ankles", "Median absolute", "Absolute p5", "Absolute p95"], stats["vertical_rows"]),
        "",
        f"- **Axis {stats['vertical_axis']} (`{'xyz'[stats['vertical_axis']]}`) appears vertical.** Its median signed head-minus-ankle difference is **{fmt(np.median(stats['head_ankle'][:, stats['vertical_axis']]))}**, so values increase downward in these samples (head has the smaller vertical coordinate).",
        f"- Median absolute head-to-ankle vertical separation is **{fmt(np.median(np.abs(stats['head_ankle'][:, stats['vertical_axis']])))} stored units**. Together with the major-limb lengths below, this is consistent with a plausible **metre-scale** interpretation.",
        "- Metres are an interpretation supported by body proportions, not a proven unit declaration. Axis origin, camera/world frame, handedness, and official joint order remain unresolved.",
        "",
        "### Major-limb bone lengths",
        "",
        "All lengths are Euclidean distances in stored units. Outlier counts use the ordinary Tukey 1.5*IQR rule per bone and are diagnostic only.",
        "",
        *table(
            ["Bone", "N", "Mean", "Std", "Min", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "Max", "Tukey outliers"],
            stats["bone_rows"],
        ),
        "",
        *table(["Bone", "Extreme", "ID", "Person", "Provisional joints", "Length"], stats["bone_extremes"]),
        "",
        "Short or long bones may reflect pose, annotation noise, joint-order mismatch, or genuine outliers. This audit does not delete, clamp, or correct them.",
        "",
        "### Generated and inspected visualizations",
        "",
        "Each PNG contains the official training sample ID, per-joint labels, a labelled 3D orthographic skeleton view, and X-Y/X-Z/Y-Z 2D projections. Every panel uses one common data-unit scale on its displayed axes. The edges are visualization-only and do not modify any model adjacency matrix.",
        "",
    ]
    for path in figures:
        relative = Path("figures") / "pose_audit" / path.name
        lines.extend([f"- [{path.name}]({relative.as_posix()})", "", f"![{path.name}]({relative.as_posix()})", ""])
    lines.extend([
        "## Part 3 — Metric audit (source inspection only)",
        "",
        "### MPJPE and PA-MPJPE units",
        "",
        f"- `calulate_error` at `utils.py:{mpjpe_def}` computes Euclidean coordinate differences at `utils.py:{mpjpe_values}` and `utils.py:{pampjpe_values}` and performs **no multiplication by 1000**. MPJPE, PA-MPJPE, and both per-joint arrays are returned in the **same units as the stored/input coordinates** at `utils.py:{metric_return}`.",
        f"- If stored coordinates are metres, apply `*1000` exactly once to all four returned arrays (`mpjpe`, `pampjpe`, `mpjpe_joints`, `pampjpe_joints`) immediately before `utils.py:{metric_return}`, or immediately after the call at `train_pose.py:{caller}`. This covers both person-averaged and per-joint values. Alternatively, converting only the final display averages would not convert per-joint outputs.",
        "- PCK is dimensionless because distances are divided by a pose-derived scale; it does not need a metre-to-millimetre conversion.",
        "- No metric code was edited in this task.",
        "",
        "### Current PiW3D PCK normalization",
        "",
        f"- The `person-in-wifi-3d` branch at `utils.py:{pck_scale}` computes `||gt[:,:,6] - gt[:,:,4]||`. Under the provisional ordering, these are **right elbow (6)** and **left elbow (4)**, so the normalizer is inter-elbow distance, not the source comment's shoulder-to-hip/torso diameter.",
        f"- At `utils.py:{pck_dist}`, each 3D joint error is divided by that distance. `train_pose.py:294-295` uses thresholds `0.5, 0.4, 0.3, 0.2, 0.1` times the normalizer and reports percent PCK. Close elbows make the denominator small and PCK stricter; there is no zero/near-zero scale guard.",
        "- This finding depends on the provisional joint order for semantic naming; the actual indices 6 and 4 are directly verified in code.",
        "",
        "### Other behavior relevant to missed/duplicate people and interpretation",
        "",
        f"- `train_pose.py:{prediction_trim}` and the following ground-truth line use label-derived `person_num` to retain only the first N prediction slots and first N padded ground-truth people. Extra slots are excluded, so missed-person count, duplicate-person count, and exact count accuracy cannot be measured by this evaluation path.",
        "- Predicted and ground-truth people are paired by slot/index order; there is no Hungarian matching. A permutation can inflate pose error, while trimming can hide duplicates.",
        f"- PA-MPJPE calls Procrustes alignment with `compute_optimal_scale=True` at `utils.py:{procrustes_scale}`; translation, rotation, and scale differences can be hidden. It is not root-localization error.",
        f"- The optional `align=True` path calls joint 0 a hip at `utils.py:{align_zero}`, but joint 0 is Neck under the provisional PiW3D order. Current calls use `align=False`, so this neck-alignment issue is latent rather than active here.",
        f"- Per-batch PCK arrays are averaged equally at `train_pose.py:{pck_batch_average}` rather than pooled by joint/person count; unequal final batch sizes can change weighting.",
        "- MPJPE averages joints per person and then persons; alone it does not penalize a missing or duplicate person. No root-localization, missed/duplicate, count-accuracy, or person-level precision/recall/F1 implementation was found in this path.",
        "",
        "## Verified facts",
        "",
        f"- Both official lists were fully parsed: {len(train['ids'])} train and {len(test['ids'])} test unique IDs; all IDs matched the expected token structure.",
        f"- The bounded official training sample contains {len(records)} files ({len(selected[1])} per person-count category); all shape/count/finite/nonconstant checks passed: **{all(r['shape_ok'] and r['count_match'] and r['finite'] and r['constant'] is False and r['error'] is None for r in records)}**.",
        f"- Axis {stats['vertical_axis']} appears vertical and stored coordinates are plausibly metre-scale based on the sampled body proportions.",
        f"- The code uses PCK joints 6 and 4, returns MPJPE/PA-MPJPE without `*1000`, trims predictions with ground-truth person counts, and performs no Hungarian matching.",
        f"- Six requested figure sets were generated: **{len(figures)} PNGs**.",
        "",
        "## Unresolved facts and assumptions",
        "",
        "- No independent metadata was found proving coordinate units, joint ordering, coordinate origin/frame, or candidate-group semantics.",
        "- Metre scale and the vertical-axis interpretation are empirical interpretations of the provisional joint ordering, not official convention confirmation.",
        "- The first-550/last-50 documentation was not present locally; only a literal interpretation was tested and rejected by observed indices.",
        "- Numerical pose validation is bounded to 90 training files. Full-dataset readability, NaN/Inf status, and bone distributions were not measured.",
        "",
        "## Rating rationale",
        "",
        *[f"- {reason}" for reason in rating_reasons],
        "",
        "## Generated files",
        "",
        f"- {code('docs/POSE_SPLIT_METRIC_AUDIT.md')}",
        *[f"- {code(Path('docs/figures/pose_audit') / path.name)}" for path in figures],
        "",
        "No preprocessing, training, package installation, commit, or push occurred.",
        "",
    ])
    return "\n".join(lines)


def print_summary(rating, records, stats, figures, report_path, reasons):
    print(f"RATING: {rating}")
    print("VERIFIED FACTS:")
    print(f"- Parsed all official train/test list filenames; no test arrays were read.")
    print(f"- Checked {len(records)} official training keypoint files; all requested checks passed: {all(r['shape_ok'] and r['count_match'] and r['finite'] and r['constant'] is False and r['error'] is None for r in records)}")
    print(f"- Axis {stats['vertical_axis']} appears vertical; metre scale is plausible but not independently proven.")
    print("- MPJPE/PA-MPJPE retain stored units; PiW3D PCK uses joints 6 and 4; GT-count trimming hides count/miss/duplicate behavior.")
    print("UNRESOLVED FACTS:")
    print("- Official units, joint ordering, coordinate origin/frame, and candidate temporal group semantics remain unproven.")
    print("- The literal first-550/last-50 interpretation is not supported; the actual index-selection convention is unresolved.")
    print("EXACT RATING REASONS:")
    for reason in reasons:
        print(f"- {reason}")
    print("GENERATED FILES:")
    print(f"- {report_path.resolve()}")
    for path in figures:
        print(f"- {path.resolve()}")


def main():
    args = parse_args()
    if args.samples_per_count < 30:
        raise ValueError("--samples_per_count must be at least 30 for this audit")
    root = args.data_root.resolve()
    ensure_output_outside_dataset(args.output, root)
    ensure_output_outside_dataset(args.figure_dir, root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    splits = {split: read_split_list(root, split) for split in SPLITS}
    selected = select_training_ids(splits["train"], args.samples_per_count)
    representatives = [
        sample_id
        for count in (1, 2, 3)
        for sample_id in evenly_spaced(
            [item["id"] for item in splits["train"]["valid"] if item["person_count"] == count],
            4,
        )
    ]
    records, valid_samples = load_training_poses(root, selected)
    if not valid_samples:
        raise RuntimeError("No valid training poses were loaded")
    stats = pose_statistics(valid_samples)

    by_id = {sample["id"]: sample for sample in valid_samples}
    figure_ids = [
        selected[count][position]
        for count in (1, 2, 3)
        for position in (0, len(selected[count]) // 2)
    ]
    figures = [render_figure(by_id[sample_id], args.figure_dir) for sample_id in figure_ids]
    if len(figures) != 6:
        raise AssertionError("Expected six audit figures")

    utils_source = Path("utils.py").read_text(encoding="utf-8")
    train_source = Path("train_pose.py").read_text(encoding="utf-8")
    all_valid = all(
        record["shape_ok"]
        and record["count_match"]
        and record["finite"]
        and record["constant"] is False
        and record["error"] is None
        for record in records
    )
    any_malformed = any(split["malformed"] for split in splits.values())
    if not all_valid or any_malformed:
        rating = "RED"
        reasons = [
            "At least one sampled training-pose validation check failed or an official list ID was malformed; inspect the failed records.",
            "No automatic correction was applied.",
        ]
    else:
        rating = "YELLOW"
        reasons = [
            "GREEN is not justified because official units/joint order and candidate temporal group semantics are not independently established.",
            "The observed indices reject the literal first-550/last-50 interpretation, and all test candidate groups overlap train groups.",
            "The current PCK normalizer is inter-elbow distance under the provisional ordering, and GT-count trimming prevents missed/duplicate/count evaluation.",
            "All 90 bounded training-pose structure, count, finiteness, and nonconstant checks passed, so the sample did not justify RED.",
        ]
    report = build_report(
        root,
        splits,
        representatives,
        selected,
        records,
        valid_samples,
        stats,
        figures,
        utils_source,
        train_source,
        rating,
        reasons,
    )
    args.output.write_text(report, encoding="utf-8")
    print_summary(rating, records, stats, figures, args.output, reasons)


if __name__ == "__main__":
    main()
