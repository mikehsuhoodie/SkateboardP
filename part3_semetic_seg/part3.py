from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import numpy as np
import cv2
from skimage.morphology import skeletonize
import networkx as nx
import json
from datetime import datetime
from collections import defaultdict

# Model selection
# Use ADE20K by default for broader categories (indoor + outdoor).
# You can override with: export SEGFORMER_MODEL_ID="..."
DEFAULT_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
MODEL_ID = os.getenv("SEGFORMER_MODEL_ID", DEFAULT_MODEL_ID)

# Input / output
INPUT_DIR = "images"
OUTPUT_DIR = "results"
DEFAULT_IMAGE_NAME = "street.jpg"
IMAGE_NAME = os.getenv("INPUT_IMAGE", DEFAULT_IMAGE_NAME)

# Morphology kernel tuning
MORPH_KERNEL_RATIO = 0.06  # relative to min(image_width, image_height)
MORPH_KERNEL_MIN = 9
MORPH_KERNEL_MAX = 71

# Playability-oriented path extraction (safe incremental pass)
PLAYABLE_SCAN_STEP_RATIO = 0.01
PLAYABLE_SCAN_STEP_MIN = 2
PLAYABLE_SCAN_STEP_MAX = 8
PLAYABLE_CLUSTER_GAP_RATIO = 0.015
PLAYABLE_CLUSTER_GAP_MIN = 4
PLAYABLE_CLUSTER_GAP_MAX = 24
PLAYABLE_MAX_VERTICAL_STEP_RATIO = 0.08
PLAYABLE_MAX_VERTICAL_STEP_MIN = 10
PLAYABLE_MAX_VERTICAL_STEP_MAX = 80
PLAYABLE_MAX_GAP_BUCKETS = 2
PLAYABLE_MIN_RUN_POINTS = 12
PLAYABLE_SMOOTH_WINDOW = 5

# RDP simplification tuning
RDP_EPSILON_RATIO = 0.005  # relative to min(image_width, image_height)
RDP_EPSILON_MIN = 1.0
RDP_EPSILON_MAX = 8.0

# Multi-class track selection
DETECT_MIN_AREA_RATIO = 0.001  # keep very small classes in detected_classes debug
MIN_CLASS_AREA_RATIO = 0.01
MAX_CANDIDATE_CLASSES = 12
MAX_OUTPUT_TRACKS = 3
MIN_TRACK_POINTS = 6
MIN_TRACK_X_SPAN_RATIO = 0.12


def _to_odd_clamped(value, min_value, max_value):
    value = max(min_value, min(int(round(value)), max_value))
    if value % 2 == 0:
        value = value + 1 if value < max_value else value - 1
    return max(3, value)


def _to_int_clamped(value, min_value, max_value):
    return max(min_value, min(int(round(value)), max_value))


def get_morph_kernel_size(image_size):
    width, height = image_size
    min_dim = min(width, height)
    return _to_odd_clamped(min_dim * MORPH_KERNEL_RATIO, MORPH_KERNEL_MIN, MORPH_KERNEL_MAX)


def get_rdp_epsilon(image_size):
    width, height = image_size
    min_dim = min(width, height)
    epsilon = float(min_dim) * RDP_EPSILON_RATIO
    return float(np.clip(epsilon, RDP_EPSILON_MIN, RDP_EPSILON_MAX))


def sanitize_label(label):
    lowered = str(label).strip().lower()
    safe = []
    for ch in lowered:
        if ch.isalnum():
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe).strip("_") or "unknown"


def _cluster_1d(values, gap_threshold):
    if not values:
        return []

    clusters = []
    current = [int(values[0])]
    for v in values[1:]:
        v = int(v)
        if abs(v - current[-1]) <= gap_threshold:
            current.append(v)
        else:
            clusters.append(current)
            current = [v]
    clusters.append(current)

    reps = []
    for cluster in clusters:
        reps.append((int(np.median(cluster)), len(cluster)))
    return reps


def _smooth_rows(path_points, window_size, image_height):
    if len(path_points) < 3 or window_size <= 1:
        return path_points
    if window_size % 2 == 0:
        window_size += 1

    rows = np.array([r for r, _ in path_points], dtype=np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    padded = np.pad(rows, (window_size // 2, window_size // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")

    result = []
    for (_, c), new_r in zip(path_points, smoothed):
        r = int(np.clip(round(float(new_r)), 0, image_height - 1))
        result.append((r, c))
    return result


def _dedupe_nonincreasing_x(path_points):
    if not path_points:
        return []

    deduped = [path_points[0]]
    for r, c in path_points[1:]:
        prev_r, prev_c = deduped[-1]
        if c > prev_c:
            deduped.append((r, c))
        elif c == prev_c:
            deduped[-1] = (int(round((prev_r + r) / 2.0)), prev_c)
    return deduped


def get_path_span_stats(path_points):
    if len(path_points) < 2:
        return {"x_span": 0, "y_span": 0, "ratio": 0.0}

    xs = [c for _, c in path_points]
    ys = [r for r, _ in path_points]
    x_span = max(xs) - min(xs)
    y_span = max(ys) - min(ys)
    ratio = float(x_span) / max(1.0, float(y_span))
    return {"x_span": int(x_span), "y_span": int(y_span), "ratio": round(ratio, 4)}


def get_longest_path(skeleton_img):
    points = np.argwhere(skeleton_img > 0)
    if len(points) == 0:
        return []

    graph = nx.Graph()
    point_set = set(map(tuple, points))

    for r, c in points:
        graph.add_node((r, c))
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                neighbor = (r + dr, c + dc)
                if neighbor in point_set:
                    graph.add_edge((r, c), neighbor)

    if len(graph) == 0:
        return []

    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    largest_cc = graph.subgraph(components[0])
    nodes = list(largest_cc.nodes())
    if not nodes:
        return []

    u = nodes[0]
    lengths = nx.single_source_shortest_path_length(largest_cc, u)
    v = max(lengths, key=lengths.get)

    lengths_v = nx.single_source_shortest_path_length(largest_cc, v)
    w = max(lengths_v, key=lengths_v.get)

    path = nx.shortest_path(largest_cc, v, w)
    return path


def build_playable_scanline_path(skeleton_img, image_size):
    width, height = image_size
    points = np.argwhere(skeleton_img > 0)
    if len(points) == 0:
        return []

    min_dim = min(width, height)
    scan_step = _to_int_clamped(
        min_dim * PLAYABLE_SCAN_STEP_RATIO,
        PLAYABLE_SCAN_STEP_MIN,
        PLAYABLE_SCAN_STEP_MAX,
    )
    cluster_gap = _to_int_clamped(
        height * PLAYABLE_CLUSTER_GAP_RATIO,
        PLAYABLE_CLUSTER_GAP_MIN,
        PLAYABLE_CLUSTER_GAP_MAX,
    )
    max_vertical_step = _to_int_clamped(
        height * PLAYABLE_MAX_VERTICAL_STEP_RATIO,
        PLAYABLE_MAX_VERTICAL_STEP_MIN,
        PLAYABLE_MAX_VERTICAL_STEP_MAX,
    )

    bucket_rows = defaultdict(list)
    for r, c in points:
        bucket_rows[int(c) // scan_step].append(int(r))

    bucket_indices = sorted(bucket_rows.keys())
    if not bucket_indices:
        return []

    min_run_points = max(PLAYABLE_MIN_RUN_POINTS, max(8, width // max(1, scan_step * 10)))
    runs = []
    current_run = []
    last_bucket = None

    def finalize_current_run():
        nonlocal current_run, last_bucket
        if len(current_run) >= min_run_points:
            runs.append(current_run)
        current_run = []
        last_bucket = None

    for bucket_idx in bucket_indices:
        y_candidates = _cluster_1d(sorted(bucket_rows[bucket_idx]), cluster_gap)
        if not y_candidates:
            continue

        if not current_run:
            chosen_y = max(y_candidates, key=lambda item: (item[1], item[0]))[0]
            candidate_x = min(width - 1, max(0, int(round((bucket_idx + 0.5) * scan_step))))
            current_run.append((chosen_y, candidate_x))
            last_bucket = bucket_idx
            continue

        gap_buckets = bucket_idx - last_bucket - 1
        if gap_buckets > PLAYABLE_MAX_GAP_BUCKETS:
            finalize_current_run()
            chosen_y = max(y_candidates, key=lambda item: (item[1], item[0]))[0]
            candidate_x = min(width - 1, max(0, int(round((bucket_idx + 0.5) * scan_step))))
            current_run.append((chosen_y, candidate_x))
            last_bucket = bucket_idx
            continue

        prev_y, prev_x = current_run[-1]
        chosen_y, _ = min(
            y_candidates,
            key=lambda item: (abs(item[0] - prev_y), -item[1], item[0]),
        )

        if abs(chosen_y - prev_y) > max_vertical_step:
            chosen_y = int(np.clip(chosen_y, prev_y - max_vertical_step, prev_y + max_vertical_step))

        candidate_x = min(width - 1, max(0, int(round((bucket_idx + 0.5) * scan_step))))

        if gap_buckets > 0:
            for i in range(1, gap_buckets + 1):
                t = i / float(gap_buckets + 1)
                interp_bucket = last_bucket + i
                interp_x = min(width - 1, max(0, int(round((interp_bucket + 0.5) * scan_step))))
                interp_y = int(round(prev_y + (chosen_y - prev_y) * t))
                interp_y = int(np.clip(interp_y, 0, height - 1))
                if interp_x > prev_x:
                    current_run.append((interp_y, interp_x))
                    prev_y, prev_x = current_run[-1]

        if candidate_x > current_run[-1][1]:
            current_run.append((chosen_y, candidate_x))
        elif candidate_x == current_run[-1][1]:
            avg_y = int(round((current_run[-1][0] + chosen_y) / 2.0))
            current_run[-1] = (avg_y, candidate_x)

        last_bucket = bucket_idx

    finalize_current_run()

    if not runs:
        return []

    best_run = max(runs, key=lambda run: ((run[-1][1] - run[0][1]), len(run)))
    best_run = _smooth_rows(best_run, PLAYABLE_SMOOTH_WINDOW, height)
    best_run = _dedupe_nonincreasing_x(best_run)
    return best_run


def rdp_simplify(points, epsilon):
    if len(points) < 3:
        return points

    dmax = 0.0
    index = 0
    end = len(points) - 1

    a = np.array(points[0], dtype=np.float32)
    b = np.array(points[end], dtype=np.float32)
    ab = b - a
    norm_ab = np.linalg.norm(ab)

    for i in range(1, end):
        c = np.array(points[i], dtype=np.float32)
        if norm_ab == 0:
            d = np.linalg.norm(c - a)
        else:
            d = np.abs(ab[0] * (c[1] - a[1]) - ab[1] * (c[0] - a[0])) / norm_ab

        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        left = rdp_simplify(points[: index + 1], epsilon)
        right = rdp_simplify(points[index:], epsilon)
        return left[:-1] + right

    return [points[0], points[end]]


def build_path_image(path_points, image_size):
    width, height = image_size
    path_img = np.zeros((height, width), dtype=np.uint8)
    for r, c in path_points:
        if 0 <= r < height and 0 <= c < width:
            path_img[r, c] = 255
    kernel_path = np.ones((3, 3), np.uint8)
    return cv2.dilate(path_img, kernel_path, iterations=1)


def extract_track_from_mask(mask_bool, image_size):
    if np.count_nonzero(mask_bool) == 0:
        return None

    mask_uint8 = (mask_bool.astype(np.uint8) * 255)

    morph_kernel_size = get_morph_kernel_size(image_size)
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    opening = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    skeleton = skeletonize(closing > 0)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    longest_path = get_longest_path(skeleton)
    playable_path = build_playable_scanline_path(skeleton, image_size)

    path_source = "playable_scanline" if len(playable_path) >= 2 else "longest_path"
    path_for_simplify = playable_path if len(playable_path) >= 2 else longest_path

    if len(path_for_simplify) < 2:
        return None

    rdp_epsilon = get_rdp_epsilon(image_size)
    simplified_path = rdp_simplify(path_for_simplify, epsilon=rdp_epsilon)
    simplified_path = _dedupe_nonincreasing_x(simplified_path)

    if len(simplified_path) < 2:
        return None

    stats = get_path_span_stats(simplified_path)

    return {
        "mask_uint8": mask_uint8,
        "closing": closing,
        "skeleton_uint8": skeleton_uint8,
        "path_pixels": simplified_path,
        "path_source": path_source,
        "morph_kernel_size": morph_kernel_size,
        "rdp_epsilon": rdp_epsilon,
        "stats": stats,
    }


def compute_track_quality(track_stats, point_count, area_ratio, mean_confidence, image_size):
    width, height = image_size
    x_span_ratio = float(track_stats["x_span"]) / max(1.0, float(width))
    orientation_ratio = float(track_stats["x_span"]) / max(1.0, float(track_stats["y_span"]))

    orientation_score = min(1.0, orientation_ratio / 1.2)
    point_score = min(1.0, float(point_count) / 40.0)
    area_score = min(1.0, float(area_ratio) / 0.25)
    conf_score = float(np.clip(mean_confidence, 0.0, 1.0))

    quality = (
        0.35 * x_span_ratio
        + 0.20 * orientation_score
        + 0.20 * point_score
        + 0.15 * area_score
        + 0.10 * conf_score
    )

    return round(float(quality), 4), {
        "x_span_ratio": round(x_span_ratio, 4),
        "orientation_ratio": round(orientation_ratio, 4),
        "point_score": round(point_score, 4),
        "area_score": round(area_score, 4),
        "conf_score": round(conf_score, 4),
    }


def detect_present_classes(pred_map, prob_map, id2label):
    unique_ids, counts = np.unique(pred_map, return_counts=True)
    total_pixels = pred_map.size

    detected = []
    for class_id, count in zip(unique_ids, counts):
        area_ratio = float(count) / float(total_pixels)
        if area_ratio < DETECT_MIN_AREA_RATIO:
            continue

        class_mask = pred_map == int(class_id)
        mean_conf = float(prob_map[int(class_id)][class_mask].mean()) if class_mask.any() else 0.0
        label = str(id2label.get(int(class_id), f"class_{class_id}"))

        detected.append(
            {
                "class_id": int(class_id),
                "semantic_label": label,
                "area_ratio": round(area_ratio, 4),
                "mean_confidence": round(mean_conf, 4),
            }
        )

    detected.sort(key=lambda x: (x["area_ratio"], x["mean_confidence"]), reverse=True)
    return detected


def main():
    # Read access token (file first, then env var)
    token_file = "hf_token.txt"
    if os.path.exists(token_file):
        with open(token_file, "r", encoding="utf-8") as f:
            hf_token = f.read().strip()
    else:
        hf_token = os.getenv("HF_TOKEN")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processor = SegformerImageProcessor.from_pretrained(MODEL_ID, token=hf_token)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, token=hf_token)

    print(f"Using model: {MODEL_ID}")
    print(f"Loaded labels: {len(model.config.id2label)} classes")

    image_path = os.path.join(INPUT_DIR, IMAGE_NAME)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    print(f"Input image: {IMAGE_NAME}, size={width}x{height}")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    prob_map = torch.softmax(upsampled_logits, dim=1)[0].detach().cpu().numpy()
    pred_map = upsampled_logits.argmax(1)[0].detach().cpu().numpy()

    detected_classes = detect_present_classes(pred_map, prob_map, model.config.id2label)
    print("Detected classes (top 10 by area):")
    for info in detected_classes[:10]:
        print(
            f"  id={info['class_id']:>3} label={info['semantic_label']:<24} "
            f"area={info['area_ratio']:.4f} conf={info['mean_confidence']:.4f}"
        )

    candidate_classes = [
        info for info in detected_classes if info["area_ratio"] >= MIN_CLASS_AREA_RATIO
    ][:MAX_CANDIDATE_CLASSES]

    if not candidate_classes and detected_classes:
        candidate_classes = detected_classes[: min(len(detected_classes), MAX_CANDIDATE_CLASSES)]

    if not candidate_classes:
        raise RuntimeError("No semantic classes detected from this image.")

    print(f"Processing {len(candidate_classes)} candidate classes...")

    candidate_tracks = []
    for info in candidate_classes:
        class_id = info["class_id"]
        label = info["semantic_label"]
        safe_label = sanitize_label(label)

        class_mask = pred_map == class_id
        extracted = extract_track_from_mask(class_mask, image.size)
        if extracted is None:
            print(f"  skip class={label} (id={class_id}): no valid path")
            continue

        path_pixels = extracted["path_pixels"]
        quality_score, quality_breakdown = compute_track_quality(
            extracted["stats"],
            len(path_pixels),
            info["area_ratio"],
            info["mean_confidence"],
            image.size,
        )

        x_span_ratio = float(extracted["stats"]["x_span"]) / max(1.0, float(width))
        is_playable = len(path_pixels) >= MIN_TRACK_POINTS and x_span_ratio >= MIN_TRACK_X_SPAN_RATIO

        prefix = f"class_{class_id:03d}_{safe_label}"
        Image.fromarray(extracted["mask_uint8"]).save(
            os.path.join(OUTPUT_DIR, f"{prefix}_mask.png")
        )
        Image.fromarray(extracted["closing"]).save(
            os.path.join(OUTPUT_DIR, f"{prefix}_mask_closing.png")
        )
        Image.fromarray(extracted["skeleton_uint8"]).save(
            os.path.join(OUTPUT_DIR, f"{prefix}_skeleton.png")
        )
        Image.fromarray(build_path_image(path_pixels, image.size)).save(
            os.path.join(OUTPUT_DIR, f"{prefix}_path.png")
        )

        candidate_tracks.append(
            {
                "class_id": class_id,
                "semantic_label": label,
                "area_ratio": info["area_ratio"],
                "mean_confidence": info["mean_confidence"],
                "quality_score": quality_score,
                "quality_breakdown": quality_breakdown,
                "is_playable": is_playable,
                "path_source": extracted["path_source"],
                "path_pixels": path_pixels,
                "stats": extracted["stats"],
                "morph_kernel_size": extracted["morph_kernel_size"],
                "rdp_epsilon": round(extracted["rdp_epsilon"], 4),
            }
        )

        print(
            f"  class={label:<24} quality={quality_score:.4f} "
            f"playable={is_playable} points={len(path_pixels)} x/y={extracted['stats']['ratio']}"
        )

    if not candidate_tracks:
        raise RuntimeError("No valid tracks could be extracted from detected classes.")

    playable_tracks = [track for track in candidate_tracks if track["is_playable"]]
    if playable_tracks:
        selected_tracks = sorted(
            playable_tracks,
            key=lambda item: item["quality_score"],
            reverse=True,
        )[:MAX_OUTPUT_TRACKS]
    else:
        selected_tracks = sorted(
            candidate_tracks,
            key=lambda item: item["quality_score"],
            reverse=True,
        )[:1]
        print("Warning: no track passed playable thresholds, fallback to best candidate only.")

    color_palette = [
        (0, 255, 100, 255),
        (255, 196, 0, 255),
        (0, 180, 255, 255),
    ]

    overlay_rgba = image.convert("RGBA")
    draw_overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(draw_overlay)

    combined_path_img = np.zeros((height, width), dtype=np.uint8)

    output_tracks = []
    for stage_index, track in enumerate(selected_tracks, start=1):
        path_pixels = track["path_pixels"]
        normalized_points = [
            [round(float(c) / width, 4), round(float(r) / height, 4)]
            for r, c in path_pixels
        ]

        output_tracks.append(
            {
                "stage_index": stage_index,
                "semantic_label": track["semantic_label"],
                "points": normalized_points,
            }
        )

        color = color_palette[(stage_index - 1) % len(color_palette)]
        if len(path_pixels) > 1:
            coords = [(c, r) for r, c in path_pixels]
            draw.line(coords, fill=color, width=5)

        for r, c in path_pixels:
            if 0 <= r < height and 0 <= c < width:
                combined_path_img[r, c] = 255

    combined_path_img = cv2.dilate(combined_path_img, np.ones((3, 3), np.uint8), iterations=1)
    Image.fromarray(combined_path_img).save(os.path.join(OUTPUT_DIR, "main_path.png"))

    final_overlay = Image.alpha_composite(overlay_rgba, draw_overlay)
    final_overlay.convert("RGB").save(os.path.join(OUTPUT_DIR, "result_final.jpg"))

    output_data = {
        "version": "1.1",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source_image": IMAGE_NAME,
        "aspect_ratio": round(width / height, 4),
        "points": output_tracks[0]["points"] if output_tracks else [],
        "occluded_segments": [],
        "terrain_type": output_tracks[0]["semantic_label"] if output_tracks else "unknown",
        "friction_coefficient": 0.8,
        "stage_count": len(output_tracks),
        "tracks": output_tracks,
    }

    output_path = os.path.join(OUTPUT_DIR, "output.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)

    print(f"Exported {len(output_tracks)} track(s) to: {output_path}")
    print(f"Saved preview: {os.path.join(OUTPUT_DIR, 'result_final.jpg')}")


if __name__ == "__main__":
    main()
