"""
VKITTI2 Relational Annotations Generation
==========================================
SAM2 + YOLO-World + GT-Depth 기반 관계 추출

Usage:
    python src/relation_generation/generate_vkitti2_relations.py \
        --config configs/config_vkitti2_relational_gen.yaml

Output:
    data/vkitti2_relational/
    ├── Scene01_clone/
    │   ├── rgb_00000_masks.npy
    │   ├── rgb_00000_relations.json
    │   └── ...
    ├── Scene02_clone/
    └── ...
"""

import os
import json
import glob
import argparse

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm


# VKITTI2용 클래스 목록
VKITTI2_CLASSES = [
    # 차량
    "car",
    "van",
    "truck",
    "bus",
    "trailer",
    "caravan",
    # 사람/이동체
    "pedestrian",
    "cyclist",
    "motorcyclist",
    "bicycle",
    "motorcycle",
    # 도로 인프라
    "traffic sign",
    "traffic light",
    "pole",
    "street light",
    "fence",
    "guardrail",
    "billboard",
    # 자연물
    "tree",
    "vegetation",
    "terrain",
    # 기타
    "building",
    "road",
    "sidewalk",
    "parking",
]


def load_models(sam_ckpt, yolo_ckpt, device: str = "cuda"):
    """SAM2 + YOLO-World 로드"""
    try:
        from ultralytics import SAM, YOLOWorld
    except ImportError as e:
        raise ImportError("ultralytics 패키지가 필요합니다: pip install ultralytics") from e
    
    print(f"[SAM2] Loading from {sam_ckpt}...")
    sam_model = SAM(sam_ckpt)
    sam_model.to(device)
    
    print(f"[YOLO-World] Loading from {yolo_ckpt}...")
    yolo_model = YOLOWorld(yolo_ckpt)
    yolo_model.to(device)
    yolo_model.set_classes(VKITTI2_CLASSES)
    
    return sam_model, yolo_model


def detect_and_segment(image_np, sam_model, yolo_model, conf_thresh: float = 0.5, min_area: int = 500, max_objects: int = 20):
    """YOLO-World 검출 + SAM2 세그멘테이션"""
    H, W = image_np.shape[:2]
    
    # YOLO 검출
    results = yolo_model(image_np, verbose=False)
    
    if not results or results[0].boxes is None:
        return [], []
    
    boxes = results[0].boxes
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    labels = boxes.cls.cpu().numpy().astype(int)
    
    # Confidence 필터링
    valid_idx = scores >= conf_thresh
    boxes_xyxy = boxes_xyxy[valid_idx]
    scores = scores[valid_idx]
    labels = labels[valid_idx]
    
    if len(boxes_xyxy) == 0:
        return [], []
    
    # SAM2 세그멘테이션
    bboxes = boxes_xyxy.tolist()
    sam_results = sam_model(image_np, bboxes=bboxes, verbose=False)
    
    masks = []
    box_infos = []
    
    if sam_results and sam_results[0].masks is not None:
        mask_data = sam_results[0].masks.data.cpu().numpy()
        
        for i in range(min(len(bboxes), max_objects)):
            if i >= len(mask_data):
                break
            
            mask = mask_data[i].astype(bool)
            area = mask.sum()
            
            if area < min_area:
                continue
            
            # Centroid 계산
            ys, xs = np.where(mask)
            if len(ys) == 0:
                continue
            centroid = (float(np.mean(xs)), float(np.mean(ys)))
            
            masks.append(mask)
            box_infos.append(
                {
                    "id": len(box_infos),
                    "bbox": bboxes[i],
                    "class_id": int(labels[i]),
                    "class_name": VKITTI2_CLASSES[int(labels[i])] if int(labels[i]) < len(VKITTI2_CLASSES) else "unknown",
                    "confidence": float(scores[i]),
                    "area": int(area),
                    "centroid": centroid,
                }
            )
    
    return masks, box_infos


def compute_depth_stats(mask, depth, min_depth: float = 0.1, max_depth: float = 80.0):
    """마스크 영역의 깊이 통계량 계산"""
    valid = mask & (depth > min_depth) & (depth < max_depth)
    valid_depths = depth[valid]
    
    if len(valid_depths) < 100:  # 최소 픽셀 수
        return None
    
    return {
        "mean": float(np.mean(valid_depths)),
        "std": float(np.std(valid_depths)),
        "max": float(np.max(valid_depths)),
        "min": float(np.min(valid_depths)),
        "median": float(np.median(valid_depths)),
        "count": int(len(valid_depths)),
    }


def generate_relations_statistical(
    box_infos,
    depth_stats,
    image_shape,
    max_rel_per_object: int = 5,
    max_centroid_dist: float = 0.5,
):
    """
    Statistical 방식 관계 생성 (논문 방식)
    
    조건: |μ_i - μ_j| > (M_i - μ_i) + (M_j - μ_j)
    대표 깊이: r_k = μ_k + σ_k
    """
    H, W = image_shape[:2]
    relations = []
    relation_count = {info["id"]: 0 for info in box_infos}
    
    for i, info_i in enumerate(box_infos):
        stats_i = depth_stats.get(info_i["id"])
        if stats_i is None:
            continue
        
        for j, info_j in enumerate(box_infos):
            if i >= j:
                continue
            
            stats_j = depth_stats.get(info_j["id"])
            if stats_j is None:
                continue
            
            # Relation 수 제한 체크
            if (relation_count[info_i["id"]] >= max_rel_per_object or relation_count[info_j["id"]] >= max_rel_per_object):
                continue
            
            # 중심 거리 조건
            cx_i, cy_i = info_i["centroid"]
            cx_j, cy_j = info_j["centroid"]
            norm_dist = np.sqrt(((cx_i - cx_j) / W) ** 2 + ((cy_i - cy_j) / H) ** 2)
            
            if norm_dist > max_centroid_dist:
                continue
            
            # Statistical 조건: |μ_i - μ_j| > (M_i - μ_i) + (M_j - μ_j)
            depth_diff = abs(stats_i["mean"] - stats_j["mean"])
            spread_i = stats_i["max"] - stats_i["mean"]
            spread_j = stats_j["max"] - stats_j["mean"]
            
            if depth_diff <= spread_i + spread_j:
                continue
            
            # 대표 깊이로 순서 결정: r_k = μ_k + σ_k
            r_i = stats_i["mean"] + stats_i["std"]
            r_j = stats_j["mean"] + stats_j["std"]
            
            if r_i < r_j:
                front_id, back_id = info_i["id"], info_j["id"]
                front_info, back_info = info_i, info_j
            else:
                front_id, back_id = info_j["id"], info_i["id"]
                front_info, back_info = info_j, info_i
            
            # Confidence 계산
            margin = depth_diff - (spread_i + spread_j)
            confidence = min(1.0, margin / 5.0)  # 5m 이상이면 confidence=1
            
            relations.append(
                {
                    "subject_idx": front_id,
                    "object_idx": back_id,
                    "subject_class": front_info["class_name"],
                    "object_class": back_info["class_name"],
                    "relation": "in_front_of",
                    "confidence": confidence,
                    "depth_diff": float(abs(r_i - r_j)),
                }
            )
            
            relation_count[info_i["id"]] += 1
            relation_count[info_j["id"]] += 1
    
    return relations


def process_scene(vkitti2_root, scene, condition, out_dir, sam_model, yolo_model, args):
    """단일 scene 처리"""
    rgb_dir = os.path.join(vkitti2_root, scene, condition, "frames", "rgb", "Camera_0")
    depth_dir = os.path.join(vkitti2_root, scene, condition, "frames", "depth", "Camera_0")
    
    if not os.path.exists(rgb_dir):
        print(f"[SKIP] RGB dir not found: {rgb_dir}")
        return 0
    
    scene_out_dir = os.path.join(out_dir, f"{scene}_{condition}")
    os.makedirs(scene_out_dir, exist_ok=True)
    
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "rgb_*.jpg")))
    
    total_relations = 0
    
    for rgb_path in tqdm(rgb_files, desc=f"{scene}/{condition}"):
        fname = os.path.basename(rgb_path)
        frame_id = fname.replace("rgb_", "").replace(".jpg", "")
        depth_path = os.path.join(depth_dir, f"depth_{frame_id}.png")
        
        if not os.path.exists(depth_path):
            continue
        
        # Load image
        image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load depth (VKITTI2: cm in uint16)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32) / 100.0  # cm -> m
        
        # Detect and segment
        masks, box_infos = detect_and_segment(
            image_rgb,
            sam_model,
            yolo_model,
            conf_thresh=getattr(args, "yolo_conf_thresh", 0.5),
            min_area=getattr(args, "sam_min_region_area", 500),
            max_objects=getattr(args, "sam_max_objects", 20),
        )
        
        if len(masks) < 2:
            continue
        
        # Compute depth stats
        depth_stats = {}
        for info in box_infos:
            idx = info["id"]
            if idx < len(masks):
                stats = compute_depth_stats(
                    masks[idx],
                    depth,
                    min_depth=getattr(args, "min_depth", 0.1),
                    max_depth=getattr(args, "max_depth", 80.0),
                )
                if stats:
                    depth_stats[idx] = stats
        
        # Generate relations (statistical 방식)
        relations = generate_relations_statistical(
            box_infos,
            depth_stats,
            image.shape,
            max_rel_per_object=getattr(args, "max_rel_per_object", 5),
            max_centroid_dist=getattr(args, "max_centroid_dist", 0.5),
        )

        # Fallback: 통계 조건으로 아무 관계도 생성되지 않은 경우,
        # 보수적인 depth 차이 + 공간 인접 조건을 만족하는 객체 쌍만 소량 추가.
        if not relations and len(box_infos) >= 2 and depth_stats:
            fb_min_dd = getattr(args, "fallback_min_depth_diff", 2.0)
            fb_max_dist = getattr(
                args,
                "fallback_max_centroid_dist",
                getattr(args, "max_centroid_dist", 0.6),
            )
            fb_max_per_frame = getattr(args, "fallback_max_relations_per_frame", 2)

            H, W = image.shape[:2]
            candidates = []
            for i, info_i in enumerate(box_infos):
                stats_i = depth_stats.get(info_i["id"])
                if stats_i is None:
                    continue
                for j, info_j in enumerate(box_infos):
                    if i >= j:
                        continue
                    stats_j = depth_stats.get(info_j["id"])
                    if stats_j is None:
                        continue

                    # mean depth 기반 보수적 조건
                    mu_i = stats_i["mean"]
                    mu_j = stats_j["mean"]
                    depth_diff = abs(mu_i - mu_j)
                    if depth_diff < fb_min_dd:
                        continue

                    # 중심 거리 조건 (정규화 거리)
                    cx_i, cy_i = info_i["centroid"]
                    cx_j, cy_j = info_j["centroid"]
                    norm_dist = np.sqrt(((cx_i - cx_j) / W) ** 2 + ((cy_i - cy_j) / H) ** 2)
                    if norm_dist > fb_max_dist:
                        continue

                    candidates.append((depth_diff, info_i, info_j, mu_i, mu_j))

            # 깊이 차이가 작은 것부터 소수만 선택
            candidates.sort(key=lambda x: x[0])
            for k, (depth_diff, info_i, info_j, mu_i, mu_j) in enumerate(candidates[:fb_max_per_frame]):
                if mu_i < mu_j:
                    front_info, back_info = info_i, info_j
                else:
                    front_info, back_info = info_j, info_i

                relations.append(
                    {
                        "subject_idx": front_info["id"],
                        "object_idx": back_info["id"],
                        "subject_class": front_info["class_name"],
                        "object_class": back_info["class_name"],
                        "relation": "in_front_of",
                        "confidence": min(0.9, 0.6 + depth_diff / 10.0),
                        "depth_diff": float(depth_diff),
                        "fallback": True,
                    }
                )
        
        total_relations += len(relations)
        
        # Save masks
        masks_np = np.stack(masks, axis=0).astype(np.uint8) if masks else np.zeros((0, *depth.shape), dtype=np.uint8)
        out_mask_path = os.path.join(scene_out_dir, f"rgb_{frame_id}_masks.npy")
        np.save(out_mask_path, masks_np)
        
        # Save relations
        out_rel_path = os.path.join(scene_out_dir, f"rgb_{frame_id}_relations.json")
        with open(out_rel_path, "w") as f:
            json.dump(relations, f, indent=2)
    
    print(f"  → {len(rgb_files)} frames, {total_relations} relations")
    return total_relations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    for key, value in config.items():
        setattr(args, key, value)
    
    # Paths
    vkitti2_root = args.vkitti2_root
    out_dir = args.out_base_dir
    
    # Scenes and conditions
    scenes = args.scenes or ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
    conditions = args.conditions or ["clone"]
    
    print(f"VKITTI2 Root: {vkitti2_root}")
    print(f"Output: {out_dir}")
    print(f"Scenes: {scenes}")
    print(f"Conditions: {conditions}")
    
    # Load models
    sam_model, yolo_model = load_models(
        args.sam_ckpt,
        args.yolo_ckpt,
        device=args.device,
    )
    
    # Process all scenes
    total_relations = 0
    for scene in scenes:
        for condition in conditions:
            n_rel = process_scene(
                vkitti2_root,
                scene,
                condition,
                out_dir,
                sam_model,
                yolo_model,
                args,
            )
            total_relations += n_rel
    
    print(f"\n✅ Done! Total relations: {total_relations}")
    print(f"   Output saved to: {out_dir}")


if __name__ == "__main__":
    main()

