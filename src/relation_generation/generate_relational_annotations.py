# NYU Depth V2 40-class benchmark (Gupta et al.)
nyu_v2_standard_classes = [
    # 벽, 천장, 바닥, 벽에 붙은 것 제외
    "cabinet",           # 3
    "bed",               # 4
    "chair",             # 5
    "sofa",              # 6
    "table",             # 7
    "door",              # 8
    "window",            # 9
    "bookshelf",         # 10
    "picture",           # 11
    "counter",           # 12
    "blinds",            # 13
    "desk",              # 14
    "shelves",           # 15
    "curtain",           # 16
    "dresser",           # 17
    "pillow",            # 18
    "mirror",            # 19
    "floor mat",         # 20
    "clothes",           # 21
    "books",             # 23
    "refrigerator",      # 24
    "television",        # 25
    "paper",             # 26
    "towel",             # 27
    "shower curtain",    # 28
    "box",               # 29
    "whiteboard",        # 30
    "person",            # 31
    "night stand",       # 32
    "toilet",            # 33
    "sink",              # 34
    "lamp",              # 35
    "bathtub",           # 36
    "bag",               # 37
    "other structure",   # 38
    "other furniture",   # 39
    "other prop",        # 40

    # 실제 scene에서 자주 등장할 법한 물체 (벽/천장/바닥/벽부착형 제외)
    "microwave", "oven", "stove", "dishwasher", "coffee machine", "toaster",
    "washing machine", "dryer",
    "computer", "monitor", "keyboard", "mouse", "printer", "projector",
    "fan", "air conditioner", "heater",
    "bottle", "cup", "mug", "plate", "bowl", "utensil",
    "plant", "flower pot",
    "bench", "locker", "rack",
    "exercise equipment", "treadmill", "dumbbell",
    "bicycle",
    "backpack", "suitcase",
    "trash can", "recycle bin",
    "shoes", "slippers",
    "handrail", "stairs",
    "faucet", "soap dispenser",
    "food", "fruit", "vegetable",
    "notebook", "pen", "pencil",
    "remote control", "phone",
    "light switch", "outlet",
    "makeup table",
    "tissue box", "napkin",
    "coat rack", "umbrella",
    "shopping bag",
    "pet", "cat", "dog"
]
"""
SAM2 (ultralytics) + GT-Depth 기반 Relational Annotations 생성 (NYU-Depth-v2)

# Train 데이터셋 처리
python src/relation_generation/generate_relational_annotations.py --config configs/config_relational_gen.yaml --mode train

# Test 데이터셋 처리 (설정 파일에서 test 섹션 주석 해제 필요)
python src/relation_generation/generate_relational_annotations.py --config configs/config_relational_gen.yaml --mode test

# relation_style: median (기본) | statistical
#   median     : median depth + 0.3~3.0m + 공간 인접 (Relational-WorDepth)
#   statistical: μ, σ, M 기반 (Robustness 논문)
"""

import os
import json
import glob
import torch
import numpy as np
import cv2



# =========================
# 1) YOLO + SAM2: 객체 탐지 후 마스크 생성 (ultralytics)
# =========================
try:
    from ultralytics import YOLOWorld, SAM
    SAM2_AVAILABLE = True
    YW_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    YW_AVAILABLE = False
    raise ImportError(
        "ultralytics 패키지가 필요합니다.\n"
        "다음을 실행하세요: pip install ultralytics"
    )





def load_sam_and_yolo(
    sam_ckpt_path: str = "sam2_b.pt",
    yolo_world_ckpt_path: str = "yolov8s-worldv2.pt",
    device: str = "cuda",
):
    """
    yolo-world + SAM2 로더
    """
    if not SAM2_AVAILABLE or not YW_AVAILABLE:
        raise ImportError(
            "yolo-world 및 ultralytics 패키지가 필요합니다.\n"
            "다음을 실행하세요: pip install yoloworld ultralytics"
        )

    # config/CLI에서 전달된 device를 기본 CUDA 디바이스로도 맞춰 줌
    if isinstance(device, str) and device.startswith("cuda:") and torch.cuda.is_available():
        try:
            device_index = int(device.split(":")[1])
            torch.cuda.set_device(device_index)
        except Exception as exc:
            print(f"[WARN] Failed to set CUDA device to {device}: {exc}")

    print(f"[SAM2] Loading ultralytics SAM2 model from {sam_ckpt_path}...")
    sam2_model = SAM(sam_ckpt_path)
    if device and hasattr(sam2_model, "to"):
        sam2_model.to(device)

    print(f"[YOLO-World] Loading YOLO-World model from {yolo_world_ckpt_path}...")
    yolo_model = YOLOWorld(yolo_world_ckpt_path)
    # 필요시 device로 이동
    if device and hasattr(yolo_model, "to"):
        yolo_model.to(device)
    # NYU 40-class로 클래스 세팅
    yolo_model.set_classes(nyu_v2_standard_classes)
    return sam2_model, yolo_model



# 객체 탐지 클래스 필터 (원하는 클래스만 사용)
ALLOWED_CLASSES = set(range(40))

def generate_object_masks_with_yolo_sam(image, sam2_model, yolo_model, min_mask_region_area=1000, allowed_classes=ALLOWED_CLASSES, yolo_conf_thresh=None):
    """
    YOLO-World로 객체 bbox 탐지 후, 각 bbox를 SAM2에 prompt로 넣어 마스크 생성
    OpenCV 기반 벡터화 및 numpy 연산 최적화
    yolo_conf_thresh: YOLO confidence threshold
    """
    if isinstance(image, str):
        image_np = cv2.imread(image, cv2.IMREAD_COLOR)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    elif isinstance(image, np.ndarray):
        image_np = image
    else:
        image_np = np.array(image)

    H, W = image_np.shape[:2]
    if yolo_conf_thresh is None:
        yolo_conf_thresh = 0.5
    # ultralytics YOLOWorld inference (Results 객체)
    results = yolo_model(image_np)
    if not results or not hasattr(results[0], 'boxes') or results[0].boxes is None:
        print("[YOLO-World+SAM2] No detections (empty results or boxes).")
        return [], []
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # (N, 4)
    scores = results[0].boxes.conf.cpu().numpy()      # (N,)
    labels = results[0].boxes.cls.cpu().numpy()       # (N,)

    # confidence/class filtering 제거: 모든 bbox 사용
    det_indices = list(range(len(labels)))
    if len(det_indices) == 0:
        print("[YOLO-World+SAM2] No detections from YOLO.")
        return [], []
    bboxes = [boxes_xyxy[i].tolist() for i in det_indices]
    sam_masks = []
    boxes = []
    idx = 0
    try:
        sam_result = sam2_model(image_np, bboxes=bboxes, verbose=False)
        masks = None
        if sam_result and hasattr(sam_result[0], 'masks') and sam_result[0].masks is not None:
            masks = sam_result[0].masks.data.cpu().numpy()  # (N, H, W)
        for i in range(len(bboxes)):
            bbox = bboxes[i]
            det_info_idx = det_indices[i]
            cls_name = labels[det_info_idx]
            conf_val = float(scores[det_info_idx])
            # SAM2 마스크가 있으면 사용, 없으면 bbox 사각형 마스크 생성
            if masks is not None and i < len(masks):
                mask_bool = masks[i].astype(bool)
            else:
                mask_bool = np.zeros((H, W), dtype=bool)
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                mask_bool[y1:y2, x1:x2] = True
            area = int(mask_bool.sum())
            coords = np.where(mask_bool)
            if len(coords[0]) == 0:
                x1b, y1b, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])
            else:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                x1b, y1b, w, h = float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)
            sam_masks.append({
                'segmentation': mask_bool,
                'bbox': [x1b, y1b, w, h],
                'area': area,
                'predicted_iou': 0.9,
                'stability_score': 0.95,
            })
            boxes.append({
                "id": idx,
                "bbox": [float(x1b), float(y1b), float(x1b + w), float(y1b + h)]
            })
            idx += 1
    except Exception as e:
        print(f"[YOLO-World+SAM2] Warning: Failed to generate mask for bboxes: {e}")
        return [], []
    return sam_masks, boxes


def masks_from_sam_output(sam_masks, image_size, max_objects=30):
    """
    sam_masks: SAM2가 반환하는 list[dict]
    image_size: (H, W)
    max_objects: 최대 객체 수 제한
    Returns:
        masks: (N_obj, H, W), uint8
        boxes: list of dict, [{'id': i, 'bbox': [x1,y1,x2,y2]}]
    """
    H, W = image_size
    # area 기준으로 정렬 (큰 물체 먼저)
    sam_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)

    # 너무 많은 객체는 제한
    sam_masks = sam_masks[:max_objects]

    mask_list = []
    box_list = []

    for idx, m in enumerate(sam_masks):
        mask = m['segmentation']  # (H, W), bool
        bbox = m['bbox']          # [x, y, w, h]
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h

        mask_list.append(mask.astype(np.uint8))
        box_list.append({
            "id": idx,
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    if len(mask_list) == 0:
        masks = np.zeros((0, H, W), dtype=np.uint8)
    else:
        masks = np.stack(mask_list, axis=0)  # (N_obj, H, W)

    return masks, box_list


def load_text_caption(text_file_path):
    """
    WorDepth의 text caption 파일 로드
    """
    try:
        with open(text_file_path, 'r') as f:
            caption = f.readline().strip()
            return caption
    except Exception:
        return None


def load_depth_gt(depth_path):
    """
    Ground truth depth 로드 (NYU-v2 등)
    OpenCV 기반 벡터화 방식 (PNG: uint16, mm → m)
    """
    try:
        if depth_path.endswith('.png'):
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise ValueError(f"cv2.imread failed for {depth_path}")
            depth = depth.astype(np.float32) / 1000.0  # mm -> m
        elif depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            return None
        return depth
    except Exception as e:
        print(f"Failed to load depth: {e}")
        return None


def _infer_relations_statistical_style(
    boxes: list,
    depth_gt: np.ndarray,
    sam_masks: np.ndarray | None = None,
    max_rel_per_object: int = 5,
    spatial_neighbor_max_norm_dist: float = 0.5,
) -> list[dict]:
    """
    Robustness 논문 방식(μ, σ, M 기반)으로 객체 간 depth 관계(front/behind) 생성.

    - 각 객체 i에 대해:
        μ_i: 마스크 내 유효 깊이(mean)
        σ_i: 마스크 내 깊이 표준편차(std)
        M_i: 마스크 내 최대 깊이(max)
    - 관계 존재 조건:
        |μ_i - μ_j| > (M_i - μ_i) + (M_j - μ_j)
      즉, 두 객체의 평균 깊이 차이가 각 객체의 (평균~최대) 범위 합보다 클 때만 관계를 생성.
    - 순서 결정:
        rep_i = μ_i + σ_i, rep_j = μ_j + σ_j (객체의 뒤쪽·깊은 쪽; 평균+1σ).
        rep_i < rep_j 이면 i의 뒤가 j의 뒤보다 가깝다 → i가 앞(front). 보수적 판단.

    Args:
        boxes: list of dict, 각 객체의 bbox 정보 {'id', 'bbox': [x1,y1,x2,y2]}
        depth_gt: (H, W) GT depth (meters)
        sam_masks: (N, H, W) SAM이 생성한 마스크 (선택, 없으면 bbox로 대체)
        max_rel_per_object: 각 subject 객체당 최대 관계 수 (과도한 관계 수 방지용 소프트 상한)
        spatial_neighbor_max_norm_dist: 공간적으로 너무 멀면 관계 생성하지 않기 위한 정규화 거리 상한
    """
    relations: list[dict] = []

    if depth_gt.ndim == 3:
        depth_gt = depth_gt[:, :, 0] if depth_gt.shape[2] > 0 else depth_gt.squeeze()
    if depth_gt.ndim != 2:
        raise ValueError(f"depth_gt must be 2D, got shape {depth_gt.shape}")

    H, W = depth_gt.shape
    num_objects = len(boxes)

    # 객체별 마스크 및 깊이 통계(μ, σ, M) 계산
    masks: list[np.ndarray | None] = []
    mus: list[float | None] = []
    sigmas: list[float | None] = []
    max_depths: list[float | None] = []

    for i, box_i in enumerate(boxes):
        if sam_masks is not None and i < len(sam_masks):
            mask_i = sam_masks[i] > 0
        else:
            x1i, y1i, x2i, y2i = [int(v) for v in box_i["bbox"]]
            x1i, y1i = max(0, x1i), max(0, y1i)
            x2i, y2i = min(W, x2i), min(H, y2i)
            if x2i <= x1i or y2i <= y1i:
                masks.append(None)
                mus.append(None)
                sigmas.append(None)
                max_depths.append(None)
                continue
            mask_i = np.zeros_like(depth_gt, dtype=bool)
            mask_i[y1i:y2i, x1i:x2i] = True

        depth_i = depth_gt[mask_i]
        if depth_i.size == 0 or np.all(depth_i <= 1e-3):
            masks.append(None)
            mus.append(None)
            sigmas.append(None)
            max_depths.append(None)
            continue

        depth_i_valid = depth_i[depth_i > 1e-3]
        if depth_i_valid.size == 0:
            masks.append(None)
            mus.append(None)
            sigmas.append(None)
            max_depths.append(None)
            continue

        mu_i = float(np.mean(depth_i_valid))
        sigma_i = float(np.std(depth_i_valid))
        max_i = float(np.max(depth_i_valid))

        masks.append(mask_i)
        mus.append(mu_i)
        sigmas.append(sigma_i)
        max_depths.append(max_i)

    # 객체별 관계 수 카운트(soft cap)
    rel_count_per_obj = np.zeros(num_objects, dtype=int)

    # 모든 객체 쌍에 대해 Robustness 조건 기반 depth 관계 생성
    for i in range(num_objects):
        for j in range(num_objects):
            if i == j:
                continue
            if mus[i] is None or mus[j] is None:
                continue

            mu_i = mus[i]
            mu_j = mus[j]
            sigma_i = sigmas[i]
            sigma_j = sigmas[j]
            max_i = max_depths[i]
            max_j = max_depths[j]

            if sigma_i is None or sigma_j is None or max_i is None or max_j is None:
                continue

            # Robustness-style 관계 존재 조건
            lhs = abs(mu_i - mu_j)
            rhs = (max_i - mu_i) + (max_j - mu_j)
            if lhs <= rhs:
                continue

            # 공간적으로 너무 멀리 떨어진 객체는 무시 (우리 기존 방식과 동일한 필터)
            mask_i = masks[i]
            mask_j = masks[j]
            if mask_i is None or mask_j is None:
                continue

            mask_i_coords = np.where(mask_i)
            mask_j_coords = np.where(mask_j)
            if mask_i_coords[0].size == 0 or mask_j_coords[0].size == 0:
                continue

            center_i_y = float(np.mean(mask_i_coords[0]))
            center_i_x = float(np.mean(mask_i_coords[1]))
            center_j_y = float(np.mean(mask_j_coords[0]))
            center_j_x = float(np.mean(mask_j_coords[1]))

            spatial_dist = np.sqrt((center_i_y - center_j_y) ** 2 + (center_i_x - center_j_x) ** 2)
            max_dist = np.sqrt(H ** 2 + W ** 2)
            normalized_dist = float(spatial_dist / max_dist)
            if normalized_dist > spatial_neighbor_max_norm_dist:
                continue

            # 객체당 관계 수 soft cap
            if rel_count_per_obj[i] >= max_rel_per_object:
                continue

            # 대표 깊이: rep = μ + σ (객체의 깊은 쪽=뒤쪽). 이걸로 비교해 보수적으로 앞/뒤 판단
            rep_i = mu_i + sigma_i
            rep_j = mu_j + sigma_j

            # 표현을 median 스타일과 맞추기 위해
            # subject/object는 (i, j) 순서를 그대로 두고,
            # relation만 'front' / 'behind'로 나눈다.
            if rep_i < rep_j:
                # i가 j보다 앞
                subject_idx = int(i)
                object_idx = int(j)
                relation = "front"
            else:
                # j가 i보다 앞 → i는 j보다 뒤
                subject_idx = int(i)
                object_idx = int(j)
                relation = "behind"

            confidence = float(min(0.9, 0.6 + lhs / 5.0))
            relations.append(
                {
                    "subject_idx": subject_idx,
                    "object_idx": object_idx,
                    "relation": relation,
                    "confidence": confidence,
                }
            )
            rel_count_per_obj[subject_idx] += 1

    return relations


def infer_relations_from_depth_only(
    boxes,
    depth_gt,
    sam_masks=None,
    max_rel_per_object: int = 5,
):
    """
    GT depth만으로 객체 간 front/behind 관계를 생성.
    
    현재 구현은 Robustness 논문 방식(μ, σ, M 기반)의 statistical 스타일만 지원한다.
    """
    # 항상 statistical 스타일 사용
    # (이 아래 median 기반 구현은 제거되었음)
    return _infer_relations_statistical_style(
        boxes=boxes,
        depth_gt=depth_gt,
        sam_masks=sam_masks,
        max_rel_per_object=max_rel_per_object,
    )
    
    relations = []

    if depth_gt.ndim == 3:
        depth_gt = depth_gt[:, :, 0] if depth_gt.shape[2] > 0 else depth_gt.squeeze()
    if depth_gt.ndim != 2:
        raise ValueError(f"depth_gt must be 2D, got shape {depth_gt.shape}")

    H, W = depth_gt.shape
    num_objects = len(boxes)

    avg_depths = []
    masks = []
    for i, box_i in enumerate(boxes):
        if sam_masks is not None and i < len(sam_masks):
            mask_i = sam_masks[i] > 0
        else:
            x1i, y1i, x2i, y2i = [int(v) for v in box_i["bbox"]]
            x1i, y1i = max(0, x1i), max(0, y1i)
            x2i, y2i = min(W, x2i), min(H, y2i)
            if x2i <= x1i or y2i <= y1i:
                masks.append(None)
                avg_depths.append(None)
                continue
            mask_i = np.zeros_like(depth_gt, dtype=bool)
            mask_i[y1i:y2i, x1i:x2i] = True
        depth_i = depth_gt[mask_i]
        if len(depth_i) == 0 or np.all(depth_i <= 1e-3):
            masks.append(None)
            avg_depths.append(None)
            continue
        depth_i_valid = depth_i[depth_i > 1e-3]
        if len(depth_i_valid) == 0:
            masks.append(None)
            avg_depths.append(None)
            continue
        avg_depths.append(float(np.median(depth_i_valid)))
        masks.append(mask_i)

    for i in range(num_objects):
        for j in range(num_objects):
            if i == j:
                continue
            if avg_depths[i] is None or avg_depths[j] is None:
                continue
            depth_diff = avg_depths[j] - avg_depths[i]
            if abs(depth_diff) < 0.3 or abs(depth_diff) > 3.0:
                continue
            # 공간적으로 너무 멀리 떨어진 객체는 무시
            mask_i = masks[i]
            mask_j = masks[j]
            if mask_i is None or mask_j is None:
                continue
            mask_i_coords = np.where(mask_i)
            mask_j_coords = np.where(mask_j)
            if len(mask_i_coords[0]) == 0 or len(mask_j_coords[0]) == 0:
                continue
            center_i_y = np.mean(mask_i_coords[0])
            center_i_x = np.mean(mask_i_coords[1])
            center_j_y = np.mean(mask_j_coords[0])
            center_j_x = np.mean(mask_j_coords[1])
            spatial_dist = np.sqrt((center_i_y - center_j_y)**2 + (center_i_x - center_j_x)**2)
            max_dist = np.sqrt(H**2 + W**2)
            normalized_dist = spatial_dist / max_dist
            if normalized_dist > 0.5:
                continue
            # 관계 생성: i가 j보다 앞이면 front, j가 i보다 앞이면 behind
            if depth_diff > 0:  # i가 j보다 앞
                relations.append({
                    "subject_idx": int(i),
                    "object_idx": int(j),
                    "relation": "front",
                    "confidence": float(min(0.9, 0.6 + abs(depth_diff) / 5.0))
                })
            else:  # j가 i보다 앞
                relations.append({
                    "subject_idx": int(i),
                    "object_idx": int(j),
                    "relation": "behind",
                    "confidence": float(min(0.9, 0.6 + abs(depth_diff) / 5.0))
                })
    return relations


def process_single_scene(
    image_dir,
    depth_dir,
    text_dir,
    out_dir,
    sam_mask_gen,
    vlm,
    args
):
    """
    단일 scene 디렉토리 처리 (OpenCV 기반 벡터화)
    """
    os.makedirs(out_dir, exist_ok=True)

    # 이미지 목록 가져오기
    all_images = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )

    # NYU 등: RGB 이미지(rgb_*)만 처리, depth_*는 GT로만 사용
    image_paths = [path for path in all_images if not os.path.basename(path).startswith("depth")]

    scene_name = os.path.basename(image_dir.rstrip(os.sep))
    print(f"\n[{scene_name}] Found {len(image_paths)} RGB images "
          f"(excluding {len(all_images) - len(image_paths)} depth files).")

    for img_path in image_paths:
        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_mask_path = os.path.join(out_dir, f"{fname}_masks.npy")
        out_rel_path = os.path.join(out_dir, f"{fname}_relations.json")

        # 이미 처리된 파일은 스킵
        if os.path.exists(out_mask_path) and os.path.exists(out_rel_path):
            print(f"[Skip] {fname} already processed.")
            continue

        print(f"[Process] {fname}")

        # 이미지 로드 (OpenCV, numpy)
        np_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if np_img is None:
            print(f"  ⚠️ Failed to load image: {img_path}")
            continue
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
        H, W = np_img.shape[:2]

        # YOLO + SAM2로 객체 마스크 생성
        try:
            sam_masks, boxes = generate_object_masks_with_yolo_sam(
                np_img,
                sam_mask_gen['sam2_model'],
                sam_mask_gen['yolo_model'],
                min_mask_region_area=args.sam_min_region_area,
                yolo_conf_thresh=getattr(args, 'yolo_conf_thresh', None)
            )
            if len(sam_masks) > 0:
                masks_np = np.stack([m['segmentation'].astype(np.uint8) for m in sam_masks], axis=0)
            else:
                masks_np = np.zeros((0, H, W), dtype=np.uint8)
            print(f"  - {masks_np.shape[0]} objects from YOLO+SAM2")
        except Exception as e:
            print(f"  ⚠️ YOLO+SAM2 failed for {fname}: {str(e)[:100]}")
            print("  → Skipping this image...")
            np.save(out_mask_path, np.zeros((0, H, W), dtype=bool))
            with open(out_rel_path, "w") as f:
                json.dump([], f)
            continue

        if masks_np.shape[0] == 0:
            np.save(out_mask_path, masks_np)
            with open(out_rel_path, "w") as f:
                json.dump([], f)
            continue

        # Depth GT 로드 (optional but recommended)
        depth_gt = None
        if depth_dir:
            # 예: rgb_00001 -> depth_00001 매칭
            depth_fname = fname.replace("rgb_", "depth_") if fname.startswith("rgb_") else fname
            depth_path = os.path.join(depth_dir, f"{depth_fname}.png")
            if not os.path.exists(depth_path):
                depth_path = os.path.join(depth_dir, f"{depth_fname}.npy")
            if not os.path.exists(depth_path) and fname.startswith("rgb_"):
                # NYU sync 구조: scene/dense/sync_depth_dense_XXXXX.png
                num = fname.replace("rgb_", "")
                depth_path = os.path.join(depth_dir, "dense", f"sync_depth_dense_{num}.png")
            depth_gt = load_depth_gt(depth_path)

        # Relations 추론 (GT depth만 사용) — relation_style: 'median' | 'statistical'
        if depth_gt is not None:
            relation_style = getattr(args, "relation_style", "median")
            relations = infer_relations_from_depth_only(
                boxes,
                depth_gt,
                sam_masks=masks_np,
                max_rel_per_object=args.max_rel_per_object,
                relation_style=relation_style,
            )
            print(f"  - {len(relations)} relations from GT depth (style={relation_style}, YOLO+SAM2 masks)")
        else:
            print(f"  - WARNING: depth_gt not found, skipping relations")
            relations = []

        # 결과 저장
        try:
            # 디렉토리가 없으면 생성
            os.makedirs(os.path.dirname(out_mask_path), exist_ok=True)
            np.save(out_mask_path, masks_np)  # (N_obj, H, W)
        except Exception as e:
            print(f"  ⚠️ Failed to save masks: {e}")
            print(f"  → out_mask_path: {out_mask_path}")
            raise
        
        try:
            os.makedirs(os.path.dirname(out_rel_path), exist_ok=True)
            with open(out_rel_path, "w") as f:
                json.dump(relations, f, indent=2)
        except Exception as e:
            print(f"  ⚠️ Failed to save relations: {e}")
            print(f"  → out_rel_path: {out_rel_path}")
            raise


def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (optional, overrides command line args)")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train",
                        help="Mode to use when config file has both train and test sections")
    parser.add_argument("--dataset", type=str, choices=["nyu", "kitti"], required=False)
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--base_dir", type=str, default=None,
                        help="Base directory containing multiple scene directories (processes all subdirectories)")
    parser.add_argument("--image_dir", type=str, required=False,
                        help="Input image directory (single scene, ignored if --base_dir is set)")
    parser.add_argument("--depth_dir", type=str, default=None,
                        help="Ground truth depth directory (recommended!)")
    parser.add_argument("--text_dir", type=str, default=None,
                        help="Text caption directory (WorDepth format)")
    parser.add_argument("--out_dir", type=str, required=False,
                        help="Output directory to save masks & relations (single scene, ignored if --base_dir is set)")
    parser.add_argument("--out_base_dir", type=str, default=None,
                        help="Base output directory (used with --base_dir)")
    parser.add_argument("--sam_ckpt", type=str, required=False,
                        help="Path to SAM2 checkpoint (.pt file, e.g., sam2_b.pt)")
    # VLM 관련 인자 제거
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_depth_gt", action="store_true",
                        help="Use ground truth depth for relation generation")
    parser.add_argument("--max_rel_per_object", type=int, default=5,
                        help="Maximum number of relations per subject object")
    parser.add_argument("--sam_points_per_side", type=int, default=16,
                        help="SAM2 points_per_side (smaller = lighter)")
    parser.add_argument("--sam_min_region_area", type=int, default=1000,
                        help="Minimum mask area for SAM2 proposals")
    parser.add_argument("--sam_max_objects", type=int, default=20,
                        help="Max number of objects to keep from SAM2")
    parser.add_argument(
        "--relation_style",
        type=str,
        default="median",
        choices=["median", "statistical"],
        help="Depth 관계 추출: 'median' (median+임계값+인접성) 또는 'statistical' (μ,σ,M 기반)",
    )
    args = parser.parse_args()
    
    # YAML config 파일이 제공되면 로드
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Config file not found: {args.config}")
        
        print(f"Loading config from {args.config}...")
        with open(args.config, 'r', encoding='utf-8') as f:
            configs = list(yaml.safe_load_all(f))  # 여러 문서 지원
        
        # train/test 모드 선택
        mode = getattr(args, 'mode', 'train')
        if mode not in ['train', 'test']:
            mode = 'train'
        
        # YAML 문서 파싱 - 첫 번째 문서 사용
        config = configs[0] if configs else {}
        
        # train 또는 test 키가 있는지 확인
        if isinstance(config, dict):
            if 'train' in config or 'test' in config:
                # train/test 섹션이 있는 경우
                if mode in config:
                    config = config[mode]
                    print(f"Using {mode} configuration from YAML...")
                elif 'train' in config:
                    config = config['train']
                    print(f"Using train configuration from YAML (requested {mode} not found, defaulting to train)...")
                elif 'test' in config:
                    config = config['test']
                    print(f"Using test configuration from YAML (requested {mode} not found, defaulting to test)...")
            # train/test 키가 없으면 config를 그대로 사용 (평면 구조)
        
        # YAML의 값으로 command line args 덮어쓰기
        if config:
            for key, value in config.items():
                if value is not None and hasattr(args, key):
                    # boolean 값 처리 (YAML에서 true/false는 bool로 파싱됨)
                    if isinstance(value, bool):
                        if value:
                            setattr(args, key, True)
                        else:
                            setattr(args, key, False)
                    else:
                        setattr(args, key, value)
    
    # 필수 인자 확인
    if not args.dataset:
        raise ValueError("--dataset is required (either via --config or command line)")
    if not args.sam_ckpt:
        raise ValueError("--sam_ckpt is required (either via --config or command line)")

    # --base_dir이 설정되면 여러 디렉토리를 한 번에 처리
    if args.base_dir:
        if not args.out_base_dir:
            raise ValueError("--out_base_dir is required when using --base_dir")

        # 1) SAM2 + YOLO 준비 (한 번만!)
        print("Loading SAM2 and YOLO (will be reused for all scenes)...")
        sam2_model, yolo_model = load_sam_and_yolo(
            sam_ckpt_path=args.sam_ckpt,
            yolo_world_ckpt_path=getattr(args, 'yolo_ckpt', 'yolov8x-world.pt'),
            device=args.device
        )
        sam_mask_gen = {'sam2_model': sam2_model, 'yolo_model': yolo_model}

        # 2) 모든 서브디렉토리 처리 (GT depth만 사용)
        scene_dirs = [d for d in os.listdir(args.base_dir) 
                     if os.path.isdir(os.path.join(args.base_dir, d))]
        scene_dirs.sort()

        print(f"\nProcessing {len(scene_dirs)} scenes with single SAM2+YOLO instance...")

        for scene_name in scene_dirs:
            image_dir = os.path.join(args.base_dir, scene_name)
            depth_dir = image_dir if args.depth_dir is None else os.path.join(args.depth_dir, scene_name)
            out_dir = os.path.join(args.out_base_dir, scene_name)

            process_single_scene(
                image_dir=image_dir,
                depth_dir=depth_dir,
                text_dir=None,
                out_dir=out_dir,
                sam_mask_gen=sam_mask_gen,
                vlm=None,
                args=args
            )

        print("\n✅ All scenes processed!")

    else:
        # 단일 디렉토리 처리 (기존 방식)
        if not args.image_dir or not args.out_dir:
            raise ValueError("--image_dir and --out_dir are required when --base_dir is not set")

        # 1) SAM2 + YOLO 준비
        print("Loading SAM2 and YOLO...")
        sam2_model, yolo_model = load_sam_and_yolo(
            sam_ckpt_path=args.sam_ckpt,
            yolo_world_ckpt_path=getattr(args, 'yolo_ckpt', 'yolov8x-world.pt'),
            device=args.device
        )
        sam_mask_gen = {'sam2_model': sam2_model, 'yolo_model': yolo_model}

        # 2) 단일 scene 처리 (GT depth만 사용)
        process_single_scene(
            image_dir=args.image_dir,
            depth_dir=args.depth_dir,
            text_dir=None,
            out_dir=args.out_dir,
            sam_mask_gen=sam_mask_gen,
            vlm=None,
            args=args
        )

        print("Done.")


if __name__ == "__main__":
    main()