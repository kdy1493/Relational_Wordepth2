"""
VKITTI2 Caption Generation with ExpansionNet/BLIP
==================================================
WordDepth의 text feature를 위한 caption + CLIP embedding 생성

Usage:
    # 테스트 (샘플 5개)
    python src/relation_generation/generate_vkitti2_captions.py \
        --vkitti2_root ./data/vkitti_2.0.3 \
        --mode test --num_samples 5
    
    # 전체 처리
    python src/relation_generation/generate_vkitti2_captions.py \
        --vkitti2_root ./data/vkitti_2.0.3 \
        --output_dir ./data/vkitti2_captions \
        --mode process

Output:
    vkitti2_captions/
    ├── vkitti2_captions.json      # {key: caption}
    └── vkitti2_embeddings.npz     # {key: [768] embedding}
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class CaptionGenerator:
    """
    Image Captioning using BLIP (ExpansionNet fallback)
    
    ExpansionNet이 설치되어 있지 않으면 BLIP 사용 (비슷한 성능)
    """
    
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model = None
        self.processor = None
        self.model_type = None
        
        self._init_model()
    
    def _init_model(self) -> None:
        """모델 초기화 (ExpansionNet -> BLIP fallback)"""
        # Try ExpansionNet first
        try:
            from expansionnet_v2 import ExpansionNetV2

            self.model = ExpansionNetV2.from_pretrained("jchenghu/ExpansionNet_v2")
            self.model.to(self.device)
            self.model.eval()
            self.model_type = "expansionnet"
            print("[Caption] Using ExpansionNet v2")
            return
        except ImportError:
            pass
        
        # Fallback to BLIP
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            
            print("[Caption] Loading BLIP (ExpansionNet not available)...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(self.device)
            self.model.eval()
            self.model_type = "blip"
            print("[Caption] Using BLIP-large")
        except ImportError as e:
            raise ImportError(
                "transformers 패키지가 필요합니다: pip install transformers"
            ) from e
    
    @torch.no_grad()
    def generate(self, image):
        """단일 이미지 caption 생성"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if self.model_type == "blip":
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            output = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
        else:
            caption = self.model.generate(image)
        
        return caption


class CLIPEncoder:
    """CLIP Text Encoder for WordDepth"""
    
    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print("[CLIP] Loading CLIP ViT-L/14...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.model.eval()
            self.embed_dim = self.model.config.projection_dim  # 768
            print(f"[CLIP] Embedding dim: {self.embed_dim}")
        except ImportError as e:
            raise ImportError(
                "transformers 패키지가 필요합니다: pip install transformers"
            ) from e
    
    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        """텍스트를 CLIP embedding으로 변환"""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        out = self.model.get_text_features(**inputs)
        # HuggingFace 버전에 따라 tensor 또는 BaseModelOutputWithPooling 반환
        if isinstance(out, torch.Tensor):
            features = out
        else:
            pooler = getattr(out, "pooler_output", None)
            last_h = getattr(out, "last_hidden_state", None)
            if pooler is not None:
                features = pooler
            elif last_h is not None:
                features = last_h[:, 0]  # CLS token
            else:
                raise ValueError("CLIP output에서 embedding tensor를 찾을 수 없음")
        return features.cpu().numpy()  # [1, 768]


# VKITTI2 caption/embedding 키 형식 (dataloader와 동일해야 함)
# key = "{scene}/{condition}/{rgb_filename}"  e.g. "Scene01/clone/rgb_00000.jpg"
# dataloader: caption_key = f"{scene}/{condition}/rgb_{frame_id}.jpg" (frame_id = "00000") → 동일


def scan_vkitti2(vkitti2_root, scenes=None, conditions=None):
    """VKITTI2 이미지 목록 스캔. key 형식: SceneXX/condition/rgb_XXXXX.jpg (dataloader lookup과 일치)"""
    scenes = scenes or ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
    conditions = conditions or ["clone"]
    
    samples = []
    
    for scene in scenes:
        for condition in conditions:
            rgb_dir = os.path.join(vkitti2_root, scene, condition, "frames", "rgb", "Camera_0")
            
            if not os.path.exists(rgb_dir):
                continue
            
            rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".jpg")])
            
            for rgb_file in rgb_files:
                samples.append(
                    {
                        "path": os.path.join(rgb_dir, rgb_file),
                        "key": f"{scene}/{condition}/{rgb_file}",  # e.g. Scene01/clone/rgb_00000.jpg
                    }
                )
    
    return samples


def test_captions(vkitti2_root, num_samples: int = 5, device: str = "cuda") -> None:
    """Caption 생성 테스트"""
    print("\n" + "=" * 60)
    print("VKITTI2 Caption Test")
    print("=" * 60)
    
    samples = scan_vkitti2(vkitti2_root)[:num_samples]
    
    captioner = CaptionGenerator(device=device)
    
    for i, sample in enumerate(samples):
        caption = captioner.generate(sample["path"])
        print(f"\n[{i+1}] {sample['key']}")
        print(f"    Caption: {caption}")
    
    print("\n" + "=" * 60)


def process_all(vkitti2_root, output_dir, scenes=None, conditions=None, device: str = "cuda") -> None:
    """전체 데이터셋 caption + embedding 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    samples = scan_vkitti2(vkitti2_root, scenes, conditions)
    print(f"Found {len(samples)} images")
    
    # 모델 로드
    captioner = CaptionGenerator(device=device)
    clip_encoder = CLIPEncoder(device=device)
    
    captions = {}
    embeddings = {}
    
    for sample in tqdm(samples, desc="Processing"):
        try:
            # Caption 생성
            caption = captioner.generate(sample["path"])
            
            # CLIP embedding
            embedding = clip_encoder.encode(caption)
            
            captions[sample["key"]] = caption
            embeddings[sample["key"]] = embedding[0]  # [768]
            
        except Exception as e:  # noqa: BLE001
            print(f"Error processing {sample['key']}: {e}")
            continue
    
    # 저장
    caption_path = os.path.join(output_dir, "vkitti2_captions.json")
    with open(caption_path, "w") as f:
        json.dump(captions, f, indent=2)
    
    emb_path = os.path.join(output_dir, "vkitti2_embeddings.npz")
    np.savez_compressed(emb_path, **embeddings)
    
    # 검증: 저장된 키가 스캔한 전체 목록과 일치하는지 확인 (한 번에 정확히 나오는지)
    expected_keys = {s["key"] for s in samples}
    missing_in_emb = expected_keys - set(embeddings.keys())
    missing_in_cap = expected_keys - set(captions.keys())
    if missing_in_emb or missing_in_cap:
        print(f"\n[경고] 일부 샘플이 실패하여 누락됨: embedding {len(missing_in_emb)}개, caption {len(missing_in_cap)}개")
        if missing_in_emb and len(missing_in_emb) <= 5:
            print("  누락 embedding 키 예:", list(missing_in_emb)[:5])
    else:
        print(f"\n[검증] 모든 {len(expected_keys)}개 샘플에 대해 caption/embedding 키 일치.")
    for k in list(embeddings.keys())[:1]:
        arr = np.asarray(embeddings[k])
        if arr.shape != (768,):
            print(f"[경고] embedding shape이 (768,)이 아님: {k} -> {arr.shape}")
        break

    print(f"\nSaved {len(captions)} captions to: {caption_path}")
    print(f"Saved {len(embeddings)} embeddings to: {emb_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vkitti2_root",
        type=str,
        required=True,
        help="VKITTI2 root directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/vkitti2_captions",
        help="Output directory for captions and embeddings",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "process"],
        default="test",
        help="test: quick test, process: full dataset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples for test mode",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        nargs="+",
        default=None,
        help="Scenes to process (default: all)",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["clone"],
        help="Conditions to process (default: clone)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    if args.mode == "test":
        test_captions(args.vkitti2_root, args.num_samples, args.device)
    else:
        process_all(
            args.vkitti2_root,
            args.output_dir,
            args.scenes,
            args.conditions,
            args.device,
        )


if __name__ == "__main__":
    main()

