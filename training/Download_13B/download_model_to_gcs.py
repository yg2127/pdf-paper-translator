"""
HuggingFace Tower-13B 모델을 GCS 버킷에 다운로드하는 스크립트

사용법 (SSH 세션에서 실행):
1. GCS 버킷이 /home/yugeon/gcs_bucket에 마운트된 상태에서 실행
2. python download_model_to_gcs.py

이 스크립트는 한 번만 실행하면 됩니다.
이후 학습 시에는 GCS에서 직접 모델을 로드합니다.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 설정
MODEL_NAME = "Unbabel/TowerInstruct-13B-v0.1"
GCS_MODEL_DIR = "/home/yugeon/1207_dataset/models/13B"
# HuggingFace 캐시도 GCS 버킷 내부로 설정 (SSH 세션용)
LOCAL_CACHE_DIR = "/home/yugeon/.cache"

# 환경변수 설정 (스크립트 시작 시 바로 적용)
os.environ['HF_HOME'] = LOCAL_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = f"{LOCAL_CACHE_DIR}/transformers"
os.environ['HF_DATASETS_CACHE'] = f"{LOCAL_CACHE_DIR}/datasets"


def check_model_exists():
    """GCS에 모델이 이미 존재하는지 확인"""
    model_path = Path(GCS_MODEL_DIR)

    if not model_path.exists():
        return False

    # 필수 파일들 확인
    required_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]

    for f in required_files:
        if not (model_path / f).exists():
            return False

    # safetensors 파일 확인
    safetensor_files = list(model_path.glob("*.safetensors"))
    if len(safetensor_files) == 0:
        return False

    return True


def download_model_to_gcs():
    """HuggingFace에서 모델을 다운로드하고 GCS에 저장"""

    print("="*60)
    print("Tower-13B 모델 다운로드 시작")
    print("="*60)

    # 이미 존재하는지 확인
    if check_model_exists():
        print(f"\n모델이 이미 GCS에 존재합니다: {GCS_MODEL_DIR}")
        print("다운로드를 건너뜁니다.")
        return GCS_MODEL_DIR

    # GCS 디렉토리 생성
    os.makedirs(GCS_MODEL_DIR, exist_ok=True)

    print(f"\n모델: {MODEL_NAME}")
    print(f"저장 경로: {GCS_MODEL_DIR}")
    print("\n다운로드 중... (약 26GB, 시간이 걸릴 수 있습니다)")

    # HuggingFace Hub에서 직접 GCS로 다운로드
    try:
        # snapshot_download를 사용하여 모든 파일 다운로드
        downloaded_path = snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=GCS_MODEL_DIR,
            local_dir_use_symlinks=False,  # 심볼릭 링크 대신 실제 파일 복사
            resume_download=True,  # 중단된 다운로드 재개
        )

        print(f"\n다운로드 완료!")
        print(f"모델 저장 위치: {downloaded_path}")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("\n대안: transformers 라이브러리로 다운로드 시도...")

        # 대안: transformers로 로컬에 다운로드 후 복사
        # 환경변수는 이미 스크립트 상단에서 설정됨

        print("토크나이저 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        tokenizer.save_pretrained(GCS_MODEL_DIR)

        print("모델 다운로드 중... (이 작업은 시간이 걸립니다)")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.save_pretrained(GCS_MODEL_DIR, safe_serialization=True)

        del model
        torch.cuda.empty_cache()

        print(f"\n다운로드 완료!")
        print(f"모델 저장 위치: {GCS_MODEL_DIR}")

    # 파일 목록 출력
    print("\n저장된 파일:")
    for f in sorted(Path(GCS_MODEL_DIR).iterdir()):
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name}: {size_mb:.1f} MB")

    return GCS_MODEL_DIR


def verify_model():
    """저장된 모델 검증"""
    print("\n" + "="*60)
    print("모델 검증 중...")
    print("="*60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(GCS_MODEL_DIR, trust_remote_code=True)
        print(f"토크나이저 로드 성공")

        # 모델은 메모리 문제로 전체 로드하지 않고 config만 확인
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(GCS_MODEL_DIR, trust_remote_code=True)
        print(f"모델 config 로드 성공")
        print(f"  - 모델 타입: {config.model_type}")
        print(f"  - Hidden size: {config.hidden_size}")
        print(f"  - Num layers: {config.num_hidden_layers}")

        print("\n모델 검증 완료!")
        return True

    except Exception as e:
        print(f"\n검증 실패: {e}")
        return False


if __name__ == "__main__":
    # 다운로드 실행
    model_path = download_model_to_gcs()

    # 검증
    if verify_model():
        print("\n" + "="*60)
        print("모든 작업 완료!")
        print(f"모델 경로: {model_path}")
        print("\n이제 Colab에서 train_colab_a100.ipynb를 실행하면")
        print("GCS에서 모델을 직접 로드합니다.")
        print(f"\nColab에서 사용할 경로: /content/gcs_bucket/models/13B")
        print("="*60)
    else:
        print("\n모델 검증에 실패했습니다. 다시 다운로드해주세요.")
