#!/usr/bin/env python3
"""
DocLayNet 사전학습 YOLOv11 모델을 HuggingFace에서 다운로드하는 스크립트
- 중간 저장 기능: 부분 다운로드 지원 및 파일 무결성 검사
"""

import urllib.request
import os
import json
from pathlib import Path


# 설정
MODEL_OUTPUT_DIR = "/root/models/yolo"
BASE_URL = "https://huggingface.co/hantian/yolo-doclaynet/resolve/main"
CHECKPOINT_DIR = "/root/models/yolo/.checkpoints"

# 사용 가능한 모델 (yolov11, yolov12, yolov10, yolov8)
# YOLOv11: n, s, m, l (x는 없음)
# YOLOv12: n, s, m, l
# YOLOv10: n, s, m, b, l
# YOLOv8: n, s, m, l, x

# 예상 파일 크기 (대략적인 값, MB 단위)
EXPECTED_SIZES = {
    # YOLOv11
    'yolov11n': 5.56,
    'yolov11s': 19.3,
    'yolov11m': 40.6,
    'yolov11l': 51.3,
    # YOLOv12
    'yolov12n': 5.6,
    'yolov12s': 19,
    'yolov12m': 40.9,
    'yolov12l': 53.6,
    # YOLOv10
    'yolov10n': 5.84,
    'yolov10s': 16.6,
    'yolov10m': 33.6,
    'yolov10b': 41.6,
    'yolov10l': 52.3,
    # YOLOv8
    'yolov8n': 6.34,
    'yolov8s': 22.6,
    'yolov8m': 52.1,
    'yolov8l': 87.7,
    'yolov8x': 137,
}


def get_checkpoint_path():
    """체크포인트 파일 경로"""
    return os.path.join(CHECKPOINT_DIR, "download_model_checkpoint.json")


def load_checkpoint():
    """체크포인트 로드"""
    checkpoint_path = get_checkpoint_path()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {
        "downloaded_models": [],
        "failed_models": [],
        "status": "not_started"
    }


def save_checkpoint(checkpoint):
    """체크포인트 저장"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = get_checkpoint_path()
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    print(f"💾 체크포인트 저장됨")


def verify_model_file(model_file: Path, model_key: str) -> bool:
    """모델 파일 무결성 검사"""
    if not model_file.exists():
        return False

    file_size_mb = model_file.stat().st_size / (1024 * 1024)

    # 예상 크기의 80% 이상이면 유효한 것으로 간주
    expected = EXPECTED_SIZES.get(model_key, 10)
    if file_size_mb < expected * 0.8:
        print(f"⚠️  파일이 손상되었을 수 있음: {file_size_mb:.1f}MB (예상: ~{expected}MB)")
        return False

    return True


def download_with_resume(url: str, dest_path: Path, model_key: str) -> bool:
    """이어받기 지원 다운로드"""

    temp_path = dest_path.with_suffix('.pt.tmp')

    try:
        # 이미 완성된 파일이 있는지 확인
        if dest_path.exists() and verify_model_file(dest_path, model_key):
            return True

        # 임시 파일이 있으면 이어받기 시도
        start_byte = 0
        if temp_path.exists():
            start_byte = temp_path.stat().st_size
            print(f"📍 이전 다운로드에서 재개: {start_byte / (1024*1024):.1f}MB부터")

        # 요청 헤더 설정 (Range 요청)
        request = urllib.request.Request(url)
        if start_byte > 0:
            request.add_header('Range', f'bytes={start_byte}-')

        # 다운로드
        print(f"📥 다운로드 중...")
        with urllib.request.urlopen(request) as response:
            # 이어받기 여부 확인
            if response.status == 206:  # Partial Content
                mode = 'ab'  # 이어쓰기
            else:
                mode = 'wb'  # 새로 쓰기
                start_byte = 0

            total_size = int(response.headers.get('Content-Length', 0)) + start_byte
            downloaded = start_byte

            with open(temp_path, mode) as f:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # 진행률 출력
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  진행률: {percent:.1f}% ({downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end='', flush=True)

        print()  # 줄바꿈

        # 다운로드 완료 후 파일 이동
        temp_path.rename(dest_path)

        if verify_model_file(dest_path, model_key):
            return True
        else:
            return False

    except Exception as e:
        print(f"\n❌ 다운로드 중 오류: {e}")
        # 임시 파일은 남겨두어 이어받기 가능하게 함
        return False


def download_doclaynet_model(
    model_version: str = 'yolov11',
    model_size: str = 'l',
    output_dir: str = MODEL_OUTPUT_DIR
):
    """
    DocLayNet 사전학습 YOLO 모델 다운로드 (체크포인트 지원)

    Args:
        model_version: 모델 버전 ('yolov8', 'yolov10', 'yolov11', 'yolov12')
        model_size: 모델 크기
            - yolov8: 'n', 's', 'm', 'l', 'x'
            - yolov10: 'n', 's', 'm', 'b', 'l'
            - yolov11: 'n', 's', 'm', 'l' (x 없음!)
            - yolov12: 'n', 's', 'm', 'l'
        output_dir: 출력 디렉토리

    Returns:
        다운로드된 모델 파일 경로
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_key = f"{model_version}{model_size}"
    model_name = f"{model_key}-doclaynet.pt"
    model_url = f"{BASE_URL}/{model_name}"
    model_file = output_path / model_name

    print("=" * 60)
    print("DocLayNet 사전학습 모델 다운로드")
    print("=" * 60)
    print(f"모델: {model_name}")
    print(f"URL: {model_url}")
    print(f"저장 위치: {model_file}")
    print("=" * 60)

    # 체크포인트 로드
    checkpoint = load_checkpoint()

    # 이미 다운로드 완료된 경우
    if model_name in checkpoint['downloaded_models']:
        if model_file.exists() and verify_model_file(model_file, model_key):
            file_size = model_file.stat().st_size / (1024 * 1024)
            print(f"✓ 모델이 이미 다운로드되었습니다: {model_file}")
            print(f"  파일 크기: {file_size:.1f} MB")
            return str(model_file)
        else:
            # 체크포인트에는 있지만 파일이 손상됨
            checkpoint['downloaded_models'].remove(model_name)

    # 다운로드 시도
    checkpoint['status'] = f'downloading_{model_name}'
    save_checkpoint(checkpoint)

    success = download_with_resume(model_url, model_file, model_key)

    if success:
        file_size = model_file.stat().st_size / (1024 * 1024)
        print(f"✓ 다운로드 완료: {model_file}")
        print(f"  파일 크기: {file_size:.1f} MB")

        # 체크포인트 업데이트
        if model_name not in checkpoint['downloaded_models']:
            checkpoint['downloaded_models'].append(model_name)
        if model_name in checkpoint['failed_models']:
            checkpoint['failed_models'].remove(model_name)
        save_checkpoint(checkpoint)

        return str(model_file)
    else:
        print(f"❌ 다운로드 실패: {model_name}")
        if model_name not in checkpoint['failed_models']:
            checkpoint['failed_models'].append(model_name)
        save_checkpoint(checkpoint)
        return None


def download_all_models(model_version: str = 'yolov11', output_dir: str = MODEL_OUTPUT_DIR):
    """특정 버전의 모든 크기 모델 다운로드 (체크포인트 지원)"""

    # 버전별 사용 가능한 크기
    version_sizes = {
        'yolov8': ['n', 's', 'm', 'l', 'x'],
        'yolov10': ['n', 's', 'm', 'b', 'l'],
        'yolov11': ['n', 's', 'm', 'l'],
        'yolov12': ['n', 's', 'm', 'l'],
    }

    sizes = version_sizes.get(model_version, ['n', 's', 'm', 'l'])

    print("=" * 60)
    print(f"{model_version} DocLayNet 모델 전체 다운로드 (체크포인트 지원)")
    print("=" * 60)

    checkpoint = load_checkpoint()
    print(f"이미 다운로드됨: {checkpoint['downloaded_models']}")
    print(f"실패한 다운로드: {checkpoint['failed_models']}")

    downloaded = []
    for size in sizes:
        result = download_doclaynet_model(
            model_version=model_version,
            model_size=size,
            output_dir=output_dir
        )
        if result:
            downloaded.append(result)

    # 최종 상태 업데이트
    checkpoint = load_checkpoint()
    checkpoint['status'] = 'completed' if len(downloaded) == len(sizes) else 'partial'
    save_checkpoint(checkpoint)

    print("\n" + "=" * 60)
    print(f"✓ 다운로드 완료: {len(downloaded)}/{len(sizes)}개")
    for path in downloaded:
        print(f"  - {path}")
    print("=" * 60)

    return downloaded


def main():
    """메인 함수: YOLOv11l (가장 큰 YOLOv11 모델) 다운로드"""

    print("=" * 60)
    print("사용 가능한 DocLayNet 모델:")
    print("  - yolov8: n, s, m, l, x (가장 큰 모델: x)")
    print("  - yolov10: n, s, m, b, l")
    print("  - yolov11: n, s, m, l (x 없음!)")
    print("  - yolov12: n, s, m, l")
    print("=" * 60)

    # YOLOv11l 다운로드 (YOLOv11에서 가장 큰 모델)
    model_path = download_doclaynet_model(
        model_version='yolov11',
        model_size='l'
    )

    if model_path:
        print(f"\n모델 사용법:")
        print("```python")
        print("from ultralytics import YOLO")
        print(f"model = YOLO('{model_path}')")
        print("results = model('document.jpg')")
        print("```")


if __name__ == "__main__":
    main()
