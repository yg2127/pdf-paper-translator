"""
PubLayNet Parquet 형식을 YOLO 형식으로 변환하는 스크립트

Hugging Face에서 다운로드한 parquet 형식의 PubLayNet 데이터셋을 YOLO 형식으로 변환합니다.
GCS 버킷에 직접 저장하고, 캐시도 GCS를 사용하여 로컬 디스크 사용을 최소화합니다.

사용법:
    # GCS 버킷에 직접 저장 (권장)
    python convert_publaynet_parquet_to_yolo.py \
        --data-dir /home/yugeon/gcs_bucket/Dataset/publayset/data \
        --output-dir /home/yugeon/gcs_bucket/datasets/publaynet_yolo

필수 라이브러리:
    pip install datasets pillow tqdm
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
import os
import sys
import shutil

# ⚠️ 중요: Hugging Face 캐시를 GCS 버킷으로 리다이렉트
# 이렇게 하면 로컬 디스크 대신 GCS에 캐시가 저장됩니다
os.environ['HF_DATASETS_CACHE'] = '/home/yugeon/gcs_bucket/.cache/huggingface/datasets'
os.environ['HF_HOME'] = '/home/yugeon/gcs_bucket/.cache/huggingface'

try:
    import datasets
    from PIL import Image
except ImportError as e:
    print(f"필요한 라이브러리가 설치되지 않았습니다: {e}")
    print("설치: pip install datasets pillow")
    exit(1)


def convert_parquet_to_yolo(
    parquet_files: list,
    output_dir: str,
    split: str = "train",
    max_samples: int = None,
    batch_size: int = 1000,
    start_from: int = 0
):
    """
    Parquet 파일을 YOLO 형식으로 변환

    Args:
        parquet_files: parquet 파일 경로 리스트
        output_dir: 출력 디렉토리 (GCS 경로 권장)
        split: 'train' 또는 'validation'
        max_samples: 최대 샘플 수 (None이면 전체)
        batch_size: 배치 크기 (진행 상태 저장 주기)
        start_from: 시작 인덱스 (중단된 작업 재개용)
    """

    print(f"\n{'='*60}")
    print(f"{split.upper()} 데이터 변환")
    print(f"{'='*60}")
    print(f"⚠️  캐시 디렉토리: {os.environ.get('HF_DATASETS_CACHE', 'default')}")

    # 출력 디렉토리 생성
    output_path = Path(output_dir) / split
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # 진행 상태 파일
    progress_file = output_path / f".progress_{split}.txt"

    # 이전 진행 상태 확인
    if start_from == 0 and progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                start_from = int(f.read().strip())
            print(f"✓ 이전 진행 상태 발견: {start_from}번째부터 재개합니다.")
        except:
            pass

    # Parquet 파일 로드
    print(f"Loading parquet files: {len(parquet_files)} files")
    print(f"⏳ 데이터셋 로딩 중... (캐시에 저장됩니다)")

    dataset = datasets.load_dataset(
        'parquet',
        data_files=parquet_files,
        split='train',  # parquet에서는 항상 'train' split
        cache_dir=os.environ.get('HF_DATASETS_CACHE')
    )

    print(f"Total examples: {len(dataset):,}")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Using {len(dataset):,} samples")

    # PubLayNet 카테고리 매핑
    # category_id는 1-indexed (1: text, 2: title, 3: list, 4: table, 5: figure)
    # YOLO는 0-indexed이므로 -1 필요

    converted_count = 0
    skipped_count = 0

    # 데이터셋 변환
    print(f"시작 인덱스: {start_from}")
    print(f"전체 데이터셋 크기: {len(dataset)}")

    # start_from부터 시작하도록 dataset을 슬라이스
    if start_from > 0:
        print(f"⏭️  처음 {start_from}개 샘플 스킵")
        dataset_to_process = dataset.select(range(start_from, len(dataset)))
        start_idx = start_from
    else:
        dataset_to_process = dataset
        start_idx = 0

    for relative_idx, example in enumerate(tqdm(dataset_to_process, desc="Converting", total=len(dataset_to_process))):
        # 실제 인덱스 계산
        idx = start_idx + relative_idx

        try:
            # 이미지 저장
            image = example['image']
            image_id = example['id']
            annotations = example['annotations']

            # 이미지 타입 확인 및 변환
            if isinstance(image, dict):
                # 딕셔너리 형태로 저장된 경우 (bytes 데이터)
                if 'bytes' in image:
                    from io import BytesIO
                    image = Image.open(BytesIO(image['bytes']))
                elif 'path' in image:
                    image = Image.open(image['path'])
                else:
                    print(f"\n⚠️  Unknown image format for {image_id}, skipping...")
                    skipped_count += 1
                    continue
            elif not hasattr(image, 'size'):
                # PIL Image가 아닌 경우
                print(f"\n⚠️  Invalid image type for {image_id}: {type(image)}, skipping...")
                skipped_count += 1
                continue

            # 이미지 크기
            img_width, img_height = image.size

            # 이미지 파일명
            image_filename = f"PMC{image_id:010d}.jpg"
            image_path = images_dir / image_filename

            # 이미지 저장 (이미 존재하면 스킵)
            if not image_path.exists():
                image.save(image_path, 'JPEG', quality=95, optimize=True)

            # 라벨 파일명
            label_filename = f"PMC{image_id:010d}.txt"
            label_path = labels_dir / label_filename

            # YOLO 형식 라벨 생성
            lines = []

            for ann in annotations:
                bbox = ann['bbox']
                category_id = ann['category_id']

                # bbox가 비어있거나 유효하지 않으면 스킵
                if not bbox or len(bbox) < 4:
                    continue

                # COCO bbox: [x_min, y_min, width, height]
                x_min, y_min, bbox_width, bbox_height = bbox[:4]

                # 유효성 검사
                if bbox_width <= 0 or bbox_height <= 0:
                    continue

                # YOLO 형식으로 변환 (정규화)
                x_center = (x_min + bbox_width / 2) / img_width
                y_center = (y_min + bbox_height / 2) / img_height
                w = bbox_width / img_width
                h = bbox_height / img_height

                # 범위 클리핑 (0-1)
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                # 클래스 ID (1-indexed → 0-indexed)
                class_id = category_id - 1

                # 유효한 클래스 ID인지 확인 (0-4)
                if class_id < 0 or class_id > 4:
                    continue

                # YOLO 형식: class_id x_center y_center width height
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            # 라벨 파일 저장
            with open(label_path, 'w') as f:
                f.write('\n'.join(lines))

            converted_count += 1

            # 배치마다 진행 상태 저장
            if (relative_idx + 1) % batch_size == 0:
                # 진행 상태 저장 (실제 인덱스 기준)
                with open(progress_file, 'w') as f:
                    f.write(str(idx + 1))

                print(f"\n✓ 진행 상태 저장: {idx + 1:,}/{len(dataset):,} ({(idx+1)/len(dataset)*100:.1f}%)")
                print(f"  변환 완료: {converted_count:,}개, 스킵: {skipped_count}개")

                # 강제 flush (GCS에 데이터 쓰기)
                sys.stdout.flush()

        except Exception as e:
            print(f"\n⚠️  Error processing example {idx}: {e}")
            skipped_count += 1
            continue

    # 완료 시 진행 상태 파일 삭제
    if progress_file.exists():
        progress_file.unlink()

    print(f"\n{'='*60}")
    print(f"✓ 변환 완료: {converted_count:,}개")
    if skipped_count > 0:
        print(f"⚠️  스킵됨: {skipped_count}개")
    print(f"{'='*60}")

    return converted_count


def create_yaml_config(output_dir: str):
    """YOLO 학습용 YAML 설정 파일 생성"""

    output_path = Path(output_dir).resolve()

    yaml_content = f"""# PubLayNet Dataset Configuration for YOLO
# Auto-generated from Hugging Face parquet files

path: {output_path}
train: train/images
val: validation/images

# PubLayNet Classes (5)
names:
  0: text
  1: title
  2: list
  3: table
  4: figure

# Number of classes
nc: 5

# Dataset info
# PubLayNet: largest dataset ever for document layout analysis
# Source: https://github.com/ibm-aur-nlp/PubLayNet
# Paper: Zhong et al., ICDAR 2019
"""

    yaml_path = output_path / "publaynet.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n✓ YAML 설정 파일 생성: {yaml_path}")
    return str(yaml_path)


def verify_conversion(output_dir: str):
    """변환 결과 검증"""

    output_path = Path(output_dir)

    print(f"\n{'='*60}")
    print("변환 결과 검증")
    print(f"{'='*60}")

    for split in ['train', 'validation']:
        split_path = output_path / split

        if not split_path.exists():
            print(f"\n{split.upper()}: 존재하지 않음")
            continue

        images_dir = split_path / "images"
        labels_dir = split_path / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            print(f"\n{split.upper()}: 디렉토리 구조 불완전")
            continue

        image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
        label_count = len(list(labels_dir.glob("*.txt")))

        print(f"\n{split.upper()}:")
        print(f"  이미지: {image_count:,}개")
        print(f"  라벨: {label_count:,}개")

        if image_count != label_count:
            print(f"  ⚠️  경고: 이미지와 라벨 수가 다릅니다!")
        else:
            print(f"  ✓ 검증 통과")


def check_disk_space():
    """디스크 공간 확인"""
    import subprocess

    print(f"\n{'='*60}")
    print("💾 디스크 사용량 확인")
    print(f"{'='*60}")

    # 홈 디렉토리 확인
    result = subprocess.run(['df', '-h', os.path.expanduser('~')],
                          capture_output=True, text=True)
    print("\n로컬 디스크:")
    print(result.stdout)

    # 로컬 캐시 확인
    local_cache = Path.home() / '.cache' / 'huggingface' / 'datasets'
    if local_cache.exists():
        result = subprocess.run(['du', '-sh', str(local_cache)],
                              capture_output=True, text=True)
        print(f"⚠️  로컬 캐시: {result.stdout.strip()}")
        print(f"   위치: {local_cache}")
    else:
        print(f"✓ 로컬 캐시 없음")

    # GCS 캐시 확인
    gcs_cache = Path(os.environ.get('HF_DATASETS_CACHE', ''))
    if gcs_cache.exists():
        print(f"\n✓ GCS 캐시 위치: {gcs_cache}")


def main():
    parser = argparse.ArgumentParser(
        description='PubLayNet Parquet 형식을 YOLO 형식으로 변환 (GCS 캐시 사용)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Parquet 파일이 있는 디렉토리 (예: /home/yugeon/gcs_bucket/Dataset/publayset/data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/yugeon/gcs_bucket/datasets/publaynet_yolo',
        help='출력 디렉토리 (기본값: /home/yugeon/gcs_bucket/datasets/publaynet_yolo - GCS 버킷)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='배치 크기 (진행 상태 저장 주기, 기본값: 1000)'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='시작 인덱스 (중단된 작업 재개용, 0이면 자동 감지)'
    )
    parser.add_argument(
        '--max-train-samples',
        type=int,
        default=None,
        help='Train 최대 샘플 수 (None이면 전체, 테스트시 작은 값 사용)'
    )
    parser.add_argument(
        '--max-val-samples',
        type=int,
        default=None,
        help='Validation 최대 샘플 수 (None이면 전체)'
    )
    parser.add_argument(
        '--train-only',
        action='store_true',
        help='Train 데이터만 변환'
    )
    parser.add_argument(
        '--val-only',
        action='store_true',
        help='Validation 데이터만 변환'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # 디스크 공간 확인
    check_disk_space()

    # 데이터 디렉토리 확인
    if not data_dir.exists():
        print(f"\n❌ 오류: 데이터 디렉토리가 존재하지 않습니다: {data_dir}")
        return

    # Train 파일 찾기
    train_files = sorted(data_dir.glob("train-*.parquet"))
    val_files = sorted(data_dir.glob("validation-*.parquet"))

    print(f"\n{'='*60}")
    print("PubLayNet → YOLO 변환")
    print(f"{'='*60}")
    print(f"Train files found: {len(train_files)}")
    print(f"Validation files found: {len(val_files)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Cache directory: {os.environ.get('HF_DATASETS_CACHE')}")

    if "/gcs_bucket/" in args.output_dir:
        print("✓ 출력: GCS 버킷에 직접 저장됩니다.")
    else:
        print("⚠️  출력: 로컬 디스크에 저장됩니다. 용량 주의!")

    if "/gcs_bucket/" in os.environ.get('HF_DATASETS_CACHE', ''):
        print("✓ 캐시: GCS 버킷에 저장됩니다 (로컬 디스크 사용 최소화).")
    else:
        print("⚠️  캐시: 로컬 디스크에 저장됩니다. 100GB+ 필요!")

    if len(train_files) == 0 and len(val_files) == 0:
        print("❌ 오류: parquet 파일을 찾을 수 없습니다.")
        return

    # Train 데이터 변환
    if not args.val_only and len(train_files) > 0:
        convert_parquet_to_yolo(
            parquet_files=[str(f) for f in train_files],
            output_dir=args.output_dir,
            split="train",
            max_samples=args.max_train_samples,
            batch_size=args.batch_size,
            start_from=args.start_from
        )

    # Validation 데이터 변환
    if not args.train_only and len(val_files) > 0:
        convert_parquet_to_yolo(
            parquet_files=[str(f) for f in val_files],
            output_dir=args.output_dir,
            split="validation",
            max_samples=args.max_val_samples,
            batch_size=args.batch_size,
            start_from=0  # validation은 항상 처음부터
        )

    # YAML 설정 파일 생성
    yaml_path = create_yaml_config(args.output_dir)

    # 검증
    verify_conversion(args.output_dir)

    # 최종 디스크 확인
    check_disk_space()

    # 사용법 출력
    print(f"\n{'='*60}")
    print("✓ 모든 변환 완료!")
    print(f"{'='*60}")
    print(f"\n다음 단계: YOLO 학습")
    print(f"  cd /home/yugeon/files/YOLO")
    print(f"  python train_yolo_publaynet.py train \\")
    print(f"    --data {yaml_path} \\")
    print(f"    --model yolo11m.pt \\")
    print(f"    --epochs 100 \\")
    print(f"    --batch 16 \\")
    print(f"    --device 0")


if __name__ == "__main__":
    main()
