#!/usr/bin/env python3
"""
DocLayNet에서 Formula 클래스만 추출하여 YOLO 형식으로 저장하는 스크립트
HuggingFace의 DocLayNet-v1.1 데이터셋을 사용
- 중간 저장 기능: 처리된 인덱스를 저장하여 중단 시 이어서 처리
- 병렬 처리: ThreadPoolExecutor 사용 (멀티프로세싱 BrokenPipe 문제 해결)
"""

import os
import json
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time

import datasets
from PIL import Image
from tqdm import tqdm


# 설정
FORMULA_OUTPUT_DIR = "/root/formular_data"
CHECKPOINT_DIR = "/root/formular_data/.checkpoints"

# 병렬 처리 설정 (A100 + 100GB RAM 최적화)
# 멀티스레딩 사용 (멀티프로세싱 IPC 오버헤드 및 BrokenPipe 문제 회피)
NUM_THREADS = min(cpu_count(), 64)
BATCH_SIZE = 1000  # 배치 크기 (더 작게 해서 안정성 확보)
DATASET_NUM_PROC = min(cpu_count(), 32)


# DocLayNet 카테고리 매핑
# https://github.com/DS4SD/DocLayNet
DOCLAYNET_CATEGORIES = {
    1: "Caption",
    2: "Footnote",
    3: "Formula",      # <-- 이것!
    4: "List-item",
    5: "Page-footer",
    6: "Page-header",
    7: "Picture",
    8: "Section-header",
    9: "Table",
    10: "Text",
    11: "Title"
}

FORMULA_CATEGORY_ID = 3


def get_checkpoint_path(split: str):
    """체크포인트 파일 경로"""
    return os.path.join(CHECKPOINT_DIR, f"extract_formula_{split}_checkpoint.json")


def load_checkpoint(split: str):
    """체크포인트 로드"""
    checkpoint_path = get_checkpoint_path(split)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {
        "last_processed_idx": -1,
        "formula_count": 0,
        "no_formula_count": 0,
        "status": "not_started"
    }


def save_checkpoint(split: str, checkpoint: dict):
    """체크포인트 저장"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = get_checkpoint_path(split)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def process_and_save_single(args):
    """
    단일 아이템 처리 및 저장 (스레드풀용)
    프로세스 간 데이터 전송 없이 스레드 내에서 직접 저장
    """
    idx, image_data, bboxes, categories, split, images_dir, labels_dir = args

    try:
        # Formula만 필터링 (category_id == 3)
        formula_boxes = [
            bbox for bbox, cat_id in zip(bboxes, categories)
            if cat_id == FORMULA_CATEGORY_ID
        ]

        # Formula 없으면 스킵
        if not formula_boxes:
            return (idx, False, None)

        # 이미지 처리
        if isinstance(image_data, dict):
            if 'bytes' in image_data:
                image = Image.open(BytesIO(image_data['bytes']))
            elif 'path' in image_data:
                image = Image.open(image_data['path'])
            else:
                return (idx, False, None)
        elif hasattr(image_data, 'size'):
            image = image_data
        else:
            return (idx, False, None)

        img_width, img_height = image.size

        # YOLO 라벨 생성
        # DocLayNet bbox 형식: [x_min, y_min, width, height] (COCO 형식)
        lines = []
        for bbox in formula_boxes:
            x_min, y_min, box_w, box_h = bbox

            # YOLO 형식 변환 (COCO -> YOLO)
            x_center = (x_min + box_w / 2) / img_width
            y_center = (y_min + box_h / 2) / img_height
            w = box_w / img_width
            h = box_h / img_height

            # 클리핑 (0~1 범위)
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            # width/height가 0이면 스킵
            if w <= 0 or h <= 0:
                continue

            # 클래스 ID = 5 (formula)
            lines.append(f"5 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        # 유효한 bbox가 없으면 스킵
        if not lines:
            return (idx, False, None)

        # 파일명
        image_filename = f"doclaynet_{split}_{idx:08d}.jpg"
        label_filename = f"doclaynet_{split}_{idx:08d}.txt"

        # 이미지 저장
        image_path = images_dir / image_filename
        image.save(image_path, 'JPEG', quality=95, optimize=True)

        # 라벨 저장
        label_path = labels_dir / label_filename
        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

        return (idx, True, None)

    except Exception as e:
        return (idx, False, str(e))


def extract_doclaynet_formulas(
    output_dir: str,
    split: str = 'train',
    max_samples: int = None,
    num_threads: int = NUM_THREADS,
    batch_size: int = BATCH_SIZE
):
    """
    DocLayNet에서 Formula 클래스만 추출 (멀티스레딩 버전)
    """
    # 출력 디렉토리
    output_path = Path(output_dir) / 'formula_only' / split
    images_dir = output_path / 'images'
    labels_dir = output_path / 'labels'

    # 체크포인트 로드
    checkpoint = load_checkpoint(split)

    # 이미 완료된 경우
    if checkpoint['status'] == 'completed':
        existing_count = len(list(images_dir.glob("*.jpg"))) if images_dir.exists() else 0
        print(f"✓ {split} 추출이 이미 완료되었습니다 ({existing_count:,}개).")
        return existing_count

    print(f"\n{'='*60}")
    print(f"DocLayNet Formula 추출: {split}")
    print(f"Threads: {num_threads}, Batch Size: {batch_size}")
    print(f"Formula Category ID: {FORMULA_CATEGORY_ID}")
    print(f"{'='*60}")

    # 이전 진행 상태 출력
    if checkpoint['last_processed_idx'] >= 0:
        print(f"📍 이전 체크포인트에서 재개:")
        print(f"   마지막 처리 인덱스: {checkpoint['last_processed_idx']}")
        print(f"   Formula 추출된 이미지: {checkpoint['formula_count']:,}개")
        print(f"   Formula 없는 이미지: {checkpoint['no_formula_count']:,}개")

    # 출력 디렉토리 생성
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # DocLayNet 로드
    print("DocLayNet 데이터셋 로드 중...")
    load_start = time.time()
    dataset = datasets.load_dataset(
        'docling-project/DocLayNet-v1.1',
        split=split,
        num_proc=DATASET_NUM_PROC
    )
    print(f"✓ DocLayNet-v1.1 로드 완료: {len(dataset):,}개 ({time.time()-load_start:.1f}초)")

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total_examples = len(dataset)

    # 시작 인덱스
    start_idx = checkpoint['last_processed_idx'] + 1
    formula_count = checkpoint['formula_count']
    no_formula_count = checkpoint['no_formula_count']

    if start_idx > 0:
        print(f"인덱스 {start_idx}부터 재개합니다...")

    checkpoint['status'] = 'processing'
    save_checkpoint(split, checkpoint)

    remaining = total_examples - start_idx
    num_batches = (remaining + batch_size - 1) // batch_size
    print(f"\n총 {num_batches}개 배치 처리 예정")

    # 메인 처리 루프
    process_start = time.time()

    with tqdm(total=remaining, desc=f"Processing {split}") as pbar:
        for batch_idx in range(num_batches):
            batch_start = start_idx + batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total_examples)
            current_batch_size = batch_end - batch_start

            # 배치 데이터를 한 번에 가져오기
            batch = dataset[batch_start:batch_end]

            # 처리할 작업 리스트 생성
            tasks = []
            for i in range(current_batch_size):
                idx = batch_start + i
                tasks.append((
                    idx,
                    batch['image'][i],
                    batch['bboxes'][i],
                    batch['category_id'][i],
                    split,
                    images_dir,
                    labels_dir
                ))

            # 멀티스레딩으로 처리 및 저장
            batch_formula = 0
            batch_no_formula = 0
            errors = []

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = {executor.submit(process_and_save_single, task): task[0] for task in tasks}

                for future in as_completed(futures):
                    idx, has_formula, error = future.result()
                    if error:
                        errors.append((idx, error))
                    elif has_formula:
                        batch_formula += 1
                    else:
                        batch_no_formula += 1

            # 에러 출력 (처음 3개만)
            if errors:
                for idx, err in errors[:3]:
                    print(f"\n⚠️  Error at {idx}: {err}")
                if len(errors) > 3:
                    print(f"   ... and {len(errors) - 3} more errors")

            formula_count += batch_formula
            no_formula_count += batch_no_formula

            # 체크포인트 저장
            checkpoint['last_processed_idx'] = batch_end - 1
            checkpoint['formula_count'] = formula_count
            checkpoint['no_formula_count'] = no_formula_count
            save_checkpoint(split, checkpoint)

            pbar.update(current_batch_size)
            elapsed = time.time() - process_start
            pbar.set_postfix({
                'formula': formula_count,
                'skip': no_formula_count,
                'speed': f"{pbar.n / elapsed:.0f}/s" if elapsed > 0 else "N/A"
            })

    # 완료
    checkpoint['status'] = 'completed'
    save_checkpoint(split, checkpoint)

    elapsed = time.time() - process_start
    print(f"\n{'='*60}")
    print(f"✓ Formula 추출 완료")
    print(f"  Formula 있는 이미지: {formula_count:,}개")
    print(f"  Formula 없는 이미지: {no_formula_count:,}개")
    print(f"  처리 시간: {elapsed:.1f}초 ({remaining/elapsed:.0f} 이미지/초)" if elapsed > 0 else "")
    print(f"  저장 위치: {output_path}")
    print(f"{'='*60}")

    return formula_count


def main():
    """메인 함수"""
    print("=" * 60)
    print("DocLayNet Formula 데이터 추출 (멀티스레딩)")
    print(f"CPU Cores: {cpu_count()}")
    print(f"Threads: {NUM_THREADS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("=" * 60)

    total_start = time.time()

    # Train 세트 추출
    train_count = extract_doclaynet_formulas(
        output_dir=FORMULA_OUTPUT_DIR,
        split='train',
        max_samples=None,
        num_threads=NUM_THREADS,
        batch_size=BATCH_SIZE
    )

    # Validation 세트 추출
    val_count = extract_doclaynet_formulas(
        output_dir=FORMULA_OUTPUT_DIR,
        split='val',
        max_samples=None,
        num_threads=NUM_THREADS,
        batch_size=BATCH_SIZE
    )

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("✓ 전체 추출 완료!")
    print(f"  Train: {train_count:,}개")
    print(f"  Validation: {val_count:,}개")
    print(f"  총 소요 시간: {total_elapsed/60:.1f}분")
    print(f"  출력 디렉토리: {FORMULA_OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
