#!/usr/bin/env python3
"""
PubLayNet과 Formula 데이터셋을 병합하는 스크립트
- 극한 병렬 처리: 90GB RAM 최적화
- 멀티프로세싱 + 멀티스레딩 조합
"""

import os
import shutil
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import time

import yaml
from tqdm import tqdm


# 설정 - 로컬 데이터셋 경로
PUBLAYNET_DIR = "/root/publaynet_yolo"
FORMULA_DIR = "/root/formular_data"
MERGED_OUTPUT_DIR = "/root/merged_dataset"
CHECKPOINT_DIR = "/root/merged_dataset/.checkpoints"

# 병렬 처리 설정 (90GB RAM 최적화)
NUM_WORKERS = min(cpu_count(), 64)  # CPU 코어 최대 활용
IO_THREADS = 64  # I/O 스레드 (파일 복사 병렬화)
BATCH_SIZE = 10000  # 배치 크기


def get_checkpoint_path(split: str):
    """체크포인트 파일 경로"""
    return os.path.join(CHECKPOINT_DIR, f"merge_{split}_checkpoint.json")


def load_checkpoint(split: str):
    """체크포인트 로드"""
    checkpoint_path = get_checkpoint_path(split)
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {
        "publaynet_copied": [],
        "formula_copied": [],
        "publaynet_done": False,
        "formula_done": False,
        "status": "not_started"
    }


def save_checkpoint(split: str, checkpoint: dict):
    """체크포인트 저장"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = get_checkpoint_path(split)
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)


def copy_single_file(args):
    """단일 파일 복사 (멀티프로세싱/스레딩용)"""
    src_path, dst_path = args
    try:
        shutil.copy2(src_path, dst_path)
        return (True, None)
    except Exception as e:
        return (False, str(e))


def copy_file_pair_fast(args):
    """이미지와 라벨 파일 쌍을 복사 (고속 버전)"""
    img_name, src_img, src_label, dst_img, dst_label, label_exists = args

    try:
        # 이미지 복사
        shutil.copy2(src_img, dst_img)

        # 라벨 복사 (존재하는 경우)
        if label_exists:
            shutil.copy2(src_label, dst_label)

        return (img_name, True, None)
    except Exception as e:
        return (img_name, False, str(e))


def copy_files_batch_parallel(
    copy_tasks: list,
    desc: str,
    num_workers: int = IO_THREADS,
    batch_size: int = BATCH_SIZE
):
    """
    배치 단위로 파일을 병렬 복사

    Args:
        copy_tasks: [(img_name, src_img, src_label, dst_img, dst_label, label_exists), ...]
        desc: 진행바 설명
        num_workers: 병렬 워커 수
        batch_size: 배치 크기

    Returns:
        (성공 리스트, 실패 수)
    """
    total = len(copy_tasks)
    if total == 0:
        return [], 0

    copied_files = []
    failed_count = 0

    num_batches = (total + batch_size - 1) // batch_size

    with tqdm(total=total, desc=desc) as pbar:
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, total)
            batch_tasks = copy_tasks[batch_start:batch_end]

            # ThreadPoolExecutor로 I/O 병렬화 (파일 복사는 I/O 바운드)
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(copy_file_pair_fast, task): task[0]
                    for task in batch_tasks
                }

                for future in as_completed(futures):
                    img_name, success, error = future.result()
                    if success:
                        copied_files.append(img_name)
                    else:
                        failed_count += 1
                    pbar.update(1)

    return copied_files, failed_count


def merge_publaynet_and_formula_ultra(
    publaynet_dir: str,
    formula_dir: str,
    output_dir: str,
    split: str = 'train',
    save_interval: int = 10000
):
    """
    PubLayNet과 Formula를 병합 (극한 병렬 처리)

    Args:
        publaynet_dir: PubLayNet 데이터셋 경로
        formula_dir: Formula 데이터셋 경로
        output_dir: 출력 디렉토리
        split: 데이터셋 분할 ('train' 또는 'validation')
        save_interval: 체크포인트 저장 간격

    Returns:
        병합된 총 이미지 수
    """
    process_start = time.time()

    # 체크포인트 로드
    checkpoint = load_checkpoint(split)

    # 이미 완료된 경우
    if checkpoint['status'] == 'completed':
        output_images = Path(output_dir) / split / 'images'
        existing_count = len(list(output_images.glob("*.jpg"))) if output_images.exists() else 0
        print(f"✓ {split} 병합이 이미 완료되었습니다 ({existing_count:,}개).")
        return existing_count

    print(f"\n{'='*60}")
    print(f"데이터셋 병합 (극한 병렬): {split}")
    print(f"I/O Threads: {IO_THREADS}, Batch Size: {BATCH_SIZE}")
    print(f"{'='*60}")

    # 이전 진행 상태 출력
    if checkpoint['publaynet_copied'] or checkpoint['formula_copied']:
        print(f"📍 이전 체크포인트에서 재개:")
        print(f"   PubLayNet 복사됨: {len(checkpoint['publaynet_copied']):,}개")
        print(f"   Formula 복사됨: {len(checkpoint['formula_copied']):,}개")

    # 출력 디렉토리 생성
    output_path = Path(output_dir) / split
    output_images = output_path / 'images'
    output_labels = output_path / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # 이미 복사된 파일 세트
    copied_publaynet = set(checkpoint['publaynet_copied'])
    copied_formula = set(checkpoint['formula_copied'])

    # ========== PubLayNet 복사 ==========
    if not checkpoint['publaynet_done']:
        # PubLayNet은 'validation' 폴더를 사용하므로 split 이름 유지
        publaynet_images = Path(publaynet_dir) / split / 'images'
        publaynet_labels = Path(publaynet_dir) / split / 'labels'

# 경로가 없으면 'val' 또는 'validation' 대체 시도
        if not publaynet_images.exists():
            alt_split = 'validation' if split == 'val' else 'val'
            publaynet_images = Path(publaynet_dir) / alt_split / 'images'
            publaynet_labels = Path(publaynet_dir) / alt_split / 'labels'

        if publaynet_images.exists():
            print(f"\nPubLayNet 스캔 중... ({publaynet_images})")
            scan_start = time.time()

            # 이미지 파일 목록
            image_files = list(publaynet_images.glob("*.jpg"))
            print(f"  이미지 파일: {len(image_files):,}개 ({time.time()-scan_start:.1f}초)")

            # 복사할 파일 필터링
            to_copy = [f for f in image_files if f.name not in copied_publaynet]
            print(f"  복사 대상: {len(to_copy):,}개 (이미 복사됨: {len(copied_publaynet):,}개)")

            if to_copy:
                checkpoint['status'] = f'copying_publaynet_{split}'
                save_checkpoint(split, checkpoint)

                # 라벨 파일 목록 미리 조회
                print("  라벨 파일 목록 조회 중...")
                existing_labels = set(f.name for f in publaynet_labels.glob("*.txt")) if publaynet_labels.exists() else set()
                print(f"  라벨 파일: {len(existing_labels):,}개")

                # 복사 작업 준비
                copy_tasks = []
                for img_path in to_copy:
                    img_name = img_path.name
                    label_name = img_name.replace('.jpg', '.txt')

                    copy_tasks.append((
                        img_name,
                        str(img_path),
                        str(publaynet_labels / label_name),
                        str(output_images / img_name),
                        str(output_labels / label_name),
                        label_name in existing_labels
                    ))

                # 병렬 복사 실행
                copied_files, failed = copy_files_batch_parallel(
                    copy_tasks, "PubLayNet 복사", IO_THREADS, BATCH_SIZE
                )

                copied_publaynet.update(copied_files)
                checkpoint['publaynet_copied'] = list(copied_publaynet)
                checkpoint['publaynet_done'] = True
                save_checkpoint(split, checkpoint)
                print(f"  PubLayNet: {len(copied_files):,}개 복사 완료 (실패: {failed}개)")
            else:
                checkpoint['publaynet_done'] = True
                save_checkpoint(split, checkpoint)
        else:
            print(f"⚠️  PubLayNet 경로 없음: {publaynet_images}")
            checkpoint['publaynet_done'] = True
            save_checkpoint(split, checkpoint)
    else:
        print(f"✓ PubLayNet 이미 완료됨 ({len(copied_publaynet):,}개)")

    # ========== Formula 복사 ==========
    if not checkpoint['formula_done']:
        # Formula는 'val' 폴더를 사용하므로 split 이름 변환
        formula_split = 'val' if split == 'validation' else split
        formula_images = Path(formula_dir) / 'formula_only' / formula_split / 'images'
        formula_labels = Path(formula_dir) / 'formula_only' / formula_split / 'labels'

        if formula_images.exists():
            print(f"\nFormula 스캔 중... ({formula_images})")
            scan_start = time.time()

            # 이미지 파일 목록
            image_files = list(formula_images.glob("*.jpg"))
            print(f"  이미지 파일: {len(image_files):,}개 ({time.time()-scan_start:.1f}초)")

            # 복사할 파일 필터링
            to_copy = [f for f in image_files if f.name not in copied_formula]
            print(f"  복사 대상: {len(to_copy):,}개 (이미 복사됨: {len(copied_formula):,}개)")

            if to_copy:
                checkpoint['status'] = f'copying_formula_{split}'
                save_checkpoint(split, checkpoint)

                # 라벨 파일 목록 미리 조회
                print("  라벨 파일 목록 조회 중...")
                existing_labels = set(f.name for f in formula_labels.glob("*.txt")) if formula_labels.exists() else set()
                print(f"  라벨 파일: {len(existing_labels):,}개")

                # 복사 작업 준비
                copy_tasks = []
                for img_path in to_copy:
                    img_name = img_path.name
                    label_name = img_name.replace('.jpg', '.txt')

                    copy_tasks.append((
                        img_name,
                        str(img_path),
                        str(formula_labels / label_name),
                        str(output_images / img_name),
                        str(output_labels / label_name),
                        label_name in existing_labels
                    ))

                # 병렬 복사 실행
                copied_files, failed = copy_files_batch_parallel(
                    copy_tasks, "Formula 복사", IO_THREADS, BATCH_SIZE
                )

                copied_formula.update(copied_files)
                checkpoint['formula_copied'] = list(copied_formula)
                checkpoint['formula_done'] = True
                save_checkpoint(split, checkpoint)
                print(f"  Formula: {len(copied_files):,}개 복사 완료 (실패: {failed}개)")
            else:
                checkpoint['formula_done'] = True
                save_checkpoint(split, checkpoint)
        else:
            print(f"⚠️  Formula 경로 없음: {formula_images}")
            checkpoint['formula_done'] = True
            save_checkpoint(split, checkpoint)
    else:
        print(f"✓ Formula 이미 완료됨 ({len(copied_formula):,}개)")

    # 완료 표시
    checkpoint['status'] = 'completed'
    save_checkpoint(split, checkpoint)

    # 결과 집계
    print("\n결과 집계 중...")
    total_images = len(list(output_images.glob("*.jpg")))
    total_labels = len(list(output_labels.glob("*.txt")))

    elapsed = time.time() - process_start
    print(f"\n✓ {split} 병합 완료")
    print(f"  총 이미지: {total_images:,}개")
    print(f"  총 라벨: {total_labels:,}개")
    print(f"  소요 시간: {elapsed:.1f}초")

    return total_images


def create_yaml(output_dir: str):
    """YAML 설정 파일 생성 (6개 클래스)"""

    yaml_content = {
        'path': output_dir,
        'train': 'train/images',
        'val': 'validation/images',
        'nc': 6,
        'names': {
            0: 'text',
            1: 'title',
            2: 'list',
            3: 'table',
            4: 'figure',
            5: 'formula'
        }
    }

    yaml_path = Path(output_dir) / 'publaynet_formula.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ YAML 생성: {yaml_path}")

    # 내용 확인
    print("\n[YAML 내용]")
    with open(yaml_path, 'r') as f:
        print(f.read())

    return str(yaml_path)


def main():
    """메인 함수: Train과 Validation 세트 병합 및 YAML 생성"""

    total_start = time.time()

    print("=" * 60)
    print("PubLayNet + Formula 데이터셋 병합 (극한 병렬 처리)")
    print(f"CPU Cores: {cpu_count()}")
    print(f"I/O Threads: {IO_THREADS}, Batch Size: {BATCH_SIZE}")
    print("=" * 60)
    print(f"PubLayNet: {PUBLAYNET_DIR}")
    print(f"Formula: {FORMULA_DIR}")
    print(f"출력: {MERGED_OUTPUT_DIR}")
    print("=" * 60)

    # Train 병합
    train_count = merge_publaynet_and_formula_ultra(
        publaynet_dir=PUBLAYNET_DIR,
        formula_dir=FORMULA_DIR,
        output_dir=MERGED_OUTPUT_DIR,
        split='train',
        save_interval=10000
    )

    # Validation 병합
    val_count = merge_publaynet_and_formula_ultra(
        publaynet_dir=PUBLAYNET_DIR,
        formula_dir=FORMULA_DIR,
        output_dir=MERGED_OUTPUT_DIR,
        split='validation',
        save_interval=10000
    )

    # YAML 파일 생성
    yaml_path = create_yaml(MERGED_OUTPUT_DIR)

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("✓ 전체 병합 완료!")
    print("=" * 60)
    print(f"Train: {train_count:,}개")
    print(f"Validation: {val_count:,}개")
    print(f"총 소요 시간: {total_elapsed/60:.1f}분")
    print(f"YAML: {yaml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
