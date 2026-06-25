"""
YOLO PubLayNet 학습 스크립트 (로컬 데이터셋용)

================================================================================
📍 데이터셋 경로 설정 위치 (아래 섹션에서 수정하세요!)
================================================================================

1. 데이터셋 경로 (라인 ~35-40):
   - DATASET_PATH: 데이터셋 루트 경로 (images/, labels/ 폴더가 있는 위치)
   - DATA_YAML: YOLO 데이터셋 설정 파일 경로 (.yaml 파일)

2. 모델 경로 (라인 ~45):
   - MODEL_PATH: 로컬에 다운받은 YOLO 모델 파일 경로 (.pt 파일)

3. 학습 출력 경로 (라인 ~55):
   - OUTPUT_DIR: 학습 결과 저장 경로 (weights, logs 등)

================================================================================

주요 기능:
- 로컬 데이터셋에서 학습
- 체크포인트 자동 저장
- 이전 학습 재개 가능
- YOLOv11m 사용

사전 준비:
1. GPU 환경 확인
2. 데이터셋 경로 설정
3. ultralytics 패키지 설치
"""

import os
import subprocess
from pathlib import Path

# ==================================================================================
# 1️⃣ 데이터셋 경로 설정 (⚠️ 여기를 수정하세요!)
# ==================================================================================
DATASET_PATH = "/root/merged_dataset"  # 데이터셋 루트 경로
DATA_YAML = "/root/merged_dataset/publaynet_formula.yaml"  # 데이터셋 YAML 파일 경로

# ==================================================================================
# 2️⃣ 모델 경로 설정 (⚠️ 여기를 수정하세요!)
# ==================================================================================
MODEL_PATH = "/root/models/yolo/yolov11l-doclaynet.pt"  # 로컬에 다운받은 YOLO 모델 경로

# ==================================================================================
# 3️⃣ 학습 설정
# ==================================================================================
EPOCHS = 50
BATCH_SIZE = 64  # GPU 메모리에 따라 조절 (~35GB VRAM 사용 목표)
IMAGE_SIZE = 640
DEVICE = 0  # GPU 0번 (CPU 사용 시 "cpu")

# 메모리 설정
NUM_WORKERS = 16  # DRAM 사용량 증가를 위한 워커 수
CACHE = "disk"  # 디스크에 캐싱 (RAM 대신 HDD/SSD 사용)

# ==================================================================================
# 4️⃣ 출력 경로 설정
# ==================================================================================
OUTPUT_DIR = "./runs/publaynet"  # 학습 결과 저장 경로
EXPERIMENT_NAME = "yolo11m_publaynet"


def run_command(cmd, shell=True):
    """명령어 실행 헬퍼 함수"""
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


def check_gpu():
    """GPU 및 CUDA 확인"""
    print("=" * 60)
    print("GPU 및 CUDA 확인")
    print("=" * 60)

    # nvidia-smi로 GPU 확인
    result = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if result.returncode != 0:
        print("❌ GPU를 찾을 수 없습니다.")
        print("   CUDA GPU가 필요합니다. 환경을 확인하세요.")
        return False

    # PyTorch CUDA 확인
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ PyTorch CUDA 사용 가능")
            print(f"  - PyTorch 버전: {torch.__version__}")
            print(f"  - CUDA 버전: {torch.version.cuda}")
            print(f"  - GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ PyTorch에서 CUDA를 사용할 수 없습니다.")
            print("   CUDA 버전의 PyTorch를 설치하세요:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            return False
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return False



def check_dataset():
    """데이터셋 확인"""
    print("\n" + "=" * 60)
    print("데이터셋 확인")
    print("=" * 60)

    # YAML 파일 확인
    if os.path.exists(DATA_YAML):
        print(f"✓ 데이터셋 YAML 발견: {DATA_YAML}")
        with open(DATA_YAML, 'r') as f:
            print("\n--- YAML 내용 ---")
            print(f.read())
            print("-----------------")
    else:
        print(f"❌ 데이터셋 YAML 없음: {DATA_YAML}")
        print("   YAML 파일 경로를 확인하세요.")
        return False

    # 데이터셋 폴더 확인
    if os.path.exists(DATASET_PATH):
        print(f"✓ 데이터셋 경로 확인: {DATASET_PATH}")
    else:
        print(f"❌ 데이터셋 경로 없음: {DATASET_PATH}")
        return False

    return True


def setup_yaml():
    """데이터셋 YAML 파일 생성 (없을 경우)"""
    print("\n" + "=" * 60)
    print("데이터셋 YAML 파일 확인/생성")
    print("=" * 60)

    if os.path.exists(DATA_YAML):
        print(f"✓ 기존 YAML 파일 사용: {DATA_YAML}")
        return

    # YAML 파일이 없으면 생성
    yaml_content = f"""# PubLayNet + Formula Dataset Configuration for YOLO

path: {DATASET_PATH}
train: train/images
val: validation/images

# PubLayNet + Formula Classes (6)
names:
  0: text
  1: title
  2: list
  3: table
  4: figure
  5: formula

nc: 6
"""
    os.makedirs(os.path.dirname(DATA_YAML), exist_ok=True)
    with open(DATA_YAML, 'w') as f:
        f.write(yaml_content)

    print("✓ YAML 파일 생성 완료")
    print(yaml_content)


def check_model():
    """모델 파일 확인"""
    print("\n" + "=" * 60)
    print("모델 파일 확인")
    print("=" * 60)

    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✓ 모델 파일 발견: {MODEL_PATH} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ 모델 파일 없음: {MODEL_PATH}")
        print("   모델 파일 경로를 확인하세요.")
        return False


def check_checkpoint():
    """이전 체크포인트 확인"""
    global MODEL_PATH

    print("\n" + "=" * 60)
    print("체크포인트 확인")
    print("=" * 60)

    checkpoint_dir = Path(OUTPUT_DIR) / EXPERIMENT_NAME
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # last.pt 확인
    last_pt = checkpoint_dir / "weights" / "last.pt"
    if last_pt.exists():
        print(f"✓ 이전 체크포인트 발견: {last_pt}")
        print("   학습을 재개합니다.")
        MODEL_PATH = str(last_pt)
        return True, checkpoint_dir
    else:
        print("ℹ️  이전 체크포인트 없음. 처음부터 학습합니다.")
        return False, checkpoint_dir


def train(resume, checkpoint_dir):
    """YOLO 학습 실행"""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("학습 설정")
    print("=" * 60)
    print(f"모델: {MODEL_PATH}")
    print(f"데이터셋: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"출력 경로: {OUTPUT_DIR}/{EXPERIMENT_NAME}")

    # 모델 로드
    model = YOLO(MODEL_PATH)

    print("\n" + "=" * 60)
    print("학습 시작!")
    print("=" * 60)

    # 학습 실행
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=NUM_WORKERS,
        cache=CACHE,  # RAM에 데이터셋 캐싱
        project=OUTPUT_DIR,
        name=EXPERIMENT_NAME,
        exist_ok=True,
        resume=resume,

        # 옵티마이저 설정
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,

        # Early stopping
        patience=20,

        # 저장 설정
        save=True,
        save_period=10,  # 10 에포크마다 저장

        # 데이터 증강 (문서용 - 최소화)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.1,
        degrees=0.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,

        # 검증 설정
        val=True,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("학습 완료!")
    print("=" * 60)

    return results


def evaluate(checkpoint_dir):
    """모델 평가"""
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("모델 평가")
    print("=" * 60)

    best_model_path = checkpoint_dir / "weights" / "best.pt"

    if best_model_path.exists():
        model = YOLO(str(best_model_path))

        print("모델 평가 중...")
        metrics = model.val(
            data=DATA_YAML,
            split="val",
            device=DEVICE
        )

        print("\n" + "=" * 60)
        print("평가 결과")
        print("=" * 60)
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
        print("=" * 60)

        # 클래스별 성능
        class_names = ['text', 'title', 'list', 'table', 'figure', 'formula']
        print("\n클래스별 AP50:")
        for name, ap in zip(class_names, metrics.box.ap50):
            print(f"  {name}: {ap:.4f}")

        return model
    else:
        print(f"❌ best.pt 파일이 없습니다: {best_model_path}")
        return None


def test_inference(model, checkpoint_dir):
    """테스트 추론"""
    print("\n" + "=" * 60)
    print("테스트 추론")
    print("=" * 60)

    class_names = ['text', 'title', 'list', 'table', 'figure', 'formula']

    # validation 이미지 경로 찾기
    val_images_path = Path(DATASET_PATH) / "validation" / "images"
    if not val_images_path.exists():
        val_images_path = Path(DATASET_PATH) / "val" / "images"

    val_images = list(val_images_path.glob("*.jpg")) + list(val_images_path.glob("*.png"))

    if len(val_images) > 0 and model is not None:
        test_image = str(val_images[0])

        print(f"테스트 이미지: {Path(test_image).name}")

        # 추론
        results = model.predict(
            source=test_image,
            conf=0.5,
            save=True,
            project=str(checkpoint_dir),
            name="predictions"
        )

        # 감지된 객체 출력
        print("\n감지된 객체:")
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                if cls < len(class_names):
                    print(f"  - {class_names[cls]}: {conf:.2f}")
                else:
                    print(f"  - class_{cls}: {conf:.2f}")
    else:
        print("⚠️  테스트 이미지가 없거나 모델이 없습니다.")


def show_results(checkpoint_dir):
    """학습 결과 파일 확인"""
    print("\n" + "=" * 60)
    print("학습 결과 파일")
    print("=" * 60)

    weights_dir = checkpoint_dir / "weights"
    if weights_dir.exists():
        print(f"\n저장된 모델 위치: {weights_dir}")
        for f in weights_dir.glob("*.pt"):
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  - {f.name}: {size_mb:.1f} MB")
    else:
        print("❌ weights 폴더가 없습니다.")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("YOLO PubLayNet 학습 스크립트 (로컬 데이터셋)")
    print("=" * 60)

    # 1. GPU 및 CUDA 확인 (필수)
    if not check_gpu():
        print("\n❌ CUDA GPU가 필요합니다. 학습을 중단합니다.")
        return

    # 2. YAML 파일 확인/생성 (데이터셋 확인 전에 먼저 실행)
    setup_yaml()

    # 3. 데이터셋 확인
    if not check_dataset():
        print("\n❌ 데이터셋을 찾을 수 없습니다. 경로를 확인하세요.")
        print(f"   DATASET_PATH: {DATASET_PATH}")
        print(f"   DATA_YAML: {DATA_YAML}")
        return

    # 4. 모델 파일 확인
    if not check_model():
        print("\n❌ 모델 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        print(f"   MODEL_PATH: {MODEL_PATH}")
        return

    # 5. 체크포인트 확인
    resume, checkpoint_dir = check_checkpoint()

    # 6. 학습 실행
    train(resume, checkpoint_dir)

    # 7. 모델 평가
    model = evaluate(checkpoint_dir)

    # 8. 테스트 추론
    test_inference(model, checkpoint_dir)

    # 9. 결과 파일 확인
    show_results(checkpoint_dir)

    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)
    print(f"\n학습된 모델 위치:")
    print(f"  - Best: {checkpoint_dir}/weights/best.pt")
    print(f"  - Last: {checkpoint_dir}/weights/last.pt")


if __name__ == "__main__":
    main()
