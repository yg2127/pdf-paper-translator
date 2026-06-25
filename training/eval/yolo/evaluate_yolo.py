"""
YOLO 모델 성능 평가 (mAP 계산)

사용법:
    python evaluate_yolo.py --data /path/to/data.yaml
"""
import argparse
from ultralytics import YOLO
from pathlib import Path

def evaluate_yolo(model_path: str, data_yaml: str):
    """
    YOLO 모델 mAP 평가

    Args:
        model_path: YOLO 모델 경로 (.pt 파일)
        data_yaml: 데이터셋 설정 파일 (data.yaml)
    """
    print("=" * 80)
    print("YOLO 모델 성능 평가 (mAP)")
    print("=" * 80)
    print(f"모델: {model_path}")
    print(f"데이터셋: {data_yaml}")
    print("=" * 80)

    # 모델 로드
    print("\n[1/2] 모델 로드 중...")
    model = YOLO(model_path)
    print("✓ 모델 로드 완료")

    # 검증 실행 (mAP 계산)
    print("\n[2/2] mAP 계산 중...")
    print("⏳ Validation 데이터셋으로 평가 진행...")

    results = model.val(
        data=data_yaml,
        imgsz=640,
        conf=0.25,      # Confidence threshold
        iou=0.45,       # NMS IoU threshold
        verbose=True,   # 상세 출력
        plots=True,     # 결과 플롯 저장
    )

    # 결과 출력
    print("\n" + "=" * 80)
    print("평가 결과")
    print("=" * 80)
    print(f"mAP50     : {results.box.map50:.4f}")      # mAP at IoU=0.50
    print(f"mAP50-95  : {results.box.map:.4f}")        # mAP at IoU=0.50:0.95
    print(f"Precision : {results.box.mp:.4f}")         # Mean precision
    print(f"Recall    : {results.box.mr:.4f}")         # Mean recall
    print("=" * 80)

    # 클래스별 AP
    print("\n클래스별 AP (IoU=0.50):")
    print("-" * 80)
    class_names = results.names  # {0: 'text', 1: 'title', ...}
    for class_id, ap in enumerate(results.box.ap50):
        class_name = class_names.get(class_id, f"class_{class_id}")
        print(f"  {class_name:15s}: {ap:.4f}")
    print("-" * 80)

    # 결과 저장 경로
    print(f"\n✓ 결과 플롯 저장: runs/detect/val/")
    print("  - confusion_matrix.png")
    print("  - F1_curve.png")
    print("  - PR_curve.png")
    print("  - P_curve.png")
    print("  - R_curve.png")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 모델 mAP 평가")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/yugeon/trained_model/yolo/yolo/yolo11m_publaynet/weights/best.pt",
        help="YOLO 모델 경로 (.pt 파일)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="데이터셋 설정 파일 (data.yaml)"
    )

    args = parser.parse_args()

    # 파일 존재 확인
    if not Path(args.model).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        exit(1)

    if not Path(args.data).exists():
        print(f"❌ 데이터셋 설정 파일을 찾을 수 없습니다: {args.data}")
        exit(1)

    # 평가 실행
    evaluate_yolo(args.model, args.data)
