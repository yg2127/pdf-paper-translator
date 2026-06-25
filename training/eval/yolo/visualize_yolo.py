"""
YOLO 모델 감지 결과 시각화

사용법:
    python visualize_yolo.py --input /path/to/pdf --output /path/to/output_folder
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
from ultralytics import YOLO

# 클래스별 색상 (DocLayNet)
CLASS_COLORS = {
    0: (255, 165, 0),    # Caption - 오렌지
    1: (128, 128, 128),  # Footnote - 회색
    2: (0, 255, 255),    # Formula - 시안
    3: (0, 0, 255),      # List-item - 파랑
    4: (192, 192, 192),  # Page-footer - 연회색
    5: (64, 64, 64),     # Page-header - 진회색
    6: (255, 0, 255),    # Picture - 마젠타
    7: (0, 255, 0),      # Section-header - 초록
    8: (255, 255, 0),    # Table - 노랑
    9: (255, 0, 0),      # Text - 빨강
    10: (0, 128, 0),     # Title - 진초록
}

CLASS_NAMES = {
    0: "Caption",
    1: "Footnote",
    2: "Formula",
    3: "List-item",
    4: "Page-footer",
    5: "Page-header",
    6: "Picture",
    7: "Section-header",
    8: "Table",
    9: "Text",
    10: "Title",
}


def visualize_yolo_detection(
    pdf_path: str,
    model_path: str,
    output_dir: str,
    dpi: int = 300,
    conf_threshold: float = 0.5
):
    """
    YOLO 감지 결과를 시각화

    Args:
        pdf_path: 입력 PDF 경로
        model_path: YOLO 모델 경로
        output_dir: 출력 폴더
        dpi: PDF 변환 해상도
        conf_threshold: 감지 신뢰도 임계값
    """
    print("=" * 60)
    print("YOLO 감지 결과 시각화")
    print("=" * 60)
    print(f"입력 PDF: {pdf_path}")
    print(f"모델: {model_path}")
    print(f"출력 폴더: {output_dir}")
    print(f"신뢰도 임계값: {conf_threshold}")
    print("=" * 60)

    # 출력 폴더 생성
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # PDF → 이미지 변환
    print("\n[1/3] PDF를 이미지로 변환 중...")
    images = convert_from_path(pdf_path, dpi=dpi)
    print(f"  ✓ {len(images)}개 페이지 변환 완료")

    # YOLO 모델 로드
    print("\n[2/3] YOLO 모델 로드 중...")
    model = YOLO(model_path)
    print("  ✓ 모델 로드 완료")

    # 각 페이지 감지 및 시각화
    print("\n[3/3] 감지 및 시각화 중...")

    for page_num, image in enumerate(images):
        print(f"\n  페이지 {page_num + 1}/{len(images)}:")

        # YOLO 추론
        results = model(image, conf=conf_threshold, verbose=False)

        # 이미지에 바운딩 박스 그리기
        draw_image = image.copy()
        draw = ImageDraw.Draw(draw_image)

        # 폰트 설정 (기본 폰트 사용)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        detection_count = {}

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                # 클래스 이름과 색상
                class_name = CLASS_NAMES.get(cls, f"class_{cls}")
                color = CLASS_COLORS.get(cls, (128, 128, 128))

                # 카운트
                detection_count[class_name] = detection_count.get(class_name, 0) + 1

                # 바운딩 박스 그리기
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # 라벨 그리기
                label = f"{class_name} {conf:.2f}"

                # 라벨 배경
                bbox = draw.textbbox((x1, y1 - 25), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x1, y1 - 25), label, fill=(255, 255, 255), font=font)

        # 감지 결과 출력
        print(f"    감지된 객체:")
        for class_name, count in sorted(detection_count.items()):
            print(f"      - {class_name}: {count}개")

        if not detection_count:
            print(f"      (감지된 객체 없음)")

        # 결과 이미지 저장
        output_path = output_dir / f"page_{page_num + 1:03d}_yolo.png"
        draw_image.save(output_path)
        print(f"    저장: {output_path}")

    print("\n" + "=" * 60)
    print(f"✓ 완료! 결과 이미지: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 감지 결과 시각화")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/yugeon/PDF_before/test1.pdf",
        help="입력 PDF 경로"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/yugeon/trained_model/yolov11l-doclaynet.pt",
        help="YOLO 모델 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/yugeon/PDF_after/yolo_visualization",
        help="출력 폴더"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PDF 변환 DPI"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="감지 신뢰도 임계값"
    )

    args = parser.parse_args()

    # 파일 확인
    if not Path(args.input).exists():
        print(f"❌ PDF 파일을 찾을 수 없습니다: {args.input}")
        exit(1)

    if not Path(args.model).exists():
        print(f"❌ YOLO 모델을 찾을 수 없습니다: {args.model}")
        exit(1)

    # 시각화 실행
    visualize_yolo_detection(
        args.input,
        args.model,
        args.output,
        args.dpi,
        args.conf
    )
