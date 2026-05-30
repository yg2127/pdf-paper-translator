"""
논문 번역기 설정
"""

import os
from pathlib import Path

# Repo 루트 (이 파일 기준)
REPO_ROOT = Path(__file__).resolve().parent

# 디렉토리 경로 — 환경변수로 오버라이드 가능, 없으면 repo 상대경로 기본값 사용
# 13B 번역 모델: HuggingFace 모델 ID 또는 로컬 병합 모델 디렉토리.
# 환경변수 미설정 시 MarianMT 백업 모델로 fallback (README 참조).
MODEL_13B_DIR = os.getenv("PAPER_TRANS_MODEL_DIR", "Helsinki-NLP/opus-mt-en-ko")
yolo_Models_DIR = os.getenv(
    "PAPER_TRANS_YOLO_PATH", str(REPO_ROOT / "models" / "yolov11l-doclaynet.pt")
)
FONTS_DIR = os.getenv("PAPER_TRANS_FONT", str(REPO_ROOT / "fonts" / "BMHANNAPro.ttf"))
BOLD_FONTS_DIR = os.getenv(
    "PAPER_TRANS_BOLD_FONT", str(REPO_ROOT / "fonts" / "MaruBuri-Bold.ttf")
)
INPUT_DIR = os.getenv("PAPER_TRANS_INPUT_DIR", str(REPO_ROOT / "PDF_before"))
OUTPUT_DIR = os.getenv("PAPER_TRANS_OUTPUT_DIR", str(REPO_ROOT / "PDF_after"))
TSV_DIR = os.getenv("PAPER_TRANS_TSV", str(REPO_ROOT / "tsv" / "base_vocab.tsv"))

# 기본 설정
DEFAULT_CONFIG = {
    # 입출력 경로 (여기서 수정!)
    "input_pdf": os.getenv(
        "PAPER_TRANS_INPUT_PDF", str(Path(INPUT_DIR) / "test1.pdf")
    ),  # 입력 PDF 경로
    "output_pdf": None,  # 출력 PDF 경로 (None이면 자동 생성)
    # PDF 변환 설정
    "dpi": 300,
    # YOLO 설정
    "yolo_model_path": str(yolo_Models_DIR),
    "yolo_confidence": 0.5,
    "yolo_iou": 0.45,
    # 번역 설정
    "translation_model": str(MODEL_13B_DIR),  # 13B 병합 모델 (기본)
    "translation_max_length": 512,
    "translation_batch_size": 128,
    "translation_use_4bit": False,  # 4-bit 양자화 사용 (13B 모델용) - 속도 향상을 위해 비활성화
    "translation_model_type": None,  # 모델 타입 ('marian', 'causal', None=자동 감지)
    "terminology_dict_path": str(TSV_DIR),  # 베이스 용어 사전
    # PDF 생성 설정
    "korean_font_path": str(FONTS_DIR),
    "korean_bold_font_path": str(
        BOLD_FONTS_DIR
    ),  # 볼드 TTF 경로 (없으면 일반 폰트 재사용)
    "default_font_size": 25,
    "pdf_draw_background": True,  # 원본 페이지 이미지를 배경으로 그릴지 여부
    "pdf_text_bg_margin": 4,  # 텍스트 덮는 흰 배경 여백(px, 이미지 좌표 기준)
    "generate_no_background_variant": True,  # 배경 없이 흰 바탕+그림만 두 번째 PDF 생성
    "no_background_suffix": "_nobg",  # 배경 없는 PDF 파일명 접미사 (확장자 자동 추가)
    # 좌표 변환 설정
    "overlap_threshold": 0.3,  # IoU 임계값 (낮을수록 겹침으로 간주해 번역을 건너뜀)
    # 출력 설정
    "output_dir": str(OUTPUT_DIR),
    # 용어 사전 설정 (영어 용어 유지 + 볼드 표시)
    "term_bold": True,  # 용어 볼드체 처리 ON
    "extract_title_terms": True,  # Title에서 대문자 용어 추출 ON/OFF
    "extract_fte_terms": True,  # FTE(Figure/Table/Equation)에서 용어 추출
    "save_extracted_terms": True,  # 추출된 용어 파일로 저장
    "terms_output_path": os.path.join(
        OUTPUT_DIR, "extracted_terms.tsv"
    ),  # 용어 저장 경로
}

# 클래스 매핑 (YOLO 모델용 - DocLayNet)
CLASS_NAMES = {
    0: "Caption",  # 캡션
    1: "Footnote",  # 각주
    2: "Formula",  # 수식
    3: "List-item",  # 리스트 항목
    4: "Page-footer",  # 페이지 푸터
    5: "Page-header",  # 페이지 헤더
    6: "Picture",  # 그림
    7: "Section-header",  # 섹션 헤더
    8: "Table",  # 표
    9: "Text",  # 본문 텍스트
    10: "Title",  # 제목
}

# 용어 추출 대상 클래스 (DocLayNet)
TERM_EXTRACT_CLASSES = ["Title", "Section-header"]

# 번역 제외 클래스 (이 클래스에 포함된 텍스트는 번역하지 않음)
SKIP_TRANSLATION_CLASSES = [
    "Picture",
    "Table",
    "Formula",
    "Title",
    "Caption",
    "Section-header",
]


def get_config(overrides: dict = None) -> dict:
    """설정 가져오기 (오버라이드 적용)"""
    config = DEFAULT_CONFIG.copy()

    if overrides:
        config.update(overrides)

    return config


def ensure_directories():
    """필요한 디렉토리 생성"""
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
