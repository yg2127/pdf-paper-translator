# 논문 번역기 파이프라인 (Paper Translator Pipeline)

학술 논문 PDF를 한국어로 자동 번역하는 AI 기반 파이프라인입니다.

## 주요 기능

- PDF 논문을 고품질 한국어로 자동 번역
- 그림, 표, 수식 영역 자동 감지 및 보존
- 학술 용어 자동 추출 및 원문 유지
- OCR 기반 이미지 내 텍스트 인식
- 원본 레이아웃을 유지한 한국어 PDF 생성

## 파이프라인 아키텍처

```
PDF 입력
   ↓
[1] PDF → 이미지 변환 (pdf2image, DPI 300)
   ↓
[2] 텍스트 및 위치 정보 추출 (PyMuPDF)
   ↓
[3] 객체 감지 (YOLOv11: 그림/표/수식/캡션/제목)
   ↓
[4] 이미지 영역 추출 (Pillow)
   ↓
[5] 이미지 내 텍스트 인식 (EasyOCR)
   ↓
[6] 학술 용어 추출 (FTE/Title에서 자동 추출)
   ↓
[7] 영어→한국어 번역 (파인튜닝된 13B 모델 또는 MarianMT)
   ↓
[8] 한국어 PDF 생성 (ReportLab)
   ↓
한국어 PDF 출력
```

## 설치

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 필수 리소스 준비

#### YOLO 모델
```bash
mkdir -p models
# YOLOv11 논문 객체 감지 모델 다운로드
# models/yolov11_paper.pt
```

#### 한글 폰트
```bash
mkdir -p fonts
# NanumGothic.ttf 다운로드 및 배치
# fonts/NanumGothic.ttf
```

## 사용법

### 기본 실행

```bash
python main.py input.pdf
```

### 출력 경로 지정

```bash
python main.py input.pdf -o output_ko.pdf
```

### 고급 옵션

```bash
python main.py input.pdf \
  --dpi 300 \
  --yolo-model models/yolov11_paper.pt \
  --font fonts/NanumGothic.ttf \
  --no-bold \              # 용어 볼드체 처리 비활성화
  --no-title-terms \       # Title에서 용어 추출 비활성화
  --no-fte-terms \         # FTE에서 용어 추출 비활성화
  --no-save-terms          # 용어 파일 저장 비활성화
```

## 프로젝트 구조

```
trans_pipline_1208/
├── main.py                          # 메인 파이프라인
├── config.py                        # 설정 파일
├── requirements.txt                 # 의존성 목록
├── modules/                         # 모듈 패키지
│   ├── __init__.py
│   ├── pdf_converter.py            # PDF → 이미지 변환
│   ├── text_extractor.py           # 텍스트 추출 (PyMuPDF)
│   ├── yolo_detector.py            # YOLO 객체 감지
│   ├── image_extractor.py          # 이미지 영역 추출
│   ├── ocr_processor.py            # OCR 처리 (EasyOCR)
│   ├── term_extractor.py           # 학술 용어 추출
│   ├── translator.py               # 번역 엔진
│   ├── coordinate_transformer.py   # 좌표계 변환
│   └── pdf_generator.py            # PDF 생성 (ReportLab)
├── models/                          # YOLO 모델
│   └── yolov11_paper.pt
├── fonts/                           # 폰트 파일
│   └── NanumGothic.ttf
└── output/                          # 출력 디렉토리
    ├── *_ko.pdf                    # 번역된 PDF
    └── *_terms.tsv                 # 추출된 용어 사전
```

## 주요 모듈 설명

### 1. PDF Converter (`pdf_converter.py`)
- PDF를 고해상도 이미지로 변환
- 기본 DPI: 300

### 2. Text Extractor (`text_extractor.py`)
- PyMuPDF를 사용한 텍스트 및 바운딩 박스 추출
- 페이지별 텍스트 블록 정보 제공

### 3. YOLO Detector (`yolo_detector.py`)
- YOLOv11 기반 객체 감지
- 감지 클래스:
  - `figure`: 그림
  - `table`: 표
  - `equation`: 수식
  - `caption`: 캡션
  - `title`: 제목
  - `header`, `footer`, `page_number`

### 4. Term Extractor (`term_extractor.py`)
FTE(Figure/Table/Equation) 및 Title에서 학술 용어 자동 추출

**추출 규칙:**
- 대문자 약어 (BERT, CNN, GPT)
- 그리스 문자 (α, β, γ)
- 숫자 포함 단어 (F1, ResNet50)
- CamelCase (PyTorch, TensorFlow)
- 하이픈 기술 용어 (F1-score, self-attention)
- Figure/Table/Equation 참조

**제외 규칙:**
- 일반 영단어 (the, is, have 등)
- 소문자만 있는 단어

### 5. Translator (`translator.py`)
**지원 모델:**
- **13B 커스텀 모델** (기본) - 파인튜닝된 13B 번역 모델
  - ChatML 프롬프트 형식
  - 4-bit 양자화 지원
  - 자동 모델 타입 감지
- **Helsinki-NLP/opus-mt-en-ko** (백업) - MarianMT 모델

**주요 기능:**
- 학술 용어 보호 기능
- 수식, 참조 자동 보호
- 배치 번역 지원 (MarianMT)
- 용어 볼드체 처리 옵션

### 6. OCR Processor (`ocr_processor.py`)
- EasyOCR 기반 텍스트 인식
- 다국어 지원

### 7. PDF Generator (`pdf_generator.py`)
- ReportLab 기반 한국어 PDF 생성
- 원본 레이아웃 유지
- 한글 폰트 지원

## 설정 (config.py)

```python
DEFAULT_CONFIG = {
    # PDF 변환
    "dpi": 300,

    # YOLO
    "yolo_model_path": "models/yolov11_paper.pt",
    "yolo_confidence": 0.5,
    "yolo_iou": 0.45,

    # OCR
    "ocr_languages": ["en"],
    "ocr_gpu": True,

    # 번역
    "translation_model": "/home/yugeon/trained_model/13B_merged",  # 13B 모델 (기본)
    # "translation_model": "Helsinki-NLP/opus-mt-en-ko",  # MarianMT (백업)
    "translation_max_length": 512,
    "translation_batch_size": 8,
    "translation_use_4bit": True,       # 4-bit 양자화 (13B 모델용)
    "translation_model_type": None,     # 모델 타입 자동 감지

    # 용어 사전
    "term_bold": True,                  # 용어 볼드체 처리
    "extract_title_terms": True,        # Title에서 용어 추출
    "extract_fte_terms": True,          # FTE에서 용어 추출
    "save_extracted_terms": True,       # 용어 파일 저장

    # PDF 생성
    "korean_font_path": "fonts/NanumGothic.ttf",
    "default_font_size": 10,

    # 좌표 변환
    "overlap_threshold": 0.3,
}
```

## 용어 추출 예시

입력 텍스트:
```
BERT achieves 0.95 F1-score on the dataset
ResNet50 vs VGG16 comparison
α = 0.01, β = 0.99
Attention Is All You Need: Transformer Architecture
```

추출된 용어:
```
✓ 유지 | BERT          | figure    | 대문자 약어
✓ 유지 | F1-score      | figure    | 하이픈 기술용어
✓ 유지 | ResNet50      | table     | 모델명/버전
✓ 유지 | VGG16         | table     | 모델명/버전
✓ 유지 | α             | equation  | 그리스 문자
✓ 유지 | β             | equation  | 그리스 문자
✓ 유지 | Attention     | title     | Title 대문자
✓ 유지 | All           | title     | Title 대문자
✓ 유지 | You           | title     | Title 대문자
✓ 유지 | Need          | title     | Title 대문자
✓ 유지 | Transformer   | title     | Title 대문자
✓ 유지 | Architecture  | title     | Title 대문자
```

## 출력 결과

### 1. 번역된 PDF
- 파일명: `{원본파일명}_ko.pdf`
- 위치: `output/` 디렉토리

### 2. 추출된 용어 사전
- 파일명: `{원본파일명}_terms.tsv`
- 형식: TSV (탭 구분)
- 컬럼:
  - 용어
  - 출처 (figure/table/equation/title)
  - 페이지
  - 빈도
  - 유지여부 (O/X)
  - 이유

예시:
```
용어	출처	페이지	빈도	유지여부	이유
BERT	figure	1	5	O	대문자 약어
ResNet50	table	2	3	O	모델명/버전
α	equation	3	2	O	그리스 문자
```

## 번역 워크플로우

1. **텍스트 블록 추출**: PyMuPDF로 텍스트 및 위치 정보 추출
2. **객체 영역 감지**: YOLO로 그림/표/수식 영역 감지
3. **겹침 확인**: 텍스트가 YOLO 감지 영역과 겹치는지 확인
   - 겹치면: 번역하지 않음 (그림/표/수식 내부)
   - 겹치지 않으면: 번역 수행
4. **용어 보호**: 학술 용어는 원문 유지 + 볼드체 처리
5. **OCR 텍스트**: 이미지 내 텍스트도 번역

## 커스텀 번역 모델 사용

### 13B 모델 설정 (기본)

현재 파이프라인은 파인튜닝된 13B 모델을 기본으로 사용합니다.

**1. 모델 병합 (처음 한 번만 실행)**
```bash
# LoRA adapter와 베이스 모델을 병합
python3 /home/yugeon/merge_and_save.py
```

**2. 파이프라인 실행**
```bash
# config.py에 이미 13B 모델 경로가 설정되어 있음
python main.py input.pdf
```

### 다른 모델로 변경

**MarianMT 모델로 변경:**
```python
# config.py에서
"translation_model": "Helsinki-NLP/opus-mt-en-ko",
```

**다른 커스텀 모델 사용:**
```python
# config.py에서
"translation_model": "/path/to/your/model",
"translation_use_4bit": True,  # 대용량 모델은 True
"translation_model_type": "causal",  # 또는 "marian"
```

**코드에서 직접 설정:**
```python
from modules.translator import Translator

translator = Translator(
    model_name="/path/to/your/model",
    device="cuda",
    max_length=512,
    term_bold=True,
    use_4bit=True,
    model_type="causal"  # 'marian' 또는 'causal'
)
```

## 의존성

주요 라이브러리:
- `pdf2image`: PDF → 이미지 변환
- `PyMuPDF`: 텍스트 추출
- `Pillow`: 이미지 처리
- `ultralytics`: YOLOv11 객체 감지
- `easyocr`: OCR
- `transformers`: 번역 모델
- `torch`: 딥러닝 프레임워크
- `reportlab`: PDF 생성
- `peft`: LoRA 어댑터 지원
- `bitsandbytes`: 4-bit 양자화 지원

전체 목록: `requirements.txt` 참조

## 성능 최적화

### GPU 사용
```python
config = {
    "ocr_gpu": True,  # OCR GPU 가속
    # 번역 모델도 자동으로 CUDA 감지
}
```

### 배치 처리
```python
config = {
    "translation_batch_size": 16,  # 배치 크기 증가 (GPU 메모리에 따라 조정)
}
```

### 양자화 모델
```python
# 4-bit 양자화 모델 사용 (메모리 절약)
# requirements.txt에 이미 포함:
# - bitsandbytes
# - accelerate
```

## 알려진 제한사항

1. **복잡한 레이아웃**: 매우 복잡한 다단 레이아웃은 정확도가 떨어질 수 있음
2. **수식**: LaTeX 수식은 번역하지 않고 원문 유지
3. **폰트**: 한글 폰트는 별도로 준비 필요
4. **메모리**: 대용량 PDF(100페이지 이상)는 메모리 부족 가능

## 라이선스

MIT License

## 작성자

- 작성일: 2024-12-08
- 프로젝트명: trans_pipline_1208

## 참고

- YOLOv11: https://github.com/ultralytics/ultralytics
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Helsinki-NLP: https://huggingface.co/Helsinki-NLP
- PyMuPDF: https://pymupdf.readthedocs.io/
- ReportLab: https://www.reportlab.com/
