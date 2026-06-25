# 모델 성능 평가 가이드

## 1. YOLO 모델 평가 (mAP)

### 필요 사항
- YOLO 모델 (.pt 파일)
- Validation 데이터셋 (data.yaml 포함)

### data.yaml 예시
```yaml
# Dataset 경로
path: /home/yugeon/dataset/publaynet
train: images/train
val: images/val

# Classes
nc: 6
names:
  0: text
  1: title
  2: list
  3: table
  4: figure
  5: formula
```

### 실행 방법
```bash
python evaluate_yolo.py --data /path/to/data.yaml
```

### 옵션
- `--model`: YOLO 모델 경로 (기본값: config.py의 yolo_Models_DIR)
- `--data`: 데이터셋 설정 파일 (필수)

### 출력 결과
- **mAP50**: IoU=0.5일 때 mAP
- **mAP50-95**: IoU=0.5~0.95 평균 mAP
- **Precision**: 평균 정밀도
- **Recall**: 평균 재현율
- **클래스별 AP**: 각 클래스의 Average Precision

결과 플롯은 `runs/detect/val/`에 저장됩니다.

---

## 2. 번역 모델 평가 (BLEU + BERTScore)

### 단계 1: 예측 생성

먼저 13B 모델로 번역 예측을 생성해야 합니다.

#### 입력 파일 준비
`source.txt` (영어 문장, 한 줄에 하나씩):
```
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Natural language processing enables computers to understand human language.
```

#### 예측 생성
```bash
python generate_translation_predictions.py \
    --input source.txt \
    --output predictions.txt \
    --batch_size 8
```

#### 옵션
- `--model`: 번역 모델 경로 (기본값: 13B_merged_fp16)
- `--input`: 영어 문장 파일
- `--output`: 번역 결과 저장 경로
- `--batch_size`: 배치 크기 (기본값: 8)

#### 출력
`predictions.txt` (번역된 한국어):
```
빠른 갈색 여우가 게으른 개를 뛰어넘습니다.
기계 학습은 인공지능의 하위 집합입니다.
자연어 처리는 컴퓨터가 인간의 언어를 이해할 수 있게 합니다.
```

### 단계 2: BLEU/BERTScore 평가

#### 정답 파일 준비
`references.txt` (정답 한국어, 한 줄에 하나씩):
```
빠른 갈색 여우가 게으른 개를 뛰어넘는다.
머신러닝은 인공지능의 부분집합이다.
자연어 처리는 컴퓨터가 사람의 언어를 이해할 수 있게 한다.
```

#### 평가 실행
```bash
python evaluate_translation.py \
    --predictions predictions.txt \
    --references references.txt \
    --lang ko
```

#### 옵션
- `--predictions`: 모델 예측 파일 (필수)
- `--references`: 정답 파일 (필수)
- `--lang`: 언어 코드 (기본값: ko)

#### 출력 결과
- **BLEU Score**: 0~100 (높을수록 좋음)
  - 30+ : 이해 가능한 번역
  - 50+ : 좋은 번역
  - 60+ : 매우 좋은 번역

- **BERTScore**: 0~1 (높을수록 좋음)
  - **Precision**: 예측이 정답과 얼마나 일치하는가
  - **Recall**: 정답이 예측에 얼마나 포함되는가
  - **F1**: Precision과 Recall의 조화 평균

---

## 3. 전체 평가 워크플로우

### YOLO 평가
```bash
# 1. data.yaml 준비
# 2. YOLO 평가 실행
python evaluate_yolo.py --data /path/to/data.yaml
```

### 번역 모델 평가
```bash
# 1. 테스트 데이터 준비 (source.txt, references.txt)
# 2. 예측 생성
python generate_translation_predictions.py \
    --input test_source.txt \
    --output test_predictions.txt

# 3. BLEU/BERTScore 평가
python evaluate_translation.py \
    --predictions test_predictions.txt \
    --references test_references.txt
```

---

## 4. 필요한 패키지 설치

```bash
# YOLO 평가용
pip install ultralytics

# 번역 평가용
pip install sacrebleu bert-score transformers torch
```

---

## 5. 팁

### YOLO mAP 향상 방법
- 더 많은 학습 데이터
- 데이터 증강 (augmentation)
- 더 긴 학습 (epochs 증가)
- 하이퍼파라미터 튜닝

### 번역 BLEU/BERTScore 향상 방법
- 더 많은 학습 데이터 (병렬 말뭉치)
- 더 긴 학습 (steps 증가)
- 용어 사전 활용
- 도메인 특화 파인튜닝
