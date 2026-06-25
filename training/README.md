# 모델 학습 (training/)

파이프라인이 쓰는 두 모델을 직접 학습시킨 코드와 평가다.

- **YOLOv11 레이아웃 검출** — 논문 PDF에서 본문/제목/표/그림/수식 영역 찾기
- **TowerInstruct-13B 번역** — 학술 문장 영어→한국어 (QLoRA)

전처리·학습은 클라우드 GPU(Google Cloud L4 / Colab A100 / RunPod A100 80GB)에서 돌렸고, 데이터와 체크포인트는 GCS 버킷으로 옮겨가며 작업했다. 용량 때문에 데이터셋·가중치(`.pt`)는 저장소에 넣지 않았다.

## 1. YOLOv11 레이아웃 검출

### 데이터

PubLayNet(IBM Research, 약 36만 장, 5클래스)을 주로 쓰고, PubLayNet에 없는 수식 영역은 DocLayNet v1.1에서 Formula 클래스만 뽑아 합쳤다 — 최종 6클래스(text/title/list/table/figure/formula).

- `yolo_data_preprocessing/convert_publaynet_parquet_to_yolo.py` — PubLayNet(parquet)을 YOLO 라벨로 변환, GCS에 바로 저장
- `yolo_data_preprocessing/extract_doclaynet_formula.py` — DocLayNet에서 수식만 추출 (`category_id==3`, 클래스 ID 5)
- `yolo_data_preprocessing/merge_publaynet_formula.py` — 둘을 합쳐 6클래스 데이터셋으로

### 학습

`yolo_train_code/train_yolo.py` — 베이스는 `yolov11l-doclaynet`([hantian/yolo-doclaynet](https://huggingface.co/hantian/yolo-doclaynet)), 이걸 병합셋으로 파인튜닝.

```
Epochs 30 · Batch 32 · ImgSize 640 · SGD(lr 0.01, momentum 0.937) · EarlyStop(patience 20)
문서라 증강은 최소화 — mosaic/mixup 0, flip off
```

### 결과 (PubLayNet validation)

`eval/yolo/evaluate_yolo.py`로 측정한 값이다.

| mAP@0.5    | mAP@0.5:0.95 | Precision | Recall |
| ---------- | ------------ | --------- | ------ |
| **0.9606** | 0.9324       | 0.9830    | 0.9304 |

| 클래스별 AP@0.5 | text   | title  | list   | table  | figure |
| --------------- | ------ | ------ | ------ | ------ | ------ |
|                 | 0.9576 | 0.9664 | 0.9391 | 0.9753 | 0.9645 |

<img src="eval/result/yolo/val/BoxPR_curve.png" alt="YOLO PR Curve" width="440">

근데 이 파인튜닝 모델은 정작 파이프라인엔 못 썼다. 수식을 병합하는 과정에서 다른 클래스 라벨이 날아간 채 학습돼, 실제 페이지에선 수식만 잡았다. 그래서 파이프라인은 `yolov11l-doclaynet` 사전학습 백본(11클래스)을 그대로 쓴다 — `config.py`의 `CLASS_NAMES`가 DocLayNet 기준인 것도 이 때문이다. 위 mAP는 학습 실험에서 나온 값이다. 검증 산출물 원본은 [`eval/result/yolo/val/`](eval/result/yolo/val/)에 있다.

## 2. TowerInstruct-13B 번역

### 베이스 모델

[Unbabel/TowerInstruct-13B-v0.1](https://huggingface.co/Unbabel/TowerInstruct-13B-v0.1) — LLaMA-2 13B 기반 번역 특화 모델. 한국어 포함 10개 언어를 지원하고 ChatML 프롬프트를 쓴다.

### 학습 데이터 — AI-Hub 한영 병렬 코퍼스 4종

`13B_data_preprocessing/dataloader.py`가 4종을 합쳐(en→ko, Tower ChatML, train/val 95:5) 학습 형식으로 바꾼다.

| #   | 데이터셋 (AI-Hub)                         | 구축 | 규모         | 도메인                              |
| --- | ----------------------------------------- | ---- | ------------ | ----------------------------------- |
| 1   | 국제 학술대회용 전문분야 한영/영한 통번역 | 2023 | 약 100만+ 쌍 | 의학·IT·전기전자·기계·화학·물리수학 |
| 2   | 기술과학 분야 한-영 번역 병렬 말뭉치      | 2021 | 약 150만 쌍  | 특허·기술매뉴얼·연구보고서·IT       |
| 3   | 한국어-영어 번역 말뭉치(기술과학)         | 2020 | 약 160만 쌍  | 과학기술 문헌·뉴스·정부문서         |
| 4   | 한국어-영어 번역 말뭉치(사회과학)         | 2020 | 약 160만 쌍  | 경제·사회·정치·문화·교육·법률       |

합쳐서 약 5백만+ 문장 쌍. 모두 AI-Hub(한국지능정보사회진흥원) 제공.

### 학습 (QLoRA)

`13B_train_code/train_tower_13b.py`:

```
4-bit nf4 + double-quant, compute bf16
LoRA  r 64 · alpha 128 · dropout 0.05 · target q/k/v/o/gate/up/down_proj
학습  1 epoch · batch 36 · lr 2e-4 (cosine) · paged_adamw_8bit · max_len 512
```

7B 모델 LoRA 파인튜닝(`train_7b_lora.py`, L4 23GB)도 따로 돌렸다.

### 결과

_Attention Is All You Need_ 본문 24문장으로 평가했다(정답은 Papago로 옮긴 뒤 직접 검수).

| BLEU      | BERTScore P | BERTScore R | BERTScore F1 |
| --------- | ----------- | ----------- | ------------ |
| **59.62** | 0.8935      | 0.8867      | **0.8900**   |

좁은 도메인 24문장이라 BLEU가 높게 나온다. 절대 수치보다, 학술 문체에서 용어와 구조를 얼마나 살리는지를 보는 쪽이 맞다. 측정 절차는 [`eval/EVALUATION_GUIDE.md`](eval/EVALUATION_GUIDE.md), 원문/정답/예측은 [`eval/13B/`](eval/13B/)에 있다.

## 3. 환경

```bash
conda env create -f environment.yml
# 또는 PyTorch를 CUDA 버전에 맞춰 따로 깐 뒤
pip install -r requirements.txt
```

`python 3.10` · `pytorch(CUDA 12.4)` · `transformers` · `datasets` · `peft` · `bitsandbytes` · `accelerate` · `ultralytics` · `sacrebleu` · `bert_score`.

학습 스크립트의 데이터·모델 경로는 작업 당시 클라우드(GCS 마운트 `~/gcs_bucket` 등) 기준이라, 돌리기 전에 스크립트 상단의 경로 상수를 환경에 맞게 고쳐야 한다.

## 디렉토리

```
training/
├── environment.yml · requirements.txt
├── Download_yolo/        DocLayNet 사전학습 YOLOv11 다운로드
├── yolo_data_preprocessing/   PubLayNet 변환 · 수식 추출 · 병합
├── yolo_train_code/      YOLOv11 학습
├── Download_13B/         TowerInstruct-13B → GCS
├── 13B_data_preprocessing/    AI-Hub 4종 통합 로더
├── 13B_train_code/       TowerInstruct-13B QLoRA 학습
├── train_7b_lora.py      7B LoRA 학습
└── eval/                 YOLO mAP · 번역 BLEU/BERTScore + 검증 산출물
```
