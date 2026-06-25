"""
7B 모델 LoRA 파인튜닝 스크립트
- VRAM: 23GB (L4)
- 학습 시간: 12시간 이내
- 데이터셋: 114만 샘플
- 목표: checkpoint-2000
"""
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import time
from datetime import datetime

# ============================================================================
# 경로 설정
# ============================================================================
BASE_MODEL_PATH = "/home/yugeon/trained_model/7B_origin"
DATASET_PATH = "/home/yugeon/1207_dataset/en2ko/en2ko"
OUTPUT_DIR = "/home/yugeon/trained_model/7B_lora"

# ============================================================================
# 학습 설정 (L4 23GB VRAM 최적화 - 18GB 사용)
# ============================================================================
TRAINING_CONFIG = {
    # 배치 설정 (VRAM 최대 활용)
    "per_device_train_batch_size": 16,        # GPU당 배치 크기 (4→8 증가)
    "gradient_accumulation_steps": 4,        # 누적 스텝 (effective_batch=32)

    # 학습 스텝
    "max_steps": 10000,                      # 10,000 스텝
    "save_steps": 5000,                      # 5000 스텝마다 저장
    "logging_steps": 50,                     # 50 스텝마다 로그

    # 학습률
    "learning_rate": 2e-4,                   # LoRA 기본값
    "lr_scheduler_type": "cosine",           # Cosine 스케줄러
    "warmup_steps": 100,                     # 워밍업

    # 최적화
    "optim": "adamw_torch",                  # AdamW 옵티마이저
    "weight_decay": 0.01,                    # 가중치 감쇠
    "max_grad_norm": 1.0,                    # 그래디언트 클리핑

    # 메모리 최적화
    "gradient_checkpointing": True,          # 메모리 절약
    "bf16": True,                            # bfloat16 (L4 지원)
    "dataloader_num_workers": 4,             # 데이터 로더 워커

    # 기타
    "seed": 42,
    "report_to": "none",                     # 로깅 비활성화
}

# LoRA 설정
LORA_CONFIG = {
    "r": 64,                                 # LoRA rank (32→64 증가, 성능 향상)
    "lora_alpha": 128,                       # LoRA alpha (r*2)
    "lora_dropout": 0.05,                    # Dropout
    "target_modules": [                      # LoRA 적용할 레이어
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

print("=" * 80)
print("7B LoRA 파인튜닝 시작")
print("=" * 80)
print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"베이스 모델: {BASE_MODEL_PATH}")
print(f"데이터셋: {DATASET_PATH}")
print(f"출력 경로: {OUTPUT_DIR}")
print(f"목표 스텝: {TRAINING_CONFIG['max_steps']}")
print(f"Effective Batch: {TRAINING_CONFIG['per_device_train_batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
print("=" * 80)

start_time = time.time()

# ============================================================================
# 1. 데이터셋 로드
# ============================================================================
print("\n[1/6] 데이터셋 로드 중...")
dataset = load_from_disk(DATASET_PATH)
print(f"✓ Train: {len(dataset['train'])} 샘플")
if 'validation' in dataset:
    print(f"✓ Validation: {len(dataset['validation'])} 샘플")

# ============================================================================
# 2. 토크나이저 로드
# ============================================================================
print("\n[2/6] 토크나이저 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✓ 토크나이저 로드 완료")

# ============================================================================
# 3. 모델 로드 (8-bit 양자화)
# ============================================================================
print("\n[3/6] 모델 로드 중 (8-bit 양자화)...")
print("⏳ VRAM ~7GB 사용 예상")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 8-bit 학습 준비
model = prepare_model_for_kbit_training(model)
print("✓ 모델 로드 완료")

# ============================================================================
# 4. LoRA 설정
# ============================================================================
print("\n[4/6] LoRA 설정 중...")
lora_config = LoraConfig(**LORA_CONFIG)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print("✓ LoRA 설정 완료")

# ============================================================================
# 5. 데이터 전처리
# ============================================================================
print("\n[5/6] 데이터 전처리 중...")

def preprocess_function(examples):
    """ChatML 포맷으로 변환"""
    prompts = []
    for source, target in zip(examples['source'], examples['target']):
        # ChatML 템플릿
        prompt = f"<|im_start|>user\nTranslate the following text from English into Korean.\nEnglish: {source}\nKorean:<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"
        prompts.append(prompt)

    # 토크나이징
    model_inputs = tokenizer(
        prompts,
        max_length=512,
        truncation=True,
        padding=False,
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

# 전처리 적용
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset['train'].column_names,
    desc="Tokenizing dataset"
)
print("✓ 데이터 전처리 완료")

# Data Collator (labels 패딩 지원)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100
)

# ============================================================================
# 6. 학습 시작
# ============================================================================
print("\n[6/6] 학습 시작...")
print(f"⏳ 예상 시간: 10-12시간")
print(f"⏳ 1 스텝당 약 15-20초 예상")
print("=" * 80)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    **TRAINING_CONFIG
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    data_collator=data_collator,
)

# 학습 실행
trainer.train()

# ============================================================================
# 저장
# ============================================================================
print("\n모델 저장 중...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✓ 모델 저장 완료: {OUTPUT_DIR}")

# ============================================================================
# 완료
# ============================================================================
elapsed = time.time() - start_time
hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)

print("\n" + "=" * 80)
print("학습 완료!")
print("=" * 80)
print(f"완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"총 소요 시간: {hours}시간 {minutes}분")
print(f"저장 위치: {OUTPUT_DIR}")
print("\n다음 단계:")
print("1. LoRA adapter와 베이스 모델 병합")
print("2. GCS에 업로드")
print("3. 번역 파이프라인에 적용")
print("=" * 80)
