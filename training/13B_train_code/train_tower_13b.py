"""
Tower-13B QLoRA 파인튜닝 스크립트

================================================================================
경로 설정 위치 안내 (라인 ~25-35):
--------------------------------------------------------------------------------
1. LOCAL_DATA_DIR: 로컬 데이터셋 경로 (Arrow 형식)
   - 현재: "/content/drive/MyDrive/datasets/en2ko"

2. LOCAL_MODEL_DIR: 베이스 모델(Tower-13B) 경로 (로컬)
   - 현재: "1207_dataset/models/13B"

3. OUTPUT_DIR: 학습된 모델 저장 경로
   - 현재: "./tower_finetuned_en2ko"
================================================================================
"""

import os
import subprocess
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# 경로 설정 (여기서 수정하세요!)
# ============================================================================
# ★★★ 데이터셋 경로 (로컬) - 여기서 수정! ★★★
LOCAL_DATA_DIR = "en2ko"  # 로컬 데이터셋 경로

# ★★★ 베이스 모델 경로 (로컬) - 여기서 수정! ★★★
LOCAL_MODEL_DIR = "/root/models/13B"  # 로컬 모델 경로

# 모델 저장 경로
OUTPUT_DIR = "./tower_finetuned_en2ko"

# HuggingFace 모델 이름 (로컬에 모델이 없을 때 다운로드용)
HF_MODEL_NAME = "Unbabel/TowerInstruct-13B-v0.1"


# ============================================================================
# 설정 클래스들
# ============================================================================
@dataclass
class ModelConfig:
    """모델 설정"""
    model_name: str = LOCAL_MODEL_DIR
    max_length: int = 512
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA 설정"""
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainConfig:
    """학습 설정"""
    output_dir: str = OUTPUT_DIR
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 36  # A100 VRAM 여유 있음
    per_device_eval_batch_size: int = 36
    gradient_accumulation_steps: int = 1  # effective batch size = 48
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.3
    seed: int = 42


# ============================================================================
# 유틸리티 함수들
# ============================================================================
def check_gpu():
    """GPU 확인"""
    print("=" * 50)
    print("GPU 확인")
    print("=" * 50)
    subprocess.run(["nvidia-smi"], check=False)
    print()


def check_paths():
    """경로 확인"""
    print("=" * 50)
    print("경로 확인")
    print("=" * 50)
    print(f"데이터셋 경로: {LOCAL_DATA_DIR}")
    print(f"베이스 모델 경로: {LOCAL_MODEL_DIR}")
    print(f"모델 저장 경로: {OUTPUT_DIR}")

    # 로컬 데이터셋 존재 확인
    if os.path.exists(LOCAL_DATA_DIR):
        print(f"\n[OK] 데이터셋을 찾았습니다.")
    else:
        print(f"\n[ERROR] 데이터셋을 찾을 수 없습니다!")
        print(f"경로를 확인하세요: {LOCAL_DATA_DIR}")

    # 베이스 모델 존재 확인
    if os.path.exists(LOCAL_MODEL_DIR) and len(os.listdir(LOCAL_MODEL_DIR)) > 0:
        print(f"[OK] 베이스 모델을 찾았습니다.")
    else:
        print(f"\n[WARNING] 베이스 모델이 없습니다!")
        print(f"download_base_model() 함수를 실행하세요.")
    print()


def download_base_model():
    """베이스 모델을 로컬에 다운로드 (최초 1회만)"""
    from huggingface_hub import snapshot_download

    print("=" * 50)
    print("베이스 모델 다운로드")
    print("=" * 50)

    def check_model_exists():
        """로컬에 모델이 이미 존재하는지 확인"""
        model_path = Path(LOCAL_MODEL_DIR)
        if not model_path.exists():
            return False

        # 필수 파일 확인
        required = ["config.json", "tokenizer.json"]
        for f in required:
            if not (model_path / f).exists():
                return False

        # safetensors 파일 확인
        if len(list(model_path.glob("*.safetensors"))) == 0:
            return False

        return True

    if check_model_exists():
        print(f"[OK] 모델이 이미 존재합니다: {LOCAL_MODEL_DIR}")
        print("다운로드를 건너뜁니다.")
    else:
        print(f"모델을 다운로드합니다...")
        print(f"모델: {HF_MODEL_NAME}")
        print(f"저장 경로: {LOCAL_MODEL_DIR}")
        print("\n다운로드 중... (약 26GB, 시간이 걸릴 수 있습니다)")

        os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

        downloaded_path = snapshot_download(
            repo_id=HF_MODEL_NAME,
            local_dir=LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        print(f"\n[OK] 다운로드 완료!")
        print(f"모델 저장 위치: {downloaded_path}")

    # 파일 목록 출력
    print("\n저장된 파일:")
    for f in sorted(Path(LOCAL_MODEL_DIR).iterdir())[:10]:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  {f.name}: {size_mb:.1f} MB")
    print()


# ============================================================================
# 학습 관련 함수들
# ============================================================================
def setup_quantization_config(model_config: ModelConfig):
    """4bit 양자화 설정"""
    from transformers import BitsAndBytesConfig

    compute_dtype = getattr(torch, model_config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.use_double_quant,
    )


def load_model_and_tokenizer(model_config: ModelConfig, quantization_config):
    """모델과 토크나이저 로드"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import prepare_model_for_kbit_training

    print(f"Loading model from: {model_config.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
        padding_side="right",
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def setup_lora(model, lora_config: LoRAConfig):
    """LoRA 어댑터 설정"""
    from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType

    config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        bias=lora_config.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def preprocess_dataset(dataset, tokenizer, max_length: int):
    """데이터셋 전처리"""
    def tokenize_function(examples):
        texts = []
        for i in range(len(examples['source'])):
            source_lang = examples['source_lang'][i]
            target_lang = examples['target_lang'][i]
            source_text = examples['source'][i]
            target_text = examples['target'][i]

            prompt = f"""<|im_start|>user
Translate the following text from {source_lang} into {target_lang}.
{source_lang}: {source_text}
{target_lang}:<|im_end|>
<|im_start|>assistant
{target_text}<|im_end|>"""
            texts.append(prompt)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

        tokenized["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in tokenized["input_ids"]
        ]
        return tokenized

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def create_trainer(model, tokenizer, train_dataset, eval_dataset, train_config: TrainConfig):
    """Trainer 생성"""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        num_train_epochs=train_config.num_train_epochs,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        per_device_eval_batch_size=train_config.per_device_eval_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        learning_rate=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        warmup_ratio=train_config.warmup_ratio,
        lr_scheduler_type=train_config.lr_scheduler_type,
        logging_steps=train_config.logging_steps,
        save_steps=train_config.save_steps,
        eval_steps=train_config.eval_steps,
        eval_strategy="steps",
        save_total_limit=train_config.save_total_limit,
        fp16=train_config.fp16,
        bf16=train_config.bf16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        optim=train_config.optim,
        max_grad_norm=train_config.max_grad_norm,
        seed=train_config.seed,
        report_to="tensorboard",
        logging_dir=f"{train_config.output_dir}/logs",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_safetensors=True,
        ignore_data_skip=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def check_checkpoints():
    """체크포인트 확인"""
    output_dir = Path(OUTPUT_DIR)
    checkpoints = list(output_dir.glob("checkpoint-*"))

    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for cp in sorted(checkpoints, key=lambda x: int(x.name.split('-')[1])):
            print(f"  - {cp.name}")
    else:
        print("No checkpoints found yet.")


# ============================================================================
# 메인 함수
# ============================================================================
def train():
    """학습 실행"""
    from datasets import load_from_disk

    model_config = ModelConfig()
    lora_config = LoRAConfig()
    train_config = TrainConfig()

    print("=" * 50)
    print("Tower-13B QLoRA 파인튜닝 시작")
    print("=" * 50)
    print(f"\n베이스 모델: {model_config.model_name}")

    # 모델 존재 확인
    if not Path(model_config.model_name).exists():
        print(f"\n오류: 모델을 찾을 수 없습니다!")
        print(f"경로: {model_config.model_name}")
        print("download_base_model() 함수를 먼저 실행하세요.")
        return

    # 데이터셋 존재 확인
    if not Path(LOCAL_DATA_DIR).exists():
        print(f"\n오류: 데이터셋을 찾을 수 없습니다!")
        print(f"경로: {LOCAL_DATA_DIR}")
        return

    # 데이터 로드
    print(f"\nLoading dataset from: {LOCAL_DATA_DIR}...")
    dataset = load_from_disk(LOCAL_DATA_DIR)
    print(f"  Train: {len(dataset['train']):,} examples")
    print(f"  Validation: {len(dataset['validation']):,} examples")

    # 모델 로드
    quantization_config = setup_quantization_config(model_config)
    model, tokenizer = load_model_and_tokenizer(model_config, quantization_config)

    # LoRA 설정
    model = setup_lora(model, lora_config)

    # 데이터 전처리
    print("\nPreprocessing datasets...")
    train_dataset = preprocess_dataset(dataset['train'], tokenizer, model_config.max_length)
    eval_dataset = preprocess_dataset(dataset['validation'], tokenizer, model_config.max_length)

    # Trainer 생성
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, train_config)

    # 체크포인트 확인
    checkpoint_dir = Path(train_config.output_dir)
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))

    resume_checkpoint = None
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        resume_checkpoint = str(checkpoints[-1])
        print(f"\n{'=' * 50}")
        print(f"Found checkpoint: {resume_checkpoint}")
        print(f"Resuming training...")
        print(f"{'=' * 50}\n")
    else:
        print(f"\n{'=' * 50}")
        print("Starting training from scratch...")
        print(f"{'=' * 50}\n")

    # 학습 시작
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 최종 모델 저장
    print("\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(train_config.output_dir)
    model.save_pretrained(f"{train_config.output_dir}/lora_adapter")

    print(f"\n{'=' * 50}")
    print(f"Training complete!")
    print(f"Model saved to: {train_config.output_dir}")
    print(f"{'=' * 50}")


def main():
    """전체 파이프라인 실행"""
    print("""
================================================================================
Tower-13B QLoRA 파인튜닝 스크립트
================================================================================

사용 방법:
1. check_paths()        - 경로 확인
2. download_base_model() - 베이스 모델 다운로드 (최초 1회)
3. train()              - 학습 실행

또는 main()을 실행하면 전체 파이프라인이 순차적으로 실행됩니다.
================================================================================
    """)

    # 1. GPU 확인
    check_gpu()

    # 2. 경로 확인
    check_paths()

    # 3. 베이스 모델 다운로드 (필요한 경우)
    download_base_model()

    # 4. 학습 실행
    train()


if __name__ == "__main__":
    main()
