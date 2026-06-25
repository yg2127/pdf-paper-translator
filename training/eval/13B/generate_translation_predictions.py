"""
번역 모델로 예측 생성 (평가용)

사용법:
    python generate_translation_predictions.py --input source.txt --output predictions.txt

입력 파일 형식:
    - source.txt: 영어 문장 (한 줄에 하나씩)

출력:
    - predictions.txt: 번역된 한국어 문장 (한 줄에 하나씩)
"""
import argparse
from pathlib import Path
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_model(model_path: str):
    """13B 번역 모델 로드"""
    print(f"\n모델 로드 중: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if torch.cuda.is_available():
        print("GPU 사용 (8-bit 양자화)")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("CPU 사용")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

    model.eval()
    print("✓ 모델 로드 완료")

    return model, tokenizer


def translate_batch(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    device: str = 'cuda'
) -> List[str]:
    """배치 번역"""
    # ChatML 프롬프트 생성
    prompts = []
    for text in texts:
        prompt = f"<|im_start|>user\nTranslate the following text from English into Korean.\nEnglish: {text}\nKorean:<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(prompt)

    # 토크나이징
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    if device == 'cuda' and torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 디코딩
    translations = []
    for output in outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)

        # "assistant\n" 이후 텍스트만 추출
        if "assistant\n" in decoded:
            translation = decoded.split("assistant\n")[-1].strip()
        else:
            translation = decoded.strip()

        translations.append(translation)

    return translations


def generate_predictions(
    model_path: str,
    input_file: str,
    output_file: str,
    batch_size: int = 8
):
    """
    번역 예측 생성

    Args:
        model_path: 번역 모델 경로
        input_file: 영어 문장 파일
        output_file: 번역 결과 저장 경로
        batch_size: 배치 크기
    """
    print("=" * 80)
    print("번역 예측 생성")
    print("=" * 80)
    print(f"모델: {model_path}")
    print(f"입력: {input_file}")
    print(f"출력: {output_file}")
    print(f"배치 크기: {batch_size}")
    print("=" * 80)

    # 모델 로드
    model, tokenizer = load_model(model_path)

    # 입력 파일 로드
    print(f"\n입력 파일 로드 중: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        source_texts = [line.strip() for line in f if line.strip()]
    print(f"✓ {len(source_texts)}개 문장 로드")

    # 배치 번역
    print(f"\n번역 시작 (배치 크기: {batch_size})...")
    all_translations = []

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for i in tqdm(range(0, len(source_texts), batch_size)):
        batch_texts = source_texts[i:i+batch_size]
        translations = translate_batch(model, tokenizer, batch_texts, device=device)
        all_translations.extend(translations)

    print(f"✓ 번역 완료: {len(all_translations)}개 문장")

    # 결과 저장
    print(f"\n결과 저장 중: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for translation in all_translations:
            f.write(translation + '\n')
    print(f"✓ 저장 완료")

    # 샘플 출력
    print("\n샘플 번역 결과 (처음 3개):")
    print("-" * 80)
    for i in range(min(3, len(source_texts))):
        print(f"\n[{i+1}]")
        print(f"  영어: {source_texts[i]}")
        print(f"  한국어: {all_translations[i]}")
    print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="번역 예측 생성")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/yugeon/trained_model/1212-13B/13B_merged_fp16",
        help="번역 모델 경로"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/home/yugeon/trans_learning_1210/eval/13B/problem.txt",
        help="입력 파일 (영어 문장, 한 줄에 하나씩)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/yugeon/trans_learning_1210/eval/13B/predictions.txt",
        help="출력 파일 (번역된 한국어)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="배치 크기 (기본값: 8)"
    )

    args = parser.parse_args()

    # 파일 확인
    if not Path(args.input).exists():
        print(f"❌ 입력 파일을 찾을 수 없습니다: {args.input}")
        exit(1)

    if not Path(args.model).exists():
        print(f"❌ 모델을 찾을 수 없습니다: {args.model}")
        exit(1)

    # 예측 생성
    generate_predictions(args.model, args.input, args.output, args.batch_size)
