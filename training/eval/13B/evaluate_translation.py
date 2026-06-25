"""
번역 모델 성능 평가 (BLEU, BERTScore)

사용법:
    python evaluate_translation.py --predictions pred.txt --references ref.txt

파일 형식:
    - 각 파일은 한 줄에 하나의 문장
    - predictions.txt: 모델이 번역한 결과 (한국어)
    - references.txt: 정답 번역 (한국어)
"""
import argparse
from pathlib import Path
from typing import List
import torch

# BLEU 계산용
from sacrebleu import corpus_bleu

# BERTScore 계산용
from bert_score import score as bert_score


def load_lines(file_path: str) -> List[str]:
    """파일에서 문장 로드 (한 줄에 하나씩)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def calculate_bleu(predictions: List[str], references: List[str]) -> dict:
    """
    BLEU 점수 계산

    Returns:
        {'bleu': float, 'bleu_detail': str}
    """
    print("\n[1/2] BLEU 계산 중...")

    # sacrebleu는 references를 [[ref1], [ref2], ...] 형태로 받음
    refs_wrapped = [[ref] for ref in references]

    # BLEU 계산
    bleu_result = corpus_bleu(predictions, refs_wrapped)

    print(f"✓ BLEU 계산 완료")

    return {
        'bleu': bleu_result.score,
        'bleu_detail': str(bleu_result)
    }


def calculate_bertscore(predictions: List[str], references: List[str], lang: str = 'ko') -> dict:
    """
    BERTScore 계산

    Args:
        predictions: 예측 문장 리스트
        references: 정답 문장 리스트
        lang: 언어 코드 ('ko' for Korean)

    Returns:
        {'precision': float, 'recall': float, 'f1': float}
    """
    print("\n[2/2] BERTScore 계산 중...")
    print(f"⏳ 언어: {lang}, GPU: {torch.cuda.is_available()}")

    # BERTScore 계산
    P, R, F1 = bert_score(
        predictions,
        references,
        lang=lang,
        verbose=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # 평균 점수
    precision = P.mean().item()
    recall = R.mean().item()
    f1 = F1.mean().item()

    print(f"✓ BERTScore 계산 완료")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def evaluate_translation(pred_file: str, ref_file: str, lang: str = 'ko'):
    """
    번역 모델 평가 (BLEU + BERTScore)

    Args:
        pred_file: 예측 파일 경로
        ref_file: 정답 파일 경로
        lang: 언어 코드
    """
    print("=" * 80)
    print("번역 모델 성능 평가 (BLEU + BERTScore)")
    print("=" * 80)
    print(f"Predictions: {pred_file}")
    print(f"References : {ref_file}")
    print(f"Language   : {lang}")
    print("=" * 80)

    # 파일 로드
    print("\n파일 로드 중...")
    predictions = load_lines(pred_file)
    references = load_lines(ref_file)

    print(f"✓ Predictions: {len(predictions)}개 문장")
    print(f"✓ References : {len(references)}개 문장")

    # 개수 확인
    if len(predictions) != len(references):
        print(f"\n⚠️  경고: 예측({len(predictions)})과 정답({len(references)}) 문장 수가 다릅니다!")
        min_len = min(len(predictions), len(references))
        predictions = predictions[:min_len]
        references = references[:min_len]
        print(f"→ 처음 {min_len}개 문장만 평가합니다.")

    # BLEU 계산
    bleu_results = calculate_bleu(predictions, references)

    # BERTScore 계산
    bertscore_results = calculate_bertscore(predictions, references, lang)

    # 결과 출력
    print("\n" + "=" * 80)
    print("평가 결과")
    print("=" * 80)
    print(f"샘플 수: {len(predictions)}개")
    print("-" * 80)
    print("BLEU Score:")
    print(f"  BLEU: {bleu_results['bleu']:.2f}")
    print("-" * 80)
    print("BERTScore:")
    print(f"  Precision: {bertscore_results['precision']:.4f}")
    print(f"  Recall   : {bertscore_results['recall']:.4f}")
    print(f"  F1       : {bertscore_results['f1']:.4f}")
    print("=" * 80)

    # 샘플 출력 (처음 3개)
    print("\n샘플 비교 (처음 3개):")
    print("-" * 80)
    for i in range(min(3, len(predictions))):
        print(f"\n[{i+1}]")
        print(f"  예측: {predictions[i]}")
        print(f"  정답: {references[i]}")
    print("-" * 80)

    return {
        'bleu': bleu_results,
        'bertscore': bertscore_results
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="번역 모델 BLEU/BERTScore 평가")
    parser.add_argument(
        "--predictions",
        type=str,
        default="/home/yugeon/trans_learning_1210/eval/13B/predictions.txt",
        help="예측 파일 경로 (한 줄에 하나씩)"
    )
    parser.add_argument(
        "--references",
        type=str,
        default="/home/yugeon/trans_learning_1210/eval/13B/answer.txt",
        help="정답 파일 경로 (한 줄에 하나씩)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ko",
        help="언어 코드 (기본값: ko)"
    )

    args = parser.parse_args()

    # 파일 존재 확인
    if not Path(args.predictions).exists():
        print(f"❌ 예측 파일을 찾을 수 없습니다: {args.predictions}")
        exit(1)

    if not Path(args.references).exists():
        print(f"❌ 정답 파일을 찾을 수 없습니다: {args.references}")
        exit(1)

    # 평가 실행
    evaluate_translation(args.predictions, args.references, args.lang)
