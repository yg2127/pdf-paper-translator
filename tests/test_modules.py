"""
순수 로직 단위 테스트 — 모델·외부 라이브러리(torch/fitz/reportlab) 없이 돌아간다.

좌표 변환과 용어 추출 규칙처럼 외부 의존성이 없는 부분만 검증한다.
무거운 ML 스택을 깔지 않고도 핵심 로직을 빠르게 확인하려고,
패키지(modules/__init__.py)의 일괄 import를 우회해 해당 모듈만 직접 불러온다.

실행:  python -m pytest tests/   또는   python tests/test_modules.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # config.py
sys.path.insert(0, os.path.join(ROOT, "modules"))

from coordinate_transformer import CoordinateTransformer
from term_extractor import TermExtractor, ExtractedTerm, TermDictionary


def test_yolo_to_pymupdf_scaling():
    """이미지가 페이지의 2배 크기면 픽셀 좌표가 절반 포인트로 변환된다."""
    ct = CoordinateTransformer()
    assert ct.yolo_to_pymupdf((100, 200, 300, 400), 612, 792, 1224, 1584) == (
        50.0,
        100.0,
        150.0,
        200.0,
    )


def test_intersects():
    ct = CoordinateTransformer()
    assert ct.intersects((0, 0, 10, 10), (5, 5, 15, 15)) is True
    assert ct.intersects((0, 0, 10, 10), (10, 10, 20, 20)) is False  # 모서리만 닿음
    assert ct.intersects((0, 0, 10, 10), (20, 20, 30, 30)) is False


def test_term_should_preserve():
    te = TermExtractor()
    for w in ["BERT", "ResNet50", "PyTorch", "F1-score", "GPT3"]:
        assert te._should_preserve(w)[0] is True, w
    for w in ["the", "model", "method", "results"]:
        assert te._should_preserve(w)[0] is False, w


def test_term_dictionary_dedup():
    d = TermDictionary()
    d.add_term(ExtractedTerm(term="BERT", source_type="title", page=1))
    d.add_term(ExtractedTerm(term="BERT", source_type="title", page=1))
    assert d.terms["BERT"].frequency == 2
    assert len(d) == 1


def test_extract_from_text_populates_dict():
    te = TermExtractor()
    te.extract_from_text("BERT achieves strong results", source_type="text", page_num=1)
    preserved = te.dictionary.get_preserve_terms()
    assert "BERT" in preserved
    assert "achieves" not in preserved  # 소문자 일반어는 번역 대상


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\n{len(tests)}개 테스트 모두 통과")
