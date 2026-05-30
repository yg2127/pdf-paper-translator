"""
FTE(Figure/Table/Equation) 및 Title에서 용어를 자동 추출하는 모듈

추출 규칙:
1. 대문자 약어 (BERT, CNN, GPT 등)
2. 그리스 문자 (α, β, γ 등)
3. 숫자 포함 단어 (F1, Layer2, ResNet50 등)
4. Figure/Table/Equation + 숫자
5. 일반 영단어는 제외
"""

import re
import sys
from typing import List, Dict, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TERM_EXTRACT_CLASSES


# 일반 영단어 (번역 대상, 용어 사전 제외)
COMMON_ENGLISH_WORDS = {
    # 관사, 전치사, 접속사
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
    'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'which',
    'that', 'this', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
    
    # 일반 동사
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'must', 'get', 'got', 'make', 'made',
    'show', 'shows', 'shown', 'use', 'used', 'using',
    
    # 일반 형용사/부사
    'more', 'most', 'less', 'least', 'very', 'much', 'many', 'some', 'any',
    'all', 'each', 'every', 'both', 'few', 'several', 'other', 'another',
    'such', 'same', 'different', 'new', 'old', 'high', 'low', 'large', 'small',
    'good', 'better', 'best', 'well', 'also', 'only', 'just', 'even', 'still',
    
    # 학술 논문에서 자주 나오지만 번역해야 하는 단어
    'results', 'result', 'method', 'methods', 'model', 'models', 'data',
    'analysis', 'study', 'studies', 'research', 'paper', 'work', 'approach',
    'system', 'systems', 'performance', 'accuracy', 'error', 'errors',
    'training', 'testing', 'learning', 'network', 'layer', 'layers',
    'input', 'output', 'feature', 'features', 'image', 'images', 'text',
    'value', 'values', 'number', 'numbers', 'size', 'time', 'step', 'steps',
    'process', 'algorithm', 'function', 'based', 'proposed', 'presented',
    'compared', 'obtained', 'achieved', 'improved', 'experimental',
}

# 그리스 문자
GREEK_LETTERS = {
    'α', 'β', 'γ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'λ', 'μ',
    'ν', 'ξ', 'ο', 'π', 'ρ', 'σ', 'τ', 'υ', 'φ', 'χ', 'ψ', 'ω',
    'Α', 'Β', 'Γ', 'Δ', 'Ε', 'Ζ', 'Η', 'Θ', 'Ι', 'Κ', 'Λ', 'Μ',
    'Ν', 'Ξ', 'Ο', 'Π', 'Ρ', 'Σ', 'Τ', 'Υ', 'Φ', 'Χ', 'Ψ', 'Ω',
    # 라틴 표기
    'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'theta', 'lambda',
    'sigma', 'omega', 'phi', 'psi', 'mu', 'nu', 'pi', 'rho', 'tau',
}


@dataclass
class ExtractedTerm:
    """추출된 용어 정보"""
    term: str                           # 용어
    source_type: str                    # 출처 타입 (figure, table, equation, title)
    page: int                           # 페이지 번호
    preserve: bool = True               # 유지 여부 (True=번역 안 함)
    reason: str = ""                    # 유지/제외 이유
    frequency: int = 1                  # 등장 빈도


@dataclass
class TermDictionary:
    """용어 사전"""
    terms: Dict[str, ExtractedTerm] = field(default_factory=dict)
    
    def add_term(self, term: ExtractedTerm):
        """용어 추가 (중복 시 빈도 증가)"""
        if term.term in self.terms:
            self.terms[term.term].frequency += 1
        else:
            self.terms[term.term] = term
    
    def get_preserve_terms(self) -> Set[str]:
        """유지할 용어 목록 반환"""
        return {t.term for t in self.terms.values() if t.preserve}
    
    def to_tsv(self, output_path: str):
        """TSV 파일로 저장"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("용어\t출처\t페이지\t빈도\t유지여부\t이유\n")
            for term in sorted(self.terms.values(), key=lambda x: (-x.frequency, x.term)):
                preserve_mark = "O" if term.preserve else "X"
                f.write(f"{term.term}\t{term.source_type}\t{term.page}\t{term.frequency}\t{preserve_mark}\t{term.reason}\n")
        print(f"용어 사전 저장: {output_path} ({len(self.terms)}개 용어)")
    
    def __len__(self):
        return len(self.terms)


class TermExtractor:
    """FTE 및 Title에서 용어를 추출하는 클래스"""
    
    def __init__(
        self,
        extract_from_fte: bool = True,
        extract_from_title: bool = True,
    ):
        """
        Args:
            extract_from_fte: FTE(Figure/Table/Equation)에서 추출 여부
            extract_from_title: Title에서 대문자 용어 추출 여부
        """
        self.extract_from_fte = extract_from_fte
        self.extract_from_title = extract_from_title
        self.dictionary = TermDictionary()
    
    def extract_from_ocr_results(
        self,
        ocr_results: List[Dict],
        detections: List[Dict],
        page_num: int = 0
    ) -> TermDictionary:
        """
        OCR 결과와 YOLO 감지 결과에서 용어 추출
        
        Args:
            ocr_results: OCR 결과 리스트 [{'text': ..., 'bbox': ...}, ...]
            detections: YOLO 감지 결과 [{'class': ..., 'bbox': ...}, ...]
            page_num: 페이지 번호
            
        Returns:
            TermDictionary
        """
        for det in detections:
            class_name = det.get('class', det.get('class_name', ''))
            
            # FTE에서 추출
            if self.extract_from_fte and class_name in TERM_EXTRACT_CLASSES:
                text = det.get('ocr_text', '')
                if text:
                    self._extract_terms_from_text(text, class_name, page_num)
            
            # Title에서 추출
            if self.extract_from_title and class_name == 'title':
                text = det.get('ocr_text', '')
                if text:
                    self._extract_terms_from_title(text, page_num)
        
        return self.dictionary
    
    def extract_from_text(
        self,
        text: str,
        source_type: str,
        page_num: int = 0
    ):
        """텍스트에서 용어 추출"""
        if source_type == 'title':
            self._extract_terms_from_title(text, page_num)
        else:
            self._extract_terms_from_text(text, source_type, page_num)
    
    def _extract_terms_from_text(self, text: str, source_type: str, page_num: int):
        """FTE 텍스트에서 용어 추출"""
        # 단어 분리
        words = re.findall(r'[\w\-]+|[α-ωΑ-Ω]', text)
        
        for word in words:
            should_preserve, reason = self._should_preserve(word)
            # 숫자만 있는 경우는 용어 사전에 추가하지 않음
            if word.isdigit():
                continue
            
            term = ExtractedTerm(
                term=word,
                source_type=source_type,
                page=page_num,
                preserve=should_preserve,
                reason=reason
            )
            self.dictionary.add_term(term)
    
    def _extract_terms_from_title(self, text: str, page_num: int):
        """Title에서 대문자 시작 단어만 추출"""
        words = re.findall(r'[\w\-]+', text)
        
        for word in words:
            # Title에서는 대문자로 시작하는 단어만 추출
            if not word[0].isupper():
                continue
            
            # 일반 영단어 제외
            if word.lower() in COMMON_ENGLISH_WORDS:
                continue
            
            should_preserve, reason = self._should_preserve(word)
            
            # Title 출처는 별도 표시
            if should_preserve:
                reason = f"Title 대문자: {reason}"
            
            term = ExtractedTerm(
                term=word,
                source_type='title',
                page=page_num,
                preserve=should_preserve,
                reason=reason
            )
            self.dictionary.add_term(term)
    
    def _should_preserve(self, word: str) -> Tuple[bool, str]:
        """
        단어를 유지해야 하는지 판단
        
        Returns:
            (유지 여부, 이유)
        """
        # 빈 문자열 또는 너무 짧은 단어
        if not word or len(word) < 2:
            return False, "너무 짧음"
        
        # 숫자만 있는 경우 (보호하지 않음)
        if word.isdigit():
            return False, "숫자"
        
        # 그리스 문자
        if word in GREEK_LETTERS or any(c in GREEK_LETTERS for c in word):
            return True, "그리스 문자"
        
        # Figure/Table/Equation + 숫자
        if re.match(r'^(Figure|Fig|Table|Tab|Equation|Eq)\.?\s*\d+', word, re.IGNORECASE):
            return True, "FTE 참조"
        
        # 대문자 약어 (2글자 이상)
        if re.match(r'^[A-Z]{2,}$', word):
            return True, "대문자 약어"
        
        # 대문자 + 숫자 조합 (ResNet50, GPT3 등)
        if re.match(r'^[A-Z][a-zA-Z]*\d+', word):
            return True, "모델명/버전"
        
        # 대문자로 시작하고 중간에 대문자 있음 (CamelCase: PyTorch, TensorFlow)
        if re.match(r'^[A-Z][a-z]+[A-Z]', word):
            return True, "CamelCase"
        
        # 하이픈 포함 기술 용어 (F1-score, self-attention)
        if '-' in word and not word.lower() in COMMON_ENGLISH_WORDS:
            parts = word.split('-')
            if any(p[0].isupper() if p else False for p in parts):
                return True, "하이픈 기술용어"
        
        # 일반 영단어는 제외
        if word.lower() in COMMON_ENGLISH_WORDS:
            return False, "일반 영단어"
        
        # 소문자만 있으면 일반 단어로 간주
        if word.islower():
            return False, "소문자 단어"
        
        # 나머지는 유지 (판단 불가)
        return True, "기타 (수동 확인 필요)"
    
    def get_terminology_dict(self) -> Dict[str, str]:
        """
        translator.py에서 사용할 용어 사전 반환
        
        Returns:
            {영어용어: 영어용어} (유지할 용어만)
        """
        preserve_terms = self.dictionary.get_preserve_terms()
        return {term: term for term in preserve_terms}
    
    def save_terms(self, output_path: str):
        """추출된 용어를 파일로 저장"""
        self.dictionary.to_tsv(output_path)
    
    def clear(self):
        """용어 사전 초기화"""
        self.dictionary = TermDictionary()


# 테스트용
if __name__ == "__main__":
    extractor = TermExtractor(
        extract_from_fte=True,
        extract_from_title=True
    )
    
    # 테스트 텍스트
    test_texts = [
        ("BERT achieves 0.95 F1-score on the dataset", "figure"),
        ("ResNet50 vs VGG16 comparison", "table"),
        ("α = 0.01, β = 0.99", "equation"),
        ("Attention Is All You Need: Transformer Architecture", "title"),
    ]
    
    for text, source in test_texts:
        extractor.extract_from_text(text, source, page_num=1)
    
    print("\n=== 추출된 용어 ===")
    for term, info in extractor.dictionary.terms.items():
        status = "✓ 유지" if info.preserve else "✗ 번역"
        print(f"{status} | {term:20} | {info.source_type:10} | {info.reason}")
    
    print(f"\n유지할 용어: {extractor.dictionary.get_preserve_terms()}")
