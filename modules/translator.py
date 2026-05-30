"""
영어를 한국어로 번역하는 모듈
라이브러리: Hugging Face Transformers
지원 모델:
  - Helsinki-NLP/opus-mt-en-ko (MarianMT)
  - Custom 13B model (ChatML format)
"""

from typing import List, Dict, Union
import re
from pathlib import Path

try:
    from transformers import (
        MarianMTModel, MarianTokenizer,
        AutoModelForCausalLM, AutoTokenizer,
        BitsAndBytesConfig
    )
    import torch
except ImportError:
    print("transformers가 설치되지 않았습니다. pip install transformers torch")
    MarianMTModel = None
    MarianTokenizer = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    BitsAndBytesConfig = None
    torch = None


class Translator:
    """영어-한국어 번역기"""
    
    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-en-ko",
        device: str = None,
        max_length: int = 512,
        term_bold: bool = True,
        use_4bit: bool = True,
        model_type: str = None
    ):
        """
        Args:
            model_name: Hugging Face 모델 이름 또는 로컬 경로
            device: 디바이스 ('cuda', 'cpu', 또는 None=자동)
            max_length: 최대 토큰 길이
            term_bold: 용어 사전 단어를 볼드체로 표시할지 여부
            use_4bit: 4-bit 양자화 사용 (13B 모델용)
            model_type: 모델 타입 ('marian', 'causal', 또는 None=자동 감지)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.device = device
        self.term_bold = term_bold
        self.use_4bit = use_4bit
        # 플레이스홀더 래핑 문자
        self.placeholder_prefix = "<<"
        self.placeholder_suffix = ">>"

        # 모델 타입 자동 감지
        if model_type is None:
            self.model_type = self._detect_model_type(model_name)
        else:
            self.model_type = model_type

        # 학술 용어 사전 (WordPiece/Vocabulary 확장 대안)
        self.terminology_dict = {}
        # 볼드 처리 대상 용어
        self.bold_terms = set()
        self.bold_terms_lower = set()

        self._load_model()

    def _detect_model_type(self, model_name: str) -> str:
        """
        모델 타입 자동 감지

        Returns:
            'marian' 또는 'causal'
        """
        # 로컬 경로인 경우 causal (13B 모델)
        if Path(model_name).exists():
            return 'causal'

        # Helsinki-NLP 모델은 marian
        if 'Helsinki-NLP' in model_name or 'opus-mt' in model_name:
            return 'marian'

        # 기타 경우 causal로 간주
        return 'causal'
    
    def _load_model(self):
        """모델과 토크나이저 로드"""
        if torch is None:
            print("경고: transformers와 torch가 설치되지 않았습니다.")
            return

        # 디바이스 설정
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            if self.model_type == 'marian':
                self._load_marian_model()
            elif self.model_type == 'causal':
                self._load_causal_model()
            else:
                raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")

            print(f"번역 모델 로드 완료: {self.model_name} ({self.model_type}) on {self.device}")
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()

    def _load_marian_model(self):
        """MarianMT 모델 로드"""
        if MarianMTModel is None or MarianTokenizer is None:
            raise ImportError("MarianMT 모델 로드 실패: transformers 설치 필요")

        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _load_causal_model(self):
        """Causal LM (13B 모델) 로드"""
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            raise ImportError("Causal LM 로드 실패: transformers 설치 필요")

        print(f"토크나이저 로드 중: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        print(f"모델 로드 중: {self.model_name}")
        if self.device == 'cuda':
            # 8-bit 양자화로 로드 (VRAM 절약, CPU offload 활성화)
            print("8-bit 양자화 사용 (CPU offload 활성화)")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,  # CPU offload 활성화
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU는 양자화 없이 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
        self.model.eval()
    
    def translate(self, text: str) -> str:
        """
        단일 텍스트 번역

        Args:
            text: 영어 텍스트

        Returns:
            한국어 번역 텍스트
        """
        if self.model is None:
            return text  # 모델 없으면 원본 반환

        if not text or not text.strip():
            return text

        # 용어/수식/참조 보호 후 번역
        processed_text, placeholders = self._protect_terminology(text)

        if self.model_type == 'marian':
            translated = self._translate_with_marian(processed_text)
        elif self.model_type == 'causal':
            translated = self._translate_with_causal(processed_text)
        else:
            translated = processed_text

        # 보호된 항목 복원 (용어는 볼드 처리 옵션 적용)
        return self._restore_terminology(translated, placeholders)

    def _translate_with_marian(self, text: str) -> str:
        """MarianMT 모델로 번역"""
        # 토큰화
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        # 번역
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=4,
                early_stopping=True
            )

        # 디코딩
        translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated

    def _translate_with_causal(self, text: str) -> str:
        """Causal LM (13B 모델)로 번역 (ChatML 프롬프트 사용)"""
        # 너무 짧은 텍스트는 번역하지 않음
        if len(text.strip()) < 3:
            return text

        # 숫자만 있으면 번역하지 않음
        if text.strip().replace('.', '').replace(',', '').isdigit():
            return text

        # ChatML 형식 프롬프트 생성 (평가 스크립트와 동일한 형식)
        prompt = f"<|im_start|>user\nTranslate the following text from English into Korean.\nEnglish: {text}\nKorean:<|im_end|>\n<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 텍스트 생성 (repetition_penalty 추가)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # 반복 방지
            )

        # 생성된 텍스트 디코딩 (skip_special_tokens=True로 깔끔하게!)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # "assistant\n" 이후 텍스트만 추출 (평가 스크립트와 동일)
        if "assistant\n" in decoded:
            translated = decoded.split("assistant\n")[-1].strip()
        elif "assistant" in decoded:
            translated = decoded.split("assistant")[-1].strip()
        else:
            translated = decoded.strip()

        # 이상한 출력 필터링
        if "English:" in translated or "Korean:" in translated:
            # 프롬프트가 다시 나온 경우, 마지막 Korean: 이후만 추출
            if "Korean:" in translated:
                translated = translated.split("Korean:")[-1].strip()
            else:
                translated = text  # 실패 시 원본 반환

        # 환각(hallucination) 감지 및 방지
        # 1. 번역 결과가 원본보다 3배 이상 길면 문제
        if len(translated) > len(text) * 3:
            return text

        # 2. 관련 없는 키워드 감지 (뉴스, 연도, 일반적인 환각 패턴)
        hallucination_keywords = ['20년', '장학', '공단', '신문', '안내', '일시', '선정', '전문위원']
        for keyword in hallucination_keywords:
            if keyword in translated:
                return text

        return translated
    
    def translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        """
        배치 번역 (효율적인 다중 텍스트 번역)

        Args:
            texts: 영어 텍스트 리스트
            batch_size: 배치 크기

        Returns:
            한국어 번역 텍스트 리스트
        """
        if self.model is None:
            return texts

        # Causal 모델은 배치 처리가 복잡하므로 순차 처리
        if self.model_type == 'causal':
            return [self.translate(text) for text in texts]

        # MarianMT는 기존 배치 처리 사용
        all_translations = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            # 빈 텍스트 처리
            non_empty_indices = [j for j, t in enumerate(batch) if t and t.strip()]
            non_empty_texts = [batch[j] for j in non_empty_indices]

            if not non_empty_texts:
                all_translations.extend(batch)
                continue

            # 전처리
            processed_texts = []
            all_placeholders = []
            for text in non_empty_texts:
                processed, placeholders = self._protect_terminology(text)
                processed_texts.append(processed)
                all_placeholders.append(placeholders)

            # 토큰화
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)

            # 번역
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True
                )

            # 디코딩
            batch_translations = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )

            # 후처리
            for j, trans in enumerate(batch_translations):
                batch_translations[j] = self._restore_terminology(
                    trans,
                    all_placeholders[j]
                )

            # 결과 조합
            result_batch = list(batch)
            for j, idx in enumerate(non_empty_indices):
                result_batch[idx] = batch_translations[j]

            all_translations.extend(result_batch)

        return all_translations
    
    def _make_placeholder(self, tag: str, idx: int) -> str:
        """모델이 변형하기 어려운 플레이스홀더 문자열 생성"""
        return f"<<{tag}_{idx}>>"

    def _protect_terminology(self, text: str) -> tuple:
        """
        학술 용어/참조를 플레이스홀더로 보호
        
        Returns:
            (처리된 텍스트, 플레이스홀더 딕셔너리)
        """
        placeholders = {}

        # 길이 긴 용어부터 치환 (대소문자 무시)
        terms = sorted(self.terminology_dict.keys(), key=len, reverse=True)
        term_index = 0
        for term in terms:
            if not term:
                continue
            pattern = re.escape(term)
            if term.isalnum():
                pattern = r'\b' + pattern + r'\b'

            search_pos = 0
            while True:
                m = re.search(pattern, text[search_pos:], flags=re.IGNORECASE)
                if not m:
                    break
                start = search_pos + m.start()
                end = search_pos + m.end()
                ph = self._make_placeholder("TERM", term_index)
                term_index += 1
                placeholders[ph] = term  # 원본 용어(사전상의 형태)로 복원
                text = text[:start] + ph + text[end:]
                search_pos = start + len(ph)

        # 참조 보호 (예: [1], [2,3])
        ref_pattern = r'\[\d+(?:,\s*\d+)*\]'
        ref_matches = re.finditer(ref_pattern, text)
        for i, match in enumerate(ref_matches):
            placeholder = self._make_placeholder("REF", i)
            placeholders[placeholder] = match.group(0)
            text = text.replace(match.group(0), placeholder, 1)

        return text, placeholders

    def _split_protected_segments(self, text: str) -> list:
        # 더 이상 사용하지 않지만 호환성 유지
        return [{"text": text, "protected": False}]
    
    def _restore_terminology(self, text: str, placeholders: dict) -> str:
        """
        플레이스홀더를 원래 용어로 복원 (볼드체 옵션 적용)
        - 모델이 일부 문자를 변형해도 복원하도록 여러 변형 패턴을 허용
        """
        for placeholder, original in placeholders.items():
            replacement = original
            is_term = False
            core = None

            # 플레이스홀더 유형 파악
            m = re.search(r'(TERM|MATH|REF)_\d+', placeholder)
            if m:
                core = m.group(0)
                is_term = m.group(1) == "TERM"

            original_lower = original.lower()
            if self.term_bold and is_term and (original in self.bold_terms or original_lower in self.bold_terms_lower):
                replacement = f"**{original}**"

            variants = set()
            variants.add(placeholder)
            if core:
                variants.update({
                    f"{self.placeholder_prefix}{core}{self.placeholder_suffix}",
                    f"__{core}__",
                    f"_{core}_",
                    core,
                })

            for ph in variants:
                text = text.replace(ph, replacement)
        return text
    
    def set_term_bold(self, enabled: bool):
        """볼드체 처리 ON/OFF"""
        self.term_bold = enabled
    
    def load_terminology(self, terminology_path: str, bold: bool = True):
        """
        학술 용어 사전 로드
        
        파일 형식 (TSV):
        영어용어\t한국어용어
        """
        try:
            with open(terminology_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        self.add_terminology(parts[0], parts[1], bold=bold)
            print(f"용어 사전 로드: {len(self.terminology_dict)}개 항목")
        except Exception as e:
            print(f"용어 사전 로드 실패: {e}")
    
    def add_terminology(self, en_term: str, ko_term: str, bold: bool = True):
        """단일 용어 추가"""
        self.terminology_dict[en_term] = ko_term
        if bold:
            self.bold_terms.add(en_term)
            self.bold_terms_lower.add(en_term.lower())


class MockTranslator(Translator):
    """테스트용 Mock 번역기"""
    
    def __init__(self, **kwargs):
        self.model = "mock"
        self.terminology_dict = {}
    
    def translate(self, text: str) -> str:
        """간단한 Mock 번역"""
        if not text:
            return text
        return f"[번역됨] {text}"
    
    def translate_batch(self, texts: List[str], batch_size: int = 8) -> List[str]:
        return [self.translate(t) for t in texts]
