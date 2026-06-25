"""
논문 번역기 파이프라인 (Paper Translator Pipeline)

기본 파이프라인 흐름 (process()):
1. PDF → 이미지 변환 (pdf2image, DPI 300)
2. YOLOv11 레이아웃 감지 → Text / 비Text(그림·표·수식·캡션·제목) 분리
3. PyMuPDF로 YOLO Text 박스와 겹치는 단어만 수집 (좌표 변환 + 패딩)
4. Title/Section-header 등에서 학술 용어 추출 → 번역 시 원문 보호(+ 볼드)
5. 영어→한국어 번역 (기본 MarianMT, 환경변수로 파인튜닝 13B 모델로 전환)
6. 비Text 영역 보존 + ReportLab로 한국어 PDF 생성 (배경 유지/제거 2종)

선택(Optional) 모듈:
- EasyOCR(modules/ocr_processor.py): 그림/이미지 내부 텍스트를 인식해
  term_extractor.extract_from_ocr_results()로 용어 추출에 활용 가능.
  기본 흐름은 PyMuPDF 텍스트를 사용하며 OCR은 기본적으로 비활성화 상태.
- 모델 학습 코드(YOLO·Tower 13B/7B LoRA)는 training/ 디렉토리 참조.
"""

import os
import sys
import time
from pathlib import Path

# 프로젝트 경로 설정
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.pdf_converter import PDFConverter
from modules.yolo_detector import YOLODetector
from modules.image_extractor import ImageExtractor
from modules.translator import Translator
from modules.pdf_generator import PDFGenerator
from modules.coordinate_transformer import CoordinateTransformer
from modules.text_extractor import TextExtractor
from modules.term_extractor import TermExtractor
from config import TERM_EXTRACT_CLASSES


class PaperTranslatorPipeline:
    """논문 번역 파이프라인 메인 클래스"""

    def __init__(self, config: dict):
        """
        Args:
            config: 설정 딕셔너리 (config.py의 DEFAULT_CONFIG 사용)
        """
        if config is None:
            raise ValueError(
                "config는 필수입니다. config.py의 DEFAULT_CONFIG를 사용하세요."
            )
        self.config = config

        # 모듈 초기화
        self.pdf_converter = PDFConverter(dpi=self.config["dpi"])
        self.yolo_detector = YOLODetector(model_path=self.config["yolo_model_path"])
        self.image_extractor = ImageExtractor()
        self.translator = Translator(
            model_name=self.config["translation_model"],
            term_bold=self.config.get("term_bold", True),
            max_length=self.config.get("translation_max_length", 512),
            use_4bit=self.config.get("translation_use_4bit", True),
            model_type=self.config.get("translation_model_type", None),
        )
        # 베이스 용어 사전 로드
        if self.config.get("terminology_dict_path"):
            terminology_path = self.config["terminology_dict_path"]
            if Path(terminology_path).exists():
                print(f"베이스 용어 사전 로드 중: {terminology_path}")
                self.translator.load_terminology(
                    terminology_path, bold=False
                )  # 기본 사전은 볼드 처리 안 함
            else:
                print(f"경고: 용어 사전 파일을 찾을 수 없습니다: {terminology_path}")

        self.pdf_generator = PDFGenerator(
            font_path=self.config["korean_font_path"],
            bold_font_path=self.config.get("korean_bold_font_path"),
            default_font_size=self.config.get("default_font_size", 10),
        )
        self.coord_transformer = CoordinateTransformer()
        self.text_extractor = TextExtractor()

        # 용어 추출기: 영어 용어 유지 + 볼드 처리용
        self.term_extractor = TermExtractor(
            extract_from_fte=True, extract_from_title=True
        )

    def process(self, pdf_path: str, output_path: str = None) -> str:
        """
        PDF를 YOLO Text 박스 기준으로만 번역합니다.
        - YOLO가 Text로 판정한 영역: OCR 후 번역하여 덮어쓰기
        - Text가 아닌 모든 영역: 원본 그대로 보존
        """
        start_time = time.time()

        pdf_path = Path(pdf_path)
        if output_path is None:
            output_path = Path(self.config["output_dir"]) / f"{pdf_path.stem}_ko.pdf"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        step_times = {}

        # [1/5] PDF → 이미지
        step_start = time.time()
        print(f"[1/5] PDF를 이미지로 변환 중... ({pdf_path})")
        images = self.pdf_converter.convert(pdf_path)
        step_times["1_pdf_to_image"] = time.time() - step_start
        print(f"    ✓ 완료 ({step_times['1_pdf_to_image']:.1f}초)")

        # [2/5] YOLO 감지
        step_start = time.time()
        print(f"[2/5] YOLO 감지 중...")
        detections = self.yolo_detector.detect(images)
        step_times["2_yolo_detect"] = time.time() - step_start
        print(f"    ✓ 완료 ({step_times['2_yolo_detect']:.1f}초)")

        # [3/5] 감지 분리
        step_start = time.time()
        print(f"[3/5] 텍스트/비텍스트 분리 중...")
        text_detections = []
        preserve_detections = []
        for page in detections:
            text_page = []
            preserve_page = []
            for det in page:
                cls = det.get("class_name", "").lower()
                if cls == "text":
                    text_page.append(det)
                else:
                    preserve_page.append(det)
            text_detections.append(text_page)
            preserve_detections.append(preserve_page)
        step_times["3_split"] = time.time() - step_start
        print(
            f"    ✓ 완료 ({step_times['3_split']:.1f}초) [텍스트 박스 {sum(len(p) for p in text_detections)}개]"
        )

        # [4/5] 번역 텍스트 구성 (PyMuPDF words → YOLO Text 박스와 겹치는 단어만)
        step_start = time.time()
        print(f"[4/5] 텍스트 수집 및 번역 중 (PyMuPDF)")
        # 용어 사전 초기화 후 클래스 기반 용어 추출
        known_terms = set(self.translator.terminology_dict.keys())
        if self.term_extractor:
            self.term_extractor.clear()
            known_terms = set(self.translator.terminology_dict.keys())
            self._extract_terms_from_config_classes(
                pdf_path=pdf_path,
                detections=detections,
                images=images,
                known_terms=known_terms,
            )

        translated = self._translate_from_pymupdf_words(
            pdf_path, text_detections, images, known_terms
        )
        step_times["4_translate"] = time.time() - step_start
        print(f"    ✓ 완료 ({step_times['4_translate']:.1f}초)")

        # 비텍스트 영역 추출(보존용)
        preserved_images = self.image_extractor.extract(images, preserve_detections)

        # [5/5] PDF 생성
        step_start = time.time()
        print(f"[5/5] PDF 생성 중...")
        self._generate_pdfs(
            images=images,
            preserved_images=preserved_images,
            translated_texts=translated["texts"],
            text_positions=translated["positions"],
            pdf_path=pdf_path,
            output_path=output_path,
        )
        step_times["5_generate"] = time.time() - step_start
        print(f"    ✓ 완료 ({step_times['5_generate']:.1f}초)")

        # 용어 사전 저장
        if self.term_extractor and self.config.get("save_extracted_terms", True):
            terms_path = self.config.get("terms_output_path", None)
            terms_path = (
                Path(terms_path)
                if terms_path
                else Path(self.config["output_dir"]) / "extracted_terms.tsv"
            )
            terms_output = terms_path.parent / f"{pdf_path.stem}_terms.tsv"
            self.term_extractor.save_terms(str(terms_output))

        # 종료 시간 기록 및 소요 시간 계산
        end_time = time.time()
        elapsed_time = end_time - start_time

        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = elapsed_time % 60

        print(f"\n완료! 출력 파일: {output_path}")
        print(
            f"총 소요 시간: {hours:02d}:{minutes:02d}:{seconds:05.2f} ({elapsed_time:.2f}초)"
        )

        print(f"\n=== 단계별 소요 시간 ===")
        print(f"  [1] PDF → 이미지: {step_times['1_pdf_to_image']:.1f}초")
        print(f"  [2] YOLO 감지: {step_times['2_yolo_detect']:.1f}초")
        print(f"  [3] 텍스트/비텍스트 분리: {step_times['3_split']:.1f}초")
        print(f"  [4] 텍스트 수집+번역: {step_times['4_translate']:.1f}초")
        print(f"  [5] PDF 생성: {step_times['5_generate']:.1f}초")
        print(f"=" * 30)
        return str(output_path)

    def _translate_from_pymupdf_words(
        self,
        pdf_path: Path,
        text_detections: list,
        images: list,
        known_terms: set = None,
    ) -> dict:
        """
        PyMuPDF 단어 정보를 사용해 YOLO Text 박스와 겹치는 텍스트만 번역
        - YOLO 박스를 소폭 확장해 단어 누락을 줄임
        - 블록/라인 정보로 정렬 안정화
        - 하이픈/개행 정리
        Returns: {'texts': [...], 'positions': [...]} (페이지별 리스트)
        """
        pages_words = self.text_extractor.extract_word_tuples(pdf_path)
        translated_texts = []
        text_positions = []
        count = 0
        total_boxes = sum(len(p) for p in text_detections)
        known_terms = known_terms or set(self.translator.terminology_dict.keys())

        for page_idx, page_dets in enumerate(text_detections):
            # words: (x0,y0,x1,y1,word,block,line,word_no)
            words, page_w, page_h = pages_words[page_idx]

            page_texts = []
            page_positions = []

            img_w, img_h = images[page_idx].size

            for det in page_dets:
                bbox_img = det.get("bbox")
                if not bbox_img:
                    continue

                # YOLO → PyMuPDF 좌표 변환 + 소폭 패딩(5%)으로 단어 누락 줄이기
                bbox_pdf = self.coord_transformer.yolo_to_pymupdf(
                    bbox_img, page_w, page_h, img_w, img_h
                )
                bbox_pdf = self._expand_box(bbox_pdf, padding_ratio=0.05)

                # 겹치는 단어 수집 (block, line, y, x 순 정렬)
                collected = []
                for w in words:
                    wx0, wy0, wx1, wy1, wtext, wblock, wline, _ = w
                    if self.coord_transformer.intersects(bbox_pdf, (wx0, wy0, wx1, wy1)):
                        collected.append((wblock, wline, wy0, wx0, wtext))

                if not collected:
                    continue

                collected.sort()  # block, line, y, x 순
                text_raw = " ".join([w[4] for w in collected])
                text_clean = self._clean_text(text_raw)

                # Text 박스에서는 별도 용어 추출을 하지 않음 (TERM_EXTRACT_CLASSES 기반은 별도 패스에서 처리)

                translated = self.translator.translate(text_clean)
                page_texts.append(translated)
                page_positions.append(bbox_img)
                count += 1

                if count <= 3:
                    print(f"\n    [번역 디버그 {count}]")
                    print(f"      원문: {text_clean[:80]}...")
                    print(f"      번역: {translated[:80]}...")

            translated_texts.append(page_texts)
            text_positions.append(page_positions)

        # 추출된 용어 통계 출력
        if self.term_extractor:
            preserve_terms = [
                t for t in self.term_extractor.dictionary.terms.values() if t.preserve
            ]
            title_terms = [
                t for t in preserve_terms if t.source_type.lower() == "title"
            ]
            print(
                f"    용어 등록: {len(preserve_terms)}개 (Title 볼드 {len(title_terms)}개)"
            )

        print(f"    총 번역 완료: {count}/{total_boxes}")
        return {"texts": translated_texts, "positions": text_positions}

    def _extract_terms_from_config_classes(
        self, pdf_path: Path, detections: list, images: list, known_terms: set
    ):
        """
        TERM_EXTRACT_CLASSES에 정의된 라벨의 박스에서만 용어 추출 (번역 보호용)
        - 박스는 번역/오버레이하지 않고 원본 유지
        - 추출된 용어는 번역 시 보호되며, Title 클래스만 볼드 처리
        """
        if not self.term_extractor:
            return

        classes_set = set(
            c.lower()
            for c in self.config.get("term_extract_classes", TERM_EXTRACT_CLASSES)
        )
        if not classes_set:
            return

        pages_words = self.text_extractor.extract_word_tuples(pdf_path)

        for page_idx, page_dets in enumerate(detections):
            words, page_w, page_h = pages_words[page_idx]

            img_w, img_h = images[page_idx].size

            for det in page_dets:
                class_name = det.get("class_name", "").lower()
                if class_name not in classes_set:
                    continue
                bbox_img = det.get("bbox")
                if not bbox_img:
                    continue

                bbox_pdf = self.coord_transformer.yolo_to_pymupdf(
                    bbox_img, page_w, page_h, img_w, img_h
                )
                bbox_pdf = self._expand_box(bbox_pdf, padding_ratio=0.05)

                collected = []
                for w in words:
                    wx0, wy0, wx1, wy1, wtext, wblock, wline, _ = w
                    if self.coord_transformer.intersects(bbox_pdf, (wx0, wy0, wx1, wy1)):
                        collected.append((wblock, wline, wy0, wx0, wtext))

                if not collected:
                    continue

                collected.sort()
                text_raw = " ".join([w[4] for w in collected])
                text_clean = self._clean_text(text_raw)

                self.term_extractor.extract_from_text(
                    text_clean, source_type=class_name, page_num=page_idx + 1
                )

                # 새로 추출된 용어를 번역기에 등록 (Title만 볼드)
                for term_obj in self.term_extractor.dictionary.terms.values():
                    if not term_obj.preserve:
                        continue
                    term = term_obj.term
                    if term in known_terms:
                        continue
                    is_title = term_obj.source_type.lower() == "title"
                    self.translator.add_terminology(term, term, bold=is_title)
                    known_terms.add(term)

    def _expand_box(self, box, padding_ratio=0.05):
        x0, y0, x1, y1 = box
        w = x1 - x0
        h = y1 - y0
        pad_x = w * padding_ratio
        pad_y = h * padding_ratio
        return (x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y)

    def _clean_text(self, text: str) -> str:
        # 줄바꿈/중복 공백 정리, 하이픈 줄바꿈 연결
        text = text.replace("-\n", "")
        text = text.replace("\n", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    def _generate_pdfs(
        self,
        images,
        preserved_images,
        translated_texts,
        text_positions,
        pdf_path,
        output_path,
    ):
        """배경 유지/비유지 두 가지 버전 생성"""
        self.pdf_generator.generate(
            output_path=output_path,
            images=images,
            image_positions=preserved_images,
            translated_texts=translated_texts,
            text_positions=text_positions,
            original_pdf_path=pdf_path,
            draw_background=self.config.get("pdf_draw_background", True),
            text_bg_margin=self.config.get("pdf_text_bg_margin", 2),
        )

        if self.config.get("generate_no_background_variant", False):
            raw_suffix = self.config.get("no_background_suffix", "_nobg")
            suffix = (
                raw_suffix[:-4] if raw_suffix.lower().endswith(".pdf") else raw_suffix
            )
            no_bg_output = output_path.with_name(f"{output_path.stem}{suffix}.pdf")
            print(f"    → 배경 없는 PDF 생성: {no_bg_output}")
            self.pdf_generator.generate(
                output_path=no_bg_output,
                images=images,
                image_positions=preserved_images,
                translated_texts=translated_texts,
                text_positions=text_positions,
                original_pdf_path=pdf_path,
                draw_background=False,
                text_bg_margin=self.config.get("pdf_text_bg_margin", 2),
            )

    def _is_skip_class(self, class_name: str) -> bool:
        """번역/오버레이를 건너뛰어야 하는 클래스인지 확인 (대소문자 무시)"""
        if not class_name:
            return False
        return class_name.lower() in self.skip_classes


def main():
    """메인 실행 함수 - config.py에서 모든 설정을 가져옵니다"""
    from config import DEFAULT_CONFIG, ensure_directories

    # 디렉토리 생성
    ensure_directories()

    # config.py에서 설정 로드
    config = DEFAULT_CONFIG.copy()

    print("=" * 80)
    print("논문 번역 파이프라인")
    print("=" * 80)
    print(f"입력 PDF: {config['input_pdf']}")
    print(f"출력 PDF: {config['output_pdf'] or '자동 생성'}")
    print(f"번역 모델: {config['translation_model']}")
    print(f"용어 사전: {config.get('terminology_dict_path', 'None')}")
    print("=" * 80)

    # 파이프라인 실행
    pipeline = PaperTranslatorPipeline(config)
    pipeline.process(config["input_pdf"], config["output_pdf"])


if __name__ == "__main__":
    main()
