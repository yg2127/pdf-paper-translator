"""
한국어 번역본 PDF를 생성하는 모듈
라이브러리: ReportLab

입력:
1. 이미지의 위치좌표
2. 이미지
3. 한국어로 번역한 번역본
4. 원문 영어 텍스트의 위치좌표

출력:
한국어 번역본 PDF
"""

from pathlib import Path
from typing import List, Dict, Union, Tuple
import re
from PIL import Image
import io

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import inch, mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader
except ImportError:
    print("reportlab이 설치되지 않았습니다. pip install reportlab")
    raise


class PDFGenerator:
    """한국어 번역본 PDF 생성"""
    
    def __init__(
        self,
        font_path: str = None,
        font_name: str = "Korean",
        bold_font_path: str = None,
        default_font_size: int = 10,
        page_size: tuple = None
    ):
        """
        Args:
            font_path: 한글 폰트 경로 (TTF 파일)
            font_name: 등록할 폰트 이름
            bold_font_path: 볼드체 폰트 경로 (없으면 일반 폰트로 대체)
            default_font_size: 기본 폰트 크기
            page_size: 페이지 크기 (letter, A4 등)
        """
        self.font_path = font_path
        self.font_name = font_name
        self.bold_font_path = bold_font_path
        self.bold_font_name = f"{font_name}-Bold"
        self.default_font_size = default_font_size
        # page_size가 None이면 입력 이미지 크기(픽셀)를 그대로 사용해 비율 불일치로 인한 겹침을 방지
        self.page_size = page_size
        self.has_bold_font = False
        
        self._register_font()
    
    def _register_font(self):
        """한글 폰트 등록"""
        if self.font_path and Path(self.font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont(self.font_name, self.font_path))
                print(f"폰트 등록 완료: {self.font_name}")
            except Exception as e:
                print(f"폰트 등록 실패: {e}")
                self.font_name = "Helvetica"
        else:
            print(f"폰트 파일을 찾을 수 없습니다: {self.font_path}")
            self.font_name = "Helvetica"

        # 볼드 폰트 등록 (선택)
        if self.bold_font_path and Path(self.bold_font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont(self.bold_font_name, self.bold_font_path))
                self.has_bold_font = True
                print(f"볼드 폰트 등록 완료: {self.bold_font_name}")
            except Exception as e:
                print(f"볼드 폰트 등록 실패: {e}")
                self.bold_font_name = self.font_name
                self.has_bold_font = False
        else:
            # 볼드 폰트가 없으면 일반 폰트 재사용
            self.bold_font_name = self.font_name
            self.has_bold_font = False
    
    def generate(
        self,
        output_path: Union[str, Path],
        images: List[Image.Image],
        image_positions: List[List[Dict]],
        translated_texts: List[List[str]],
        text_positions: List[List[Tuple]],
        original_pdf_path: Union[str, Path] = None,
        draw_background: bool = True,
        text_bg_margin: int = 2
    ) -> str:
        """
        한국어 번역본 PDF 생성
        
        Args:
            output_path: 출력 PDF 경로
            images: 페이지 이미지 리스트
            image_positions: 추출된 이미지 정보 (그림, 표 등)
            translated_texts: 페이지별 번역된 텍스트 리스트
            text_positions: 페이지별 텍스트 위치 리스트
            original_pdf_path: 원본 PDF 경로 (메타데이터용)
            draw_background: 원본 페이지 이미지를 배경으로 그릴지 여부
            text_bg_margin: 텍스트를 덮는 흰색 배경 여백 (픽셀 단위, 이미지 좌표 기준)
            
        Returns:
            생성된 PDF 경로
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 페이지 크기 결정: 지정된 page_size가 없으면 각 페이지 이미지 크기를 그대로 사용
        initial_page_size = self.page_size or images[0].size

        # 캔버스 생성
        c = canvas.Canvas(str(output_path), pagesize=initial_page_size)
        
        for page_num, (image, page_images, page_texts, page_positions) in enumerate(
            zip(images, image_positions, translated_texts, text_positions)
        ):
            # 페이지마다 크기가 다를 수 있으므로 동적으로 설정
            current_page_size = self.page_size or image.size
            c.setPageSize(current_page_size)
            page_width, page_height = current_page_size

            # 배경 처리
            if draw_background:
                # 원본 페이지 이미지를 그대로 그리기
                self._draw_background(c, image, page_width, page_height)
            else:
                # 흰색 배경으로 초기화
                c.setFillColorRGB(1, 1, 1)
                c.rect(0, 0, page_width, page_height, fill=True, stroke=False)
            
            # 그림/표 등 이미지 요소 유지 (텍스트보다 먼저 그려서 텍스트가 최상단에 오도록)
            for img_info in page_images:
                self._draw_image_element(
                    c,
                    img_info,
                    image.size,
                    (page_width, page_height)
                )

            # 텍스트 영역에 흰색 배경 + 번역 텍스트 추가
            self._draw_translated_texts(
                c, 
                page_texts, 
                page_positions, 
                image.size,
                (page_width, page_height),
                text_bg_margin=text_bg_margin
            )
            
            # 다음 페이지
            c.showPage()
        
        # PDF 저장
        c.save()
        
        return str(output_path)
    
    def _draw_background(
        self,
        c: canvas.Canvas,
        image: Image.Image,
        page_width: float,
        page_height: float
    ):
        """배경 이미지 그리기"""
        # PIL Image를 ReportLab ImageReader로 변환
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        img_reader = ImageReader(img_buffer)
        
        # 페이지 전체에 이미지 그리기
        c.drawImage(
            img_reader,
            0, 0,
            width=page_width,
            height=page_height,
            preserveAspectRatio=True
        )
    
    def _draw_translated_texts(
        self,
        c: canvas.Canvas,
        texts: List[str],
        positions: List[Tuple],
        image_size: Tuple[int, int],
        page_size: Tuple[float, float],
        text_bg_margin: int = 2
    ):
        """번역된 텍스트 그리기"""
        img_width, img_height = image_size
        page_width, page_height = page_size

        # 스케일 계산 (이미지 픽셀 좌표 -> PDF 포인트 좌표)
        scale_x = page_width / img_width
        scale_y = page_height / img_height

        c.setFont(self.font_name, self.default_font_size)

        # 디버그: 처음 3개 텍스트 출력
        print(f"\n[PDF 생성 디버그] 그리기 시작:")
        print(f"  폰트: {self.font_name}, 크기: {self.default_font_size}")
        print(f"  이미지 크기: {img_width}x{img_height}, PDF 크기: {page_width}x{page_height}")
        print(f"  총 텍스트: {len(texts)}개")
        for i, text in enumerate(texts[:3]):
            print(f"  [{i+1}] {text[:50]}...")

        for text, bbox in zip(texts, positions):
            if not text or not bbox:
                continue

            x0, y0, x1, y1 = bbox

            # YOLO/이미지 좌표(pixel)를 ReportLab 좌표(point)로 변환
            pdf_x = x0 * scale_x
            pdf_y = page_height - (y1 * scale_y)  # y 좌표 뒤집기
            box_width = (x1 - x0) * scale_x
            box_height = (y1 - y0) * scale_y

            # 마진 추가 (흰색 배경이 텍스트를 완전히 덮도록)
            margin = max(text_bg_margin, 0)
            bg_x = pdf_x - margin
            bg_y = pdf_y - margin
            bg_width = box_width + margin * 2
            bg_height = box_height + margin * 2

            # 흰색 배경 그리기 (텍스트 영역 완전히 덮기)
            c.setFillColorRGB(1, 1, 1)  # 흰색
            c.rect(bg_x, bg_y, bg_width, bg_height, fill=True, stroke=False)
            
            # 텍스트 그리기
            c.setFillColorRGB(0, 0, 0)  # 검정색

            # 텍스트가 박스에 맞도록 폰트 크기 조정 (줄바꿈 고려)
            font_size = self._calculate_font_size(text, box_width, box_height)
            c.setFont(self.font_name, font_size)

            # 텍스트 줄바꿈 처리
            try:
                self._draw_text_in_box(c, text, pdf_x, pdf_y, box_width, box_height, font_size)
            except Exception as e:
                print(f"  텍스트 그리기 실패: {text[:30]}... - {e}")
    
    def _draw_image_element(
        self,
        c: canvas.Canvas,
        img_info: Dict,
        image_size: Tuple[int, int],
        page_size: Tuple[float, float]
    ):
        """그림/표 등 이미지 요소 그리기"""
        if 'image' not in img_info:
            return
        
        img = img_info['image']
        bbox = img_info.get('bbox', (0, 0, 100, 100))
        
        img_width, img_height = image_size
        page_width, page_height = page_size
        
        # 스케일 계산
        scale_x = page_width / img_width
        scale_y = page_height / img_height
        
        x0, y0, x1, y1 = bbox
        
        # PDF 좌표로 변환
        pdf_x = x0 * scale_x
        pdf_y = page_height - (y1 * scale_y)
        elem_width = (x1 - x0) * scale_x
        elem_height = (y1 - y0) * scale_y
        
        # 이미지 그리기
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        img_reader = ImageReader(img_buffer)
        c.drawImage(
            img_reader,
            pdf_x, pdf_y,
            width=elem_width,
            height=elem_height
        )
    
    def _calculate_font_size(
        self,
        text: str,
        box_width: float,
        box_height: float,
        min_size: int = 8,
        max_size: int = 40
    ) -> int:
        """박스 크기에 맞는 폰트 크기 계산 (실제 폭 기반, 볼드 포함)"""
        preferred_max = min(max_size, max(self.default_font_size + 10, min_size))
        preferred_min = min_size
        line_height_ratio = 1.25

        for size in range(preferred_max, preferred_min - 1, -1):
            lines = self._wrap_text_with_bold(text, box_width, size)
            line_height = size * line_height_ratio
            needed_height = len(lines) * line_height
            if needed_height <= box_height:
                return size
        return preferred_min

    def _string_width(self, text: str, font_size: int, bold: bool = False) -> float:
        from reportlab.pdfbase.pdfmetrics import stringWidth
        font = self.bold_font_name if bold and self.has_bold_font else self.font_name
        return stringWidth(text, font, font_size)

    def _parse_bold_segments(self, text: str) -> list:
        """**term** 마크다운을 세그먼트 리스트로 분해"""
        segments = []
        idx = 0
        for m in re.finditer(r"\*\*(.+?)\*\*", text):
            if m.start() > idx:
                segments.append((text[idx:m.start()], False))
            segments.append((m.group(1), True))
            idx = m.end()
        if idx < len(text):
            segments.append((text[idx:], False))
        if not segments:
            segments.append((text, False))
        return segments

    def _tokenize_segments(self, segments: list) -> list:
        """세그먼트를 공백 기준 토큰으로 분해"""
        tokens = []
        for seg_text, is_bold in segments:
            parts = re.split(r"(\s+)", seg_text)
            for p in parts:
                if p == "":
                    continue
                tokens.append((p, is_bold))
        return tokens

    def _wrap_text_with_bold(self, text: str, max_width: float, font_size: int) -> list:
        """
        볼드 마크업을 포함한 줄바꿈 결과 반환
        Returns: [ [(token, is_bold), ...], ... ]
        """
        segments = self._parse_bold_segments(text.replace("\n", " "))
        tokens = self._tokenize_segments(segments)
        lines = []
        current = []
        current_width = 0.0

        for token, is_bold in tokens:
            token_width = self._string_width(token, font_size, is_bold)
            if token.isspace() and not current:
                continue  # 줄 시작 공백 제거

            # 한 단어가 한 줄보다 길면 분할
            if token_width > max_width:
                split_tokens = self._split_long_token(token, max_width, font_size, is_bold)
            else:
                split_tokens = [(token, is_bold)]

            for split_token, split_bold in split_tokens:
                split_width = self._string_width(split_token, font_size, split_bold)
                new_width = current_width + split_width
                if new_width <= max_width or not current:
                    current.append((split_token, split_bold))
                    current_width = new_width
                else:
                    lines.append(current)
                    current = []
                    current_width = 0.0
                    if not split_token.isspace():
                        current.append((split_token, split_bold))
                        current_width = split_width

        if current:
            lines.append(current)

        return lines or [[("", False)]]

    def _split_long_token(self, token: str, max_width: float, font_size: int, is_bold: bool) -> list:
        """한 토큰이 너무 길 때 폭에 맞게 분할"""
        parts = []
        current = ""
        for ch in token:
            candidate = current + ch
            if self._string_width(candidate, font_size, is_bold) <= max_width or not current:
                current = candidate
            else:
                parts.append((current, is_bold))
                current = ch
        if current:
            parts.append((current, is_bold))
        return parts

    def _draw_text_in_box(
        self,
        c: canvas.Canvas,
        text: str,
        x: float,
        y: float,
        width: float,
        height: float,
        font_size: int,
        min_font_size: int = 6
    ):
        """박스 안에 텍스트 그리기 (줄바꿈+볼드, 높이 초과 시 폰트 축소)"""
        size = font_size
        while size > min_font_size:
            lines = self._wrap_text_with_bold(text, width, size)
            line_height = size * 1.25
            needed_height = len(lines) * line_height
            if needed_height <= height:
                break
            size -= 1

        line_height = size * 1.25
        lines = self._wrap_text_with_bold(text, width, size)

        current_y = y + height - line_height
        for i, line in enumerate(lines):
            if current_y < y:
                break
            current_x = x + 2
            for token, is_bold in line:
                if not token:
                    continue
                font_to_use = self.bold_font_name if is_bold and self.has_bold_font else self.font_name
                c.setFont(font_to_use, size)
                try:
                    if is_bold and not self.has_bold_font:
                        # 볼드 폰트가 없으면 두 번 그려서 두껍게 보이도록 보정
                        c.drawString(current_x, current_y, token)
                        c.drawString(current_x + 0.2, current_y, token)
                    else:
                        c.drawString(current_x, current_y, token)
                except Exception as e:
                    print(f"  drawString 실패 (줄 {i+1}): {token[:20]}... - {e}")
                current_x += self._string_width(token, size, is_bold)
            current_y -= line_height
    
