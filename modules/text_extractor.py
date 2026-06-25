"""
PDF에서 텍스트 조각과 위치 좌표(Bounding Box)를 추출하는 모듈
라이브러리: PyMuPDF (fitz)
"""

from pathlib import Path
from typing import List, Dict, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF가 설치되지 않았습니다. pip install pymupdf")
    raise


class TextExtractor:
    """PDF에서 텍스트와 위치 정보 추출"""
    
    def __init__(self):
        pass
    
    def extract(self, pdf_path: Union[str, Path]) -> List[List[Dict]]:
        """
        PDF에서 텍스트 블록과 위치 정보 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            페이지별 텍스트 블록 리스트
            각 블록: {
                'text': str,
                'bbox': (x0, y0, x1, y1),
                'block_no': int,
                'line_no': int,
                'span_no': int,
                'font': str,
                'size': float,
                'flags': int,
                'page_width': float,
                'page_height': float
            }
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        all_pages = []
        
        for page_num, page in enumerate(doc):
            page_blocks = []
            page_rect = page.rect
            
            # 텍스트를 딕셔너리 형태로 추출 (상세 정보 포함)
            text_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            
            for block_idx, block in enumerate(text_dict.get("blocks", [])):
                # 이미지 블록 스킵 (type=1)
                if block.get("type") == 1:
                    continue
                
                for line_idx, line in enumerate(block.get("lines", [])):
                    for span_idx, span in enumerate(line.get("spans", [])):
                        text = span.get("text", "").strip()
                        
                        if not text:
                            continue
                        
                        bbox = span.get("bbox", (0, 0, 0, 0))
                        
                        page_blocks.append({
                            'text': text,
                            'bbox': bbox,  # (x0, y0, x1, y1) - PyMuPDF 좌표계
                            'block_no': block_idx,
                            'line_no': line_idx,
                            'span_no': span_idx,
                            'font': span.get("font", ""),
                            'size': span.get("size", 0),
                            'flags': span.get("flags", 0),
                            'color': span.get("color", 0),
                            'page_num': page_num,
                            'page_width': page_rect.width,
                            'page_height': page_rect.height
                        })
            
            all_pages.append(page_blocks)
        
        doc.close()
        return all_pages
    
    def extract_by_block(self, pdf_path: Union[str, Path]) -> List[List[Dict]]:
        """
        텍스트를 블록 단위로 추출 (같은 블록의 텍스트를 합침)
        
        Returns:
            페이지별 블록 리스트
            각 블록: {
                'text': str (전체 텍스트),
                'bbox': (x0, y0, x1, y1),
                'block_no': int,
                'page_width': float,
                'page_height': float
            }
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        all_pages = []
        
        for page_num, page in enumerate(doc):
            page_rect = page.rect
            blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
            
            page_blocks = []
            for block in blocks:
                # 이미지 블록 스킵
                if block[6] == 1:
                    continue
                
                text = block[4].strip()
                if not text:
                    continue
                
                page_blocks.append({
                    'text': text,
                    'bbox': (block[0], block[1], block[2], block[3]),
                    'block_no': block[5],
                    'page_num': page_num,
                    'page_width': page_rect.width,
                    'page_height': page_rect.height
                })
            
            all_pages.append(page_blocks)
        
        doc.close()
        return all_pages
    
    def extract_words(self, pdf_path: Union[str, Path]) -> List[List[Dict]]:
        """
        단어 단위로 텍스트 추출
        
        Returns:
            페이지별 단어 리스트
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        all_pages = []
        
        for page_num, page in enumerate(doc):
            page_rect = page.rect
            words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
            
            page_words = []
            for word in words:
                page_words.append({
                    'text': word[4],
                    'bbox': (word[0], word[1], word[2], word[3]),
                    'block_no': word[5],
                    'line_no': word[6],
                    'word_no': word[7],
                    'page_num': page_num,
                    'page_width': page_rect.width,
                    'page_height': page_rect.height
                })
            
            all_pages.append(page_words)
        
        doc.close()
        return all_pages
    
    def extract_word_tuples(self, pdf_path: Union[str, Path]):
        """
        페이지별 (words, page_width, page_height) 반환.
        words는 PyMuPDF get_text("words")의 원형 튜플 리스트
        (x0, y0, x1, y1, word, block_no, line_no, word_no).
        """
        pdf_path = Path(pdf_path)
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append((page.get_text("words"), page.rect.width, page.rect.height))
        doc.close()
        return pages

    def get_page_info(self, pdf_path: Union[str, Path]) -> List[Dict]:
        """페이지 정보 반환"""
        doc = fitz.open(str(pdf_path))
        pages_info = []
        
        for page in doc:
            rect = page.rect
            pages_info.append({
                'width': rect.width,
                'height': rect.height,
                'rotation': page.rotation
            })
        
        doc.close()
        return pages_info
