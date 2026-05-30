"""
PDF를 고해상도 이미지로 변환하는 모듈
라이브러리: pdf2image (poppler 기반)
"""

from pathlib import Path
from typing import List, Union
from PIL import Image

try:
    from pdf2image import convert_from_path
except ImportError:
    print("pdf2image가 설치되지 않았습니다. pip install pdf2image")
    raise


class PDFConverter:
    """PDF를 고해상도 이미지로 변환"""
    
    def __init__(self, dpi: int = 300):
        """
        Args:
            dpi: 이미지 해상도 (기본값: 300)
        """
        self.dpi = dpi
        
    def convert(self, pdf_path: Union[str, Path]) -> List[Image.Image]:
        """
        PDF를 이미지 리스트로 변환
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            PIL Image 객체 리스트 (페이지별)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        images = convert_from_path(
            str(pdf_path),
            dpi=self.dpi,
            fmt='RGB'
        )
        
        return images
    
    def convert_and_save(
        self, 
        pdf_path: Union[str, Path], 
        output_dir: Union[str, Path],
        format: str = 'png'
    ) -> List[Path]:
        """
        PDF를 이미지로 변환하고 파일로 저장
        
        Args:
            pdf_path: PDF 파일 경로
            output_dir: 출력 디렉토리
            format: 이미지 포맷 (png, jpg 등)
            
        Returns:
            저장된 이미지 파일 경로 리스트
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = self.convert(pdf_path)
        
        saved_paths = []
        for i, image in enumerate(images):
            output_path = output_dir / f"{pdf_path.stem}_page_{i+1:03d}.{format}"
            image.save(str(output_path), format.upper())
            saved_paths.append(output_path)
            
        return saved_paths
    
    def get_page_count(self, pdf_path: Union[str, Path]) -> int:
        """PDF 페이지 수 반환"""
        from pdf2image.pdf2image import pdfinfo_from_path
        
        info = pdfinfo_from_path(str(pdf_path))
        return info['Pages']
