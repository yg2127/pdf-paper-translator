"""
이미지 내 텍스트를 인식하는 OCR 모듈
라이브러리: EasyOCR
입력: 라벨링된 이미지조각
출력: 라벨과 이미지 안의 텍스트
"""

from typing import List, Dict, Union, Tuple
from PIL import Image
import numpy as np

try:
    import easyocr
except ImportError:
    print("easyocr가 설치되지 않았습니다. pip install easyocr")
    easyocr = None


class OCRProcessor:
    """EasyOCR 기반 텍스트 인식"""
    
    def __init__(
        self, 
        languages: List[str] = None,
        gpu: bool = True,
        model_storage_directory: str = None
    ):
        """
        Args:
            languages: 인식 언어 리스트 (예: ['en', 'ko'])
            gpu: GPU 사용 여부
            model_storage_directory: 모델 저장 디렉토리
        """
        self.languages = languages or ['en']
        self.gpu = gpu
        self.reader = None
        self.model_storage_directory = model_storage_directory
        
        self._init_reader()
    
    def _init_reader(self):
        """EasyOCR Reader 초기화"""
        if easyocr is None:
            print("경고: easyocr가 설치되지 않았습니다.")
            return
        
        self.reader = easyocr.Reader(
            self.languages,
            gpu=self.gpu,
            model_storage_directory=self.model_storage_directory
        )
    
    def process(
        self, 
        extracted_images: List[List[Dict]]
    ) -> List[List[Dict]]:
        """
        추출된 이미지들에서 텍스트 인식
        
        Args:
            extracted_images: ImageExtractor에서 반환된 이미지 정보
            
        Returns:
            페이지별 OCR 결과
            각 결과: {
                'text': str,
                'bbox': (x1, y1, x2, y2),  # 원본 이미지 기준 좌표
                'confidence': float,
                'class_name': str,
                'page_num': int
            }
        """
        if self.reader is None:
            print("경고: OCR Reader가 초기화되지 않았습니다.")
            return [[] for _ in extracted_images]
        
        all_results = []
        
        for page_extractions in extracted_images:
            page_results = []
            
            for extraction in page_extractions:
                image = extraction['image']
                base_bbox = extraction['bbox']  # 추출된 영역의 위치
                
                # PIL Image를 numpy 배열로 변환
                img_array = np.array(image)
                
                # OCR 수행
                ocr_results = self.reader.readtext(img_array)
                
                for result in ocr_results:
                    # result: (bbox, text, confidence)
                    # bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    ocr_bbox = result[0]
                    text = result[1]
                    confidence = result[2]
                    
                    # 로컬 좌표를 원본 이미지 좌표로 변환
                    local_x1 = min(p[0] for p in ocr_bbox)
                    local_y1 = min(p[1] for p in ocr_bbox)
                    local_x2 = max(p[0] for p in ocr_bbox)
                    local_y2 = max(p[1] for p in ocr_bbox)
                    
                    # 원본 이미지 기준 좌표
                    global_bbox = (
                        base_bbox[0] + local_x1,
                        base_bbox[1] + local_y1,
                        base_bbox[0] + local_x2,
                        base_bbox[1] + local_y2
                    )
                    
                    page_results.append({
                        'text': text,
                        'bbox': global_bbox,
                        'bbox_local': (local_x1, local_y1, local_x2, local_y2),
                        'confidence': confidence,
                        'class_name': extraction.get('class_name', 'unknown'),
                        'page_num': extraction.get('page_num', 0),
                        'source_bbox': base_bbox
                    })
            
            all_results.append(page_results)
        
        return all_results
    
    def process_single_image(
        self, 
        image: Union[Image.Image, np.ndarray],
        detail: int = 1
    ) -> List[Dict]:
        """
        단일 이미지에서 OCR 수행
        
        Args:
            image: PIL Image 또는 numpy 배열
            detail: 상세 수준 (0 또는 1)
            
        Returns:
            OCR 결과 리스트
        """
        if self.reader is None:
            return []
        
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        results = self.reader.readtext(image, detail=detail)
        
        ocr_results = []
        for result in results:
            if detail == 1:
                ocr_bbox = result[0]
                text = result[1]
                confidence = result[2]
                
                x1 = min(p[0] for p in ocr_bbox)
                y1 = min(p[1] for p in ocr_bbox)
                x2 = max(p[0] for p in ocr_bbox)
                y2 = max(p[1] for p in ocr_bbox)
                
                ocr_results.append({
                    'text': text,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })
            else:
                ocr_results.append({'text': result})
        
        return ocr_results
    
    def get_text_only(self, ocr_results: List[List[Dict]]) -> List[str]:
        """OCR 결과에서 텍스트만 추출"""
        texts = []
        for page_results in ocr_results:
            page_text = ' '.join([r['text'] for r in page_results])
            texts.append(page_text)
        return texts
    
    def filter_by_confidence(
        self, 
        ocr_results: List[List[Dict]], 
        min_confidence: float = 0.5
    ) -> List[List[Dict]]:
        """신뢰도 기준 필터링"""
        filtered = []
        for page_results in ocr_results:
            page_filtered = [
                r for r in page_results 
                if r['confidence'] >= min_confidence
            ]
            filtered.append(page_filtered)
        return filtered


class MockOCRProcessor(OCRProcessor):
    """테스트용 Mock OCR"""
    
    def __init__(self, **kwargs):
        self.languages = kwargs.get('languages', ['en'])
        self.reader = "mock"
    
    def process(self, extracted_images: List[List[Dict]]) -> List[List[Dict]]:
        """더미 OCR 결과 반환"""
        all_results = []
        
        for page_extractions in extracted_images:
            page_results = []
            
            for extraction in page_extractions:
                base_bbox = extraction['bbox']
                
                # 더미 텍스트 생성
                page_results.append({
                    'text': 'Sample OCR Text',
                    'bbox': base_bbox,
                    'confidence': 0.95,
                    'class_name': extraction.get('class_name', 'unknown'),
                    'page_num': extraction.get('page_num', 0)
                })
            
            all_results.append(page_results)
        
        return all_results
