"""
감지된 영역에서 원본 이미지를 추출하는 모듈
라이브러리: Pillow
입력: 그림, 표, 수식, 캡션의 이미지와 위치좌표 (Bounding Box)
출력: 위치좌표에 해당하는 이미지
"""

from pathlib import Path
from typing import List, Dict, Union, Tuple
from PIL import Image


class ImageExtractor:
    """감지된 영역에서 이미지 추출"""
    
    def __init__(self, padding: int = 0):
        """
        Args:
            padding: 바운딩 박스 주변 여백 (픽셀)
        """
        self.padding = padding
    
    def extract(
        self, 
        images: List[Image.Image], 
        detections: List[List[Dict]]
    ) -> List[List[Dict]]:
        """
        감지된 영역의 이미지 추출
        
        Args:
            images: 페이지 이미지 리스트
            detections: 페이지별 감지 결과
            
        Returns:
            페이지별 추출된 이미지 정보
            각 항목: {
                'image': PIL.Image,
                'bbox': (x1, y1, x2, y2),
                'class_name': str,
                'page_num': int
            }
        """
        all_extractions = []
        
        for page_num, (image, page_detections) in enumerate(zip(images, detections)):
            page_extractions = []
            img_width, img_height = image.size
            
            for detection in page_detections:
                bbox = detection['bbox']
                
                # 패딩 적용 (이미지 경계 내로 제한)
                x1 = max(0, int(bbox[0]) - self.padding)
                y1 = max(0, int(bbox[1]) - self.padding)
                x2 = min(img_width, int(bbox[2]) + self.padding)
                y2 = min(img_height, int(bbox[3]) + self.padding)
                
                # 영역 크롭
                cropped = image.crop((x1, y1, x2, y2))
                
                page_extractions.append({
                    'image': cropped,
                    'bbox': (x1, y1, x2, y2),
                    'bbox_original': bbox,
                    'class_name': detection.get('class_name', 'unknown'),
                    'class_id': detection.get('class_id', -1),
                    'confidence': detection.get('confidence', 0),
                    'page_num': page_num,
                    'image_size': (img_width, img_height)
                })
            
            all_extractions.append(page_extractions)
        
        return all_extractions
    
    def extract_single(
        self, 
        image: Image.Image, 
        bbox: Tuple[float, float, float, float]
    ) -> Image.Image:
        """
        단일 영역 추출
        
        Args:
            image: 원본 이미지
            bbox: (x1, y1, x2, y2) 좌표
            
        Returns:
            크롭된 이미지
        """
        img_width, img_height = image.size
        
        x1 = max(0, int(bbox[0]) - self.padding)
        y1 = max(0, int(bbox[1]) - self.padding)
        x2 = min(img_width, int(bbox[2]) + self.padding)
        y2 = min(img_height, int(bbox[3]) + self.padding)
        
        return image.crop((x1, y1, x2, y2))
    
    def extract_and_save(
        self,
        images: List[Image.Image],
        detections: List[List[Dict]],
        output_dir: Union[str, Path],
        format: str = 'png'
    ) -> List[List[Path]]:
        """
        추출된 이미지를 파일로 저장
        
        Args:
            images: 페이지 이미지 리스트
            detections: 페이지별 감지 결과
            output_dir: 출력 디렉토리
            format: 이미지 포맷
            
        Returns:
            저장된 파일 경로 리스트
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        extractions = self.extract(images, detections)
        all_paths = []
        
        for page_num, page_extractions in enumerate(extractions):
            page_paths = []
            
            for idx, extraction in enumerate(page_extractions):
                class_name = extraction['class_name']
                filename = f"page{page_num+1:03d}_{class_name}_{idx+1:03d}.{format}"
                filepath = output_dir / filename
                
                extraction['image'].save(str(filepath), format.upper())
                page_paths.append(filepath)
            
            all_paths.append(page_paths)
        
        return all_paths
    
    def get_images_by_class(
        self,
        extractions: List[List[Dict]],
        class_name: str
    ) -> List[Dict]:
        """특정 클래스의 추출 이미지만 반환"""
        results = []
        for page_extractions in extractions:
            for extraction in page_extractions:
                if extraction['class_name'] == class_name:
                    results.append(extraction)
        return results
    
    def resize_extracted(
        self,
        extractions: List[List[Dict]],
        max_size: Tuple[int, int] = (800, 800)
    ) -> List[List[Dict]]:
        """추출된 이미지 리사이즈 (OCR 최적화용)"""
        for page_extractions in extractions:
            for extraction in page_extractions:
                img = extraction['image']
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                extraction['image'] = img
        return extractions
