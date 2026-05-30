"""
좌표계 변환 모듈
PyMuPDF와 YOLO의 좌표 규격이 다르므로 좌표 변환 필요

PyMuPDF 좌표계:
- 원점: 좌상단
- 단위: 포인트 (72 points = 1 inch)
- 페이지 크기 기준

YOLO 좌표계:
- 원점: 좌상단  
- 단위: 픽셀
- 이미지 크기 기준

변환 시 고려사항:
- DPI에 따른 스케일 변환
- 페이지 크기와 이미지 크기의 비율
"""

from typing import Tuple, Dict, List


class CoordinateTransformer:
    """좌표계 변환 유틸리티"""
    
    # 기본 상수
    POINTS_PER_INCH = 72
    
    def __init__(self):
        pass
    
    def pymupdf_to_yolo(
        self,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float,
        image_width: float,
        image_height: float
    ) -> Tuple[float, float, float, float]:
        """
        PyMuPDF 좌표를 YOLO (이미지 픽셀) 좌표로 변환
        
        Args:
            bbox: PyMuPDF 바운딩 박스 (x0, y0, x1, y1) - 포인트 단위
            page_width: PDF 페이지 너비 (포인트)
            page_height: PDF 페이지 높이 (포인트)
            image_width: 변환된 이미지 너비 (픽셀)
            image_height: 변환된 이미지 높이 (픽셀)
            
        Returns:
            YOLO 좌표 (x1, y1, x2, y2) - 픽셀 단위
        """
        x0, y0, x1, y1 = bbox
        
        # 스케일 계산
        scale_x = image_width / page_width
        scale_y = image_height / page_height
        
        # 좌표 변환
        new_x0 = x0 * scale_x
        new_y0 = y0 * scale_y
        new_x1 = x1 * scale_x
        new_y1 = y1 * scale_y
        
        return (new_x0, new_y0, new_x1, new_y1)
    
    def yolo_to_pymupdf(
        self,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float,
        image_width: float,
        image_height: float
    ) -> Tuple[float, float, float, float]:
        """
        YOLO (이미지 픽셀) 좌표를 PyMuPDF 좌표로 변환
        
        Args:
            bbox: YOLO 바운딩 박스 (x1, y1, x2, y2) - 픽셀 단위
            page_width: PDF 페이지 너비 (포인트)
            page_height: PDF 페이지 높이 (포인트)
            image_width: 이미지 너비 (픽셀)
            image_height: 이미지 높이 (픽셀)
            
        Returns:
            PyMuPDF 좌표 (x0, y0, x1, y1) - 포인트 단위
        """
        x1, y1, x2, y2 = bbox
        
        # 스케일 계산
        scale_x = page_width / image_width
        scale_y = page_height / image_height
        
        # 좌표 변환
        new_x0 = x1 * scale_x
        new_y0 = y1 * scale_y
        new_x1 = x2 * scale_x
        new_y1 = y2 * scale_y
        
        return (new_x0, new_y0, new_x1, new_y1)
    
    def normalize_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        width: float,
        height: float
    ) -> Tuple[float, float, float, float]:
        """
        바운딩 박스를 정규화 좌표 (0-1)로 변환
        
        Args:
            bbox: (x1, y1, x2, y2) 좌표
            width: 이미지/페이지 너비
            height: 이미지/페이지 높이
            
        Returns:
            정규화된 좌표 (0-1 범위)
        """
        x1, y1, x2, y2 = bbox
        
        return (
            x1 / width,
            y1 / height,
            x2 / width,
            y2 / height
        )
    
    def denormalize_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        width: float,
        height: float
    ) -> Tuple[float, float, float, float]:
        """
        정규화된 좌표를 실제 좌표로 변환
        
        Args:
            bbox: 정규화된 (x1, y1, x2, y2) 좌표 (0-1)
            width: 이미지/페이지 너비
            height: 이미지/페이지 높이
            
        Returns:
            실제 좌표
        """
        x1, y1, x2, y2 = bbox
        
        return (
            x1 * width,
            y1 * height,
            x2 * width,
            y2 * height
        )
    
    def dpi_to_scale(self, dpi: int) -> float:
        """DPI를 스케일 팩터로 변환"""
        return dpi / self.POINTS_PER_INCH
    
    def points_to_pixels(self, points: float, dpi: int) -> float:
        """포인트를 픽셀로 변환"""
        return points * dpi / self.POINTS_PER_INCH
    
    def pixels_to_points(self, pixels: float, dpi: int) -> float:
        """픽셀을 포인트로 변환"""
        return pixels * self.POINTS_PER_INCH / dpi
    
    def transform_all_blocks(
        self,
        text_blocks: List[Dict],
        page_width: float,
        page_height: float,
        image_width: float,
        image_height: float
    ) -> List[Dict]:
        """
        모든 텍스트 블록의 좌표 변환
        
        Args:
            text_blocks: PyMuPDF에서 추출한 텍스트 블록 리스트
            page_width: PDF 페이지 너비
            page_height: PDF 페이지 높이
            image_width: 이미지 너비
            image_height: 이미지 높이
            
        Returns:
            YOLO 좌표로 변환된 텍스트 블록 리스트
        """
        transformed = []
        
        for block in text_blocks:
            new_block = block.copy()
            
            if 'bbox' in block:
                new_block['bbox_original'] = block['bbox']
                new_block['bbox'] = self.pymupdf_to_yolo(
                    block['bbox'],
                    page_width,
                    page_height,
                    image_width,
                    image_height
                )
            
            transformed.append(new_block)
        
        return transformed
    
    def calculate_iou(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float]
    ) -> float:
        """
        두 바운딩 박스의 IoU (Intersection over Union) 계산
        
        Args:
            box1: (x1, y1, x2, y2)
            box2: (x1, y1, x2, y2)
            
        Returns:
            IoU 값 (0-1)
        """
        # 교집합 영역
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # 각 박스의 면적
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 합집합 면적
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0
        
        return intersection / union
    
    def check_containment(
        self,
        inner_box: Tuple[float, float, float, float],
        outer_box: Tuple[float, float, float, float],
        threshold: float = 0.8
    ) -> bool:
        """
        inner_box가 outer_box 안에 포함되는지 확인
        
        Args:
            inner_box: 내부 박스
            outer_box: 외부 박스
            threshold: 포함 비율 임계값
            
        Returns:
            포함 여부
        """
        # 교집합 영역
        x1 = max(inner_box[0], outer_box[0])
        y1 = max(inner_box[1], outer_box[1])
        x2 = min(inner_box[2], outer_box[2])
        y2 = min(inner_box[3], outer_box[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # inner_box 면적
        inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
        
        if inner_area == 0:
            return False
        
        # 포함 비율
        containment_ratio = intersection / inner_area
        
        return containment_ratio >= threshold
