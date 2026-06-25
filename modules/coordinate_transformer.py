"""
좌표계 변환 모듈
PyMuPDF(포인트, 페이지 기준)와 YOLO(픽셀, 이미지 기준)의 좌표 규격이 달라 변환이 필요하다.
"""

from typing import Tuple


class CoordinateTransformer:
    """좌표계 변환 유틸리티"""

    def yolo_to_pymupdf(
        self,
        bbox: Tuple[float, float, float, float],
        page_width: float,
        page_height: float,
        image_width: float,
        image_height: float,
    ) -> Tuple[float, float, float, float]:
        """YOLO(이미지 픽셀) 좌표를 PyMuPDF(포인트) 좌표로 변환"""
        x1, y1, x2, y2 = bbox
        scale_x = page_width / image_width
        scale_y = page_height / image_height
        return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

    def intersects(
        self,
        box1: Tuple[float, float, float, float],
        box2: Tuple[float, float, float, float],
    ) -> bool:
        """두 박스(같은 좌표계)가 실제로 겹치는지 검사"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        return x2 > x1 and y2 > y1
