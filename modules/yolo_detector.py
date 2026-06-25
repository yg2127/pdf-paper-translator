"""
YOLOv11을 사용하여 논문 내 그림, 표, 수식, 캡션을 감지하는 모듈
입력: 고해상도 이미지
출력: 감지된 객체의 위치 좌표 (Bounding Box)
"""

from pathlib import Path
from typing import List, Dict, Union
from PIL import Image
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics가 설치되지 않았습니다. pip install ultralytics")
    YOLO = None


class YOLODetector:
    """YOLO 기반 문서 요소 감지기"""

    def __init__(
        self,
        model_path: str = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45
    ):
        """
        Args:
            model_path: YOLO 모델 경로 (None이면 사전훈련된 모델 사용)
            confidence_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = {}  # 모델에서 동적으로 로드

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    def _load_model(self, model_path: str):
        """YOLO 모델 로드"""
        if YOLO is None:
            raise ImportError("ultralytics 패키지가 필요합니다.")
        self.model = YOLO(model_path)
        # 모델에서 클래스 이름 가져오기
        self.class_names = self.model.names
        print(f"YOLO 모델 클래스: {self.class_names}")
    
    def detect(self, images: List[Image.Image]) -> List[List[Dict]]:
        """
        이미지들에서 문서 요소 감지
        
        Args:
            images: PIL Image 리스트 (페이지별)
            
        Returns:
            페이지별 감지 결과 리스트
            각 감지: {
                'class_id': int,
                'class_name': str,
                'confidence': float,
                'bbox': (x1, y1, x2, y2),  # 픽셀 좌표
                'bbox_normalized': (x1, y1, x2, y2),  # 정규화 좌표 (0-1)
            }
        """
        if self.model is None:
            # 모델이 없으면 빈 결과 반환 (또는 기본 감지)
            print("경고: YOLO 모델이 로드되지 않았습니다. 빈 감지 결과를 반환합니다.")
            return [[] for _ in images]
        
        all_detections = []
        
        for image in images:
            # PIL Image를 numpy 배열로 변환
            img_array = np.array(image)
            img_height, img_width = img_array.shape[:2]
            
            # YOLO 추론
            results = self.model.predict(
                img_array,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            page_detections = []
            
            for result in results:
                boxes = result.boxes
                
                if boxes is None:
                    continue
                
                for i in range(len(boxes)):
                    # 바운딩 박스 (x1, y1, x2, y2)
                    bbox = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().item()
                    class_id = int(boxes.cls[i].cpu().item())
                    
                    # 정규화 좌표
                    bbox_normalized = (
                        bbox[0] / img_width,
                        bbox[1] / img_height,
                        bbox[2] / img_width,
                        bbox[3] / img_height
                    )
                    
                    page_detections.append({
                        'class_id': class_id,
                        'class_name': self.class_names.get(class_id, f'class_{class_id}'),
                        'confidence': confidence,
                        'bbox': tuple(bbox),
                        'bbox_normalized': bbox_normalized,
                        'image_size': (img_width, img_height)
                    })
            
            all_detections.append(page_detections)
        
        return all_detections
    


class MockYOLODetector(YOLODetector):
    """
    테스트용 Mock 감지기
    실제 YOLO 모델 없이 더미 결과 반환
    """
    
    def __init__(self, **kwargs):
        # 부모 클래스 초기화 건너뜀
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.model = "mock"
    
    def detect(self, images: List[Image.Image]) -> List[List[Dict]]:
        """더미 감지 결과 반환"""
        all_detections = []
        
        for image in images:
            width, height = image.size
            
            # 테스트용 더미 감지 결과
            page_detections = [
                {
                    'class_id': 0,
                    'class_name': 'figure',
                    'confidence': 0.95,
                    'bbox': (width * 0.1, height * 0.3, width * 0.9, height * 0.6),
                    'bbox_normalized': (0.1, 0.3, 0.9, 0.6),
                    'image_size': (width, height)
                }
            ]
            
            all_detections.append(page_detections)
        
        return all_detections
