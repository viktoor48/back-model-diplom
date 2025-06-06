import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import torch
from ultralytics import YOLO
from typing import Dict, List, Optional
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        self.model = self._load_model()
        self.cameras = self._load_cameras()
        self.polygons = self._load_polygons()
        self.class_map = {
            0: ('Coupe', (0, 0, 255)),
            1: ('Crossover', (0, 255, 0)),
            2: ('Hatchback', (255, 0, 0)),
            3: ('Sedan', (0, 255, 255)),
            4: ('Station wagon', (255, 0, 255)),
            5: ('Truck', (0, 140, 255))
        }

    def _load_model(self):
        """Загрузка обученной модели YOLO"""
        from ultralytics import YOLO
        try:
            # Укажите правильный путь к вашей модели
            model_path = "models/best.pt"
            model = YOLO(model_path)
            
            # Проверка доступности GPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            
            logger.info(f"Модель YOLO успешно загружена с устройства {device}")
            return model
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise

    def _load_cameras(self) -> Dict:
        """Загрузка данных о камерах"""
        try:
            with open('data/cameras.json') as f:
                cameras = json.load(f)
                return {cam['id']: cam for cam in cameras}
        except Exception as e:
            logger.error(f"Ошибка загрузки cameras.json: {str(e)}")
            return {}

    def _load_polygons(self) -> Dict:
        """Загрузка полигонов"""
        try:
            with open('data/polygons.geojson') as f:
                polygons = json.load(f)
                
            result = {}
            for poly in polygons:
                cam_id = poly['camera_id']
                if cam_id not in result:
                    result[cam_id] = []
                
                try:
                    result[cam_id].append({
                        'polygon': Polygon(poly['geometry']['coordinates'][0]),
                        'direction': poly['direction']
                    })
                except Exception as e:
                    logger.warning(f"Ошибка обработки полигона: {str(e)}")
            
            return result
        except Exception as e:
            logger.error(f"Ошибка загрузки polygons.geojson: {str(e)}")
            return {}
        
    def analyze_frame(self, frame: np.ndarray, camera_id: int) -> Dict:
        """Анализ одного кадра с обученной моделью"""
        try:
            results = self.model(
                frame,
                imgsz=1280,
                conf=0.5,
                iou=0.45,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )[0]
            
            detections = []
            
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                if class_id not in self.class_map:
                    continue
                    
                # Извлекаем название и цвет класса
                vehicle_name, vehicle_color = self.class_map[class_id]
                center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                direction = self._get_direction(center, camera_id)
                
                detections.append({
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'type': vehicle_name,  # Сохраняем только название
                    'color': vehicle_color,  # Добавляем цвет
                    'direction': direction,
                    'confidence': confidence,
                    'weight': 3 if vehicle_name == 'Truck' else 1
                })
            
            return {
                'detections': detections,
                'frame': self._draw_results(frame.copy(), detections)
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа кадра: {str(e)}")
            return {'detections': [], 'frame': frame}
    

    def _get_direction(self, point: Point, camera_id: int) -> str:
        """Определение направления движения"""
        # Сначала проверяем полигоны
        for zone in self.polygons.get(camera_id, []):
            if zone['polygon'].contains(point):
                return zone['direction']
        
        # Затем зоны камеры
        if camera_id in self.cameras:
            for zone in self.cameras[camera_id].get('zones', []):
                poly = Polygon(self._normalize_points(zone['points'], (720, 1280)))
                if poly.contains(point):
                    return zone['name'].replace('_zone', '')
        
        return 'unknown'

    def _normalize_points(self, points: List[List[float]], shape) -> List[List[int]]:
        """Нормализация координат"""
        h, w = shape[:2]
        return [[int(x * w), int(y * h)] for x, y in points]

    def _draw_results(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Отрисовка результатов с цветами по классам"""
        for det in detections:
            x, y, w, h = det['bbox']
            color = det['color']  # Используем цвет из detection
            
            # Рисуем bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Рисуем текст с фоном
            label = f"{det['type']} {det['confidence']:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Фон для текста
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            cv2.putText(
                frame, label,
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1  # Чёрный текст
            )
        return frame