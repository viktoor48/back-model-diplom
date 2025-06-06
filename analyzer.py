import shutil
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import torch
from ultralytics import YOLO
from typing import Dict, List, Optional
import json
import os
import logging
import datetime
from filelock import FileLock

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self):
        # Сначала инициализируем критически важные пути
        self.analysis_file = 'data/analysis.json'
        self.lock_file = 'data/analysis.lock'
        
        # Создаем директорию data если ее нет
        try:
            os.makedirs('data', exist_ok=True)
            logger.info(f"Директория 'data' создана или уже существует")
        except Exception as e:
            logger.error(f"Ошибка создания директории 'data': {str(e)}")
            raise
        
        # Инициализируем файл анализа (должно быть перед всеми операциями с файлами)
        try:
            self._init_analysis_file()
        except Exception as e:
            logger.error(f"Ошибка инициализации файла анализа: {str(e)}")
            raise
        
        # Затем загружаем модель и данные
        try:
            self.model = self._load_model()
            logger.info("Модель YOLO успешно загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {str(e)}")
            raise
        
        try:
            self.cameras = self._load_cameras()
            logger.info(f"Загружены данные по {len(self.cameras)} камерам")
        except Exception as e:
            logger.error(f"Ошибка загрузки данных камер: {str(e)}")
            self.cameras = {}
        
        try:
            self.polygons = self._load_polygons()
            logger.info(f"Загружены полигоны для {len(self.polygons)} камер")
        except Exception as e:
            logger.error(f"Ошибка загрузки полигонов: {str(e)}")
            self.polygons = {}
        
        # Карта классов транспортных средств
        self.class_map = {
            0: ('Coupe', (0, 0, 255)),
            1: ('Crossover', (0, 255, 0)),
            2: ('Hatchback', (255, 0, 0)),
            3: ('Sedan', (0, 255, 255)),
            4: ('Station wagon', (255, 0, 255)),
            5: ('Truck', (0, 140, 255))
        }
        
        logger.info("Инициализация VideoAnalyzer завершена успешно")

    def _init_analysis_file(self):
        """Инициализация файла анализа с блокировкой"""
        try:
            with FileLock(self.lock_file):
                if not os.path.exists(self.analysis_file):
                    with open(self.analysis_file, 'w') as f:
                        json.dump([], f)
        except Exception as e:
            logger.error(f"Error initializing analysis file: {str(e)}")
            raise

    def _save_detection(self, camera_id: int, detection: dict):
        """Безопасное сохранение данных с блокировкой и резервированием"""
        backup_path = f"{self.analysis_file}.bak"
        temp_path = f"{self.analysis_file}.tmp"
        
        with FileLock(self.lock_file):  # Блокировка для потокобезопасности
            try:
                # 1. Создаем резервную копию
                if os.path.exists(self.analysis_file):
                    shutil.copyfile(self.analysis_file, backup_path)
                
                # 2. Загружаем существующие данные
                data = []
                if os.path.exists(self.analysis_file):
                    with open(self.analysis_file, 'r') as f:
                        try:
                            data = json.load(f)
                            if not isinstance(data, list):
                                raise ValueError("Invalid data format: not a list")
                        except json.JSONDecodeError:
                            logger.error("Ошибка чтения JSON, восстанавливаем из резервной копии")
                            if os.path.exists(backup_path):
                                shutil.copyfile(backup_path, self.analysis_file)
                                with open(self.analysis_file, 'r') as f:
                                    data = json.load(f)
                
                # 3. Добавляем новую запись
                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                record = {
                    "camera_id": camera_id,
                    "track_id": f"trk_{timestamp[:10]}_{camera_id}_{len(data)}",
                    "vehicle_type": detection['type'],
                    "weight": detection['weight'],
                    "timestamp": timestamp,
                    "direction": detection['direction'],
                    "bbox": detection['bbox'],
                    "confidence": round(float(detection['confidence']), 4),  # Округление
                    "frame_size": [1280, 720],
                    "color": detection['color']  # Сохраняем цвет для визуализации
                }
                data.append(record)
                
                # 4. Атомарная запись через временный файл
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # 5. Заменяем оригинальный файл
                os.replace(temp_path, self.analysis_file)
                
                logger.debug(f"Сохранено обнаружение: {record['track_id']}")
                
            except Exception as e:
                logger.error(f"Ошибка сохранения: {str(e)}")
                # Восстановление из резервной копии при ошибке
                if os.path.exists(backup_path) and not os.path.exists(self.analysis_file):
                    shutil.copyfile(backup_path, self.analysis_file)
                raise
            finally:
                # Удаляем временные файлы
                for path in [backup_path, temp_path]:
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except Exception as e:
                            logger.warning(f"Не удалось удалить временный файл {path}: {str(e)}")

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
                    
                vehicle_name, vehicle_color = self.class_map[class_id]
                center = Point((x1 + x2) / 2, (y1 + y2) / 2)
                direction = self._get_direction(center, camera_id)
                
                detection_data = {
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'type': vehicle_name,
                    'color': vehicle_color,
                    'direction': direction,
                    'confidence': confidence,
                    'weight': 3 if vehicle_name == 'Truck' else 1
                }
                
                detections.append(detection_data)
                self._save_detection(camera_id, detection_data)  # Сохраняем каждое обнаружение
            
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