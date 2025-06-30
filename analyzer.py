import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import torch
from ultralytics import YOLO
from typing import Dict, List, Optional
import logging
import datetime
from databases import Database
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Detection:
    bbox: List[int]  # [x, y, width, height]
    vehicle_type: str
    color: tuple
    direction: str
    confidence: float
    weight: int
    track_id: str = ""

class VideoAnalyzer:
    def __init__(self, database: Database):
        self.database = database
        
        # Инициализация модели YOLO
        try:
            self.model = self._load_model()
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
        
        # Карта классов транспортных средств
        self.class_map = {
            0: ('Coupe', (0, 0, 255)),
            1: ('Crossover', (0, 255, 0)),
            2: ('Hatchback', (255, 0, 0)),
            3: ('Sedan', (0, 255, 255)),
            4: ('Station wagon', (255, 0, 255)),
            5: ('Truck', (0, 140, 255))
        }
        
        # Кэш для данных камер и полигонов
        self._cameras_cache = {}
        self._polygons_cache = {}
        self._last_cache_update = datetime.datetime.min
        
        logger.info("VideoAnalyzer initialized successfully")

    async def initialize(self):
        """Асинхронная инициализация (загрузка данных из БД)"""
        try:
            await self._load_cameras()
            await self._load_polygons()
            logger.info(f"Loaded data for {len(self._cameras_cache)} cameras")
        except Exception as e:
            logger.error(f"Error initializing analyzer: {str(e)}")
            raise

    def _load_model(self):
        """Загрузка обученной модели YOLO"""
        try:
            model_path = "models/best.pt"
            model = YOLO(model_path)
            
            # Проверка доступности GPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)
            
            logger.info(f"Model loaded on device: {device}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    async def _load_cameras(self):
        """Загрузка данных о камерах из БД"""
        try:
            query = "SELECT * FROM cameras"
            cameras = await self.database.fetch_all(query)
            self._cameras_cache = {cam['id']: dict(cam) for cam in cameras}
            logger.info(f"Loaded {len(self._cameras_cache)} cameras from DB")
        except Exception as e:
            logger.error(f"Error loading cameras: {str(e)}")
            self._cameras_cache = {}

    async def _load_polygons(self):
        """Загрузка полигонов из БД"""
        try:
            query = """
                SELECT p.*, c.stream_url 
                FROM polygons p
                LEFT JOIN cameras c ON p.camera_id = c.id
            """
            polygons = await self.database.fetch_all(query)
            
            self._polygons_cache = {}
            for poly in polygons:
                cam_id = poly['camera_id']
                if cam_id not in self._polygons_cache:
                    self._polygons_cache[cam_id] = []
                
                try:
                    # Предполагаем, что coordinates хранится как JSON в БД
                    coords = poly['coordinates']['coordinates'][0]  # GeoJSON формат
                    self._polygons_cache[cam_id].append({
                        'polygon': Polygon(coords),
                        'direction': poly.get('direction', 'unknown'),
                        'name': poly.get('name', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing polygon: {str(e)}")
            
            logger.info(f"Loaded polygons for {len(self._polygons_cache)} cameras")
        except Exception as e:
            logger.error(f"Error loading polygons: {str(e)}")
            self._polygons_cache = {}

    async def _refresh_cache_if_needed(self):
        """Обновляет кэш данных, если прошло больше 5 минут с последнего обновления"""
        now = datetime.datetime.now()
        if (now - self._last_cache_update).total_seconds() > 300:  # 5 минут
            await self._load_cameras()
            await self._load_polygons()
            self._last_cache_update = now
            logger.info("Refreshed camera and polygon cache")

    async def analyze_frame(self, frame: np.ndarray, camera_id: int) -> Dict:
        """Анализ одного кадра с обученной моделью"""
        try:
            # Обновляем кэш при необходимости
            await self._refresh_cache_if_needed()
            
            # Получаем результаты от YOLO
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
                direction = await self._get_direction(center, camera_id)
                
                # Создаем объект обнаружения
                detection = Detection(
                    bbox=[x1, y1, x2 - x1, y2 - y1],
                    vehicle_type=vehicle_name,
                    color=vehicle_color,
                    direction=direction,
                    confidence=confidence,
                    weight=3 if vehicle_name == 'Truck' else 1
                )
                
                detections.append(detection)
                
                # Сохраняем обнаружение в БД
                await self._save_detection(camera_id, detection)
            
            return {
                'detections': [self._detection_to_dict(d) for d in detections],
                'frame': self._draw_results(frame.copy(), detections)
            }
            
        except Exception as e:
            logger.error(f"Frame analysis error: {str(e)}")
            return {'detections': [], 'frame': frame}

    async def _get_direction(self, point: Point, camera_id: int) -> str:
        """Определение направления движения"""
        # Проверяем полигоны из кэша
        for zone in self._polygons_cache.get(camera_id, []):
            if zone['polygon'].contains(point):
                return zone['direction']
        
        # Проверяем зоны камеры (если есть в кэше)
        if camera_id in self._cameras_cache:
            camera_data = self._cameras_cache[camera_id]
            if 'zones' in camera_data and camera_data['zones']:
                for zone in camera_data['zones']:
                    try:
                        poly = Polygon(self._normalize_points(zone['points'], (720, 1280)))
                        if poly.contains(point):
                            return zone['name'].replace('_zone', '')
                    except Exception as e:
                        logger.warning(f"Error processing camera zone: {str(e)}")
        
        return 'unknown'

    def _normalize_points(self, points: List[List[float]], shape) -> List[List[int]]:
        """Нормализация координат"""
        h, w = shape[:2]
        return [[int(x * w), int(y * h)] for x, y in points]

    def _draw_results(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Отрисовка результатов с цветами по классам"""
        for det in detections:
            x, y, w, h = det.bbox
            color = det.color
            
            # Рисуем bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Рисуем текст с фоном
            label = f"{det.vehicle_type} {det.confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Фон для текста
            cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
            cv2.putText(
                frame, label,
                (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )
        return frame

    async def _save_detection(self, camera_id: int, detection: Detection):
        """Сохранение обнаружения в базу данных"""
        try:
            query = """
                INSERT INTO analysis_results 
                (camera_id, track_id, vehicle_type, direction, confidence, weight, timestamp)
                VALUES (:camera_id, :track_id, :vehicle_type, :direction, :confidence, :weight, :timestamp)
            """
            
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            track_id = f"trk_{timestamp.strftime('%Y%m%d')}_{camera_id}_{hash(detection)}"[:64]
            
            values = {
                "camera_id": camera_id,
                "track_id": track_id,
                "vehicle_type": detection.vehicle_type,
                "direction": detection.direction,
                "confidence": round(float(detection.confidence), 4),
                "weight": detection.weight,
                "timestamp": timestamp
            }
            
            await self.database.execute(query, values)
            logger.debug(f"Saved detection to DB: {track_id}")
            
        except Exception as e:
            logger.error(f"Error saving detection to DB: {str(e)}")

    def _detection_to_dict(self, detection: Detection) -> dict:
        """Конвертирует объект Detection в словарь"""
        return {
            'bbox': detection.bbox,
            'type': detection.vehicle_type,
            'color': detection.color,
            'direction': detection.direction,
            'confidence': detection.confidence,
            'weight': detection.weight
        }