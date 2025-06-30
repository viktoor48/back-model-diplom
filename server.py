import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import tempfile
import os
import time
import threading
import logging
from analyzer import VideoAnalyzer
import numpy as np 
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
import pandas as pd
from io import BytesIO
from pathlib import Path
from dateutil.parser import parse
import pytz
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
import asyncio
from typing import List, Optional
from databases import Database
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Boolean, JSON, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import sqlalchemy as sa

# Настройки базы данных
DATABASE_URL = "postgresql://postgres:your_strong_password@db:5432/video_analysis"
database = Database(DATABASE_URL)

# Настройка логгера
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

Base = declarative_base()

# Модели SQLAlchemy
class Camera(Base):
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    location = Column(String(255))
    stream_url = Column(String(255))
    is_active = Column(Boolean, default=True, server_default='true')
    created_at = Column(DateTime, server_default=sa.func.now())

class Polygon(Base):
    __tablename__ = 'polygons'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=False)
    name = Column(String(255))
    coordinates = Column(JSON)
    created_at = Column(DateTime, server_default=sa.func.now())

class AnalysisResult(Base):
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=False)
    track_id = Column(String(255))
    vehicle_type = Column(String(50))
    direction = Column(String(50))
    confidence = Column(Float)
    weight = Column(Float)
    timestamp = Column(DateTime)
    created_at = Column(DateTime, server_default=sa.func.now())

app = FastAPI()
analyzer = VideoAnalyzer(database) 

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели Pydantic
class CameraModel(BaseModel):
    id: int
    name: str
    location: str
    stream_url: str
    is_active: bool
    created_at: datetime

    class Config:
        orm_mode = True

class PolygonModel(BaseModel):
    id: int
    camera_id: int
    name: str
    coordinates: dict
    created_at: datetime

    class Config:
        orm_mode = True

class AnalysisResultModel(BaseModel):
    id: int
    camera_id: int
    track_id: str
    vehicle_type: str
    direction: str
    confidence: float
    weight: float
    timestamp: datetime
    created_at: datetime

    class Config:
        orm_mode = True

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_frame(self, frame: np.ndarray, detections: list):
        if not self.active_connections:
            return
            
        _, jpeg = cv2.imencode('.jpg', frame)
        data = {
            "frame": jpeg.tobytes().hex(),
            "detections": detections,
            "timestamp": datetime.now().isoformat()
        }
        
        for connection in self.active_connections:
            try:
                await connection.send_json(data)
            except Exception as e:
                logger.error(f"Error sending frame: {e}")
                self.disconnect(connection)

manager = ConnectionManager()

class VideoProcessor:
    def __init__(self):
        self.current_frame = None
        self.detections = []
        self.processing = False
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_skip = 2
        self.target_size = (1280, 720)
        self.cap = None
        self.manager = manager
        self.processing_task = None

    async def process_video(self, file_path: str, camera_id: int):
        self.processing = True
        self.cap = cv2.VideoCapture(file_path)
        
        try:
            frame_count = 0
            while self.processing and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or not self.processing:
                    break
                
                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                try:
                    frame = cv2.resize(frame, self.target_size)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    result = await run_in_threadpool(analyzer.analyze_frame, frame, camera_id)
                    
                    with self.lock:
                        self.current_frame = result['frame']
                        self.detections = result['detections']
                    
                    await self.manager.broadcast_frame(result['frame'], result['detections'])
                    
                    elapsed = time.time() - self.last_frame_time
                    sleep_time = max(0, 0.033 - elapsed)
                    await asyncio.sleep(sleep_time)
                    self.last_frame_time = time.time()
                    
                except Exception as e:
                    logger.error(f"Frame processing error: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            if os.path.exists(file_path):
                os.remove(file_path)
            self.processing = False
            logger.info("Video processing completed")

processor = VideoProcessor()

# Инициализация асинхронного движка SQLAlchemy
engine = create_async_engine(DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))
async_session = sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

@app.on_event("startup")
async def startup():
    await database.connect()
    # Создаем таблицы через SQLAlchemy
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Инициализируем анализатор
    await analyzer.initialize()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
    await engine.dispose()

# Вебсокет endpoint
@app.websocket("/ws/video_feed")
async def websocket_video_feed(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Остальные endpoint'ы (анализ видео, управление камерами и т.д.)
@app.post("/start_analysis/{camera_id}")
async def start_analysis(camera_id: int, file: UploadFile = File(...)):
    try:
        if processor.processing:
            return {"status": "already_processing"}

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            file_path = tmp.name

        processor.processing_task = asyncio.create_task(
            processor.process_video(file_path, camera_id)
        )

        return {"status": "processing_started"}
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_analysis")
async def stop_analysis():
    if processor.processing:
        processor.processing = False
        if processor.processing_task:
            processor.processing_task.cancel()
            try:
                await processor.processing_task
            except asyncio.CancelledError:
                pass
        if processor.cap:
            processor.cap.release()
        logger.info("Video processing stopped")
    return {"status": "processing_stopped"}

@app.get("/current_frame")
async def get_current_frame():
    if processor.current_frame is None:
        raise HTTPException(status_code=404, detail="No frame data available")
    
    with processor.lock:
        _, jpeg = cv2.imencode('.jpg', processor.current_frame)
    
    return StreamingResponse(
        iter([jpeg.tobytes()]),
        media_type="image/jpeg"
    )

@app.get("/current_detections")
async def get_current_detections():
    with processor.lock:
        return processor.detections

# CRUD для камер
@app.get("/cameras", response_model=List[CameraModel])
async def get_cameras():
    async with async_session() as session:
        result = await session.execute(sa.select(Camera))
        cameras = result.scalars().all()
        return cameras

@app.post("/cameras", response_model=CameraModel)
async def create_camera(camera: CameraModel):
    async with async_session() as session:
        db_camera = Camera(**camera.dict())
        session.add(db_camera)
        await session.commit()
        await session.refresh(db_camera)
        return db_camera
    
@app.get("/polygons")
async def get_polygons(camera_id: Optional[int] = None):
    try:
        if camera_id:
            query = "SELECT * FROM polygons WHERE camera_id = :camera_id"
            polygons = await database.fetch_all(query, {"camera_id": camera_id})
        else:
            query = "SELECT * FROM polygons"
            polygons = await database.fetch_all(query)
        
        return [dict(polygon) for polygon in polygons]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_polygon")
async def update_polygon(data: dict):
    try:
        camera_id = data.get('camera_id')
        polygon_data = data.get('data', {})
        
        # Проверяем существование полигона
        existing = await database.fetch_one(
            "SELECT id FROM polygons WHERE id = :id",
            {"id": polygon_data.get('id')}
        )
        
        if existing:
            # Обновляем существующий полигон
            query = """
                UPDATE polygons 
                SET name = :name, coordinates = :coordinates
                WHERE id = :id
            """
            values = {
                "id": polygon_data.get('id'),
                "name": polygon_data.get('name'),
                "coordinates": polygon_data.get('coordinates')
            }
        else:
            # Создаем новый полигон
            query = """
                INSERT INTO polygons 
                (camera_id, name, coordinates)
                VALUES (:camera_id, :name, :coordinates)
                RETURNING id
            """
            values = {
                "camera_id": camera_id,
                "name": polygon_data.get('name'),
                "coordinates": polygon_data.get('coordinates')
            }
        
        await database.execute(query, values)
        return {"status": "success", "camera_id": camera_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis_data")
async def get_analysis_data(
    camera_id: Optional[str] = None,
    search: Optional[str] = None,
    page: int = 1,
    limit: int = 20
):
    try:
        base_query = "SELECT * FROM analysis_results"
        count_query = "SELECT COUNT(*) FROM analysis_results"
        conditions = []
        values = {}
        
        if camera_id:
            camera_ids = [int(id) for id in camera_id.split(',')]
            conditions.append("camera_id = ANY(:camera_ids)")
            values["camera_ids"] = camera_ids
        
        if search:
            search = f"%{search.lower()}%"
            conditions.append(
                "(LOWER(track_id) LIKE :search OR LOWER(vehicle_type) LIKE :search)"
            )
            values["search"] = search
        
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)
            base_query += where_clause
            count_query += where_clause
        
        # Получаем общее количество
        total = await database.fetch_val(count_query, values)
        
        # Добавляем сортировку и пагинацию
        base_query += " ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"
        values["limit"] = limit
        values["offset"] = (page - 1) * limit
        
        data = await database.fetch_all(base_query, values)
        
        return {
            "data": [dict(item) for item in data],
            "total": total,
            "page": page,
            "limit": limit,
            "has_more": (page * limit) < total
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export_report")
async def export_report(
    period: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    fields: Optional[str] = None
):
    try:
        # Определяем временной диапазон
        now = datetime.now(pytz.UTC) if hasattr(pytz, 'UTC') else datetime.utcnow()
        time_conditions = {
            "5min": now - timedelta(minutes=5),
            "10min": now - timedelta(minutes=10),
            "1h": now - timedelta(hours=1),
            "week": now - timedelta(weeks=1),
            "month": now - timedelta(days=30)
        }
        
        base_query = "SELECT * FROM analysis_results"
        conditions = []
        values = {}
        
        if period and period in time_conditions:
            conditions.append("timestamp >= :cutoff")
            values["cutoff"] = time_conditions[period]
        elif period == "custom" and start_date and end_date:
            try:
                start = parse_datetime(start_date)
                end = parse_datetime(end_date)
                conditions.append("timestamp BETWEEN :start AND :end")
                values["start"] = start
                values["end"] = end
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
        
        if conditions:
            base_query += " WHERE " + " AND ".join(conditions)
        
        # Получаем данные
        data = await database.fetch_all(base_query, values)
        if not data:
            raise HTTPException(status_code=404, detail="No data for selected period")
        
        # Определяем поля для экспорта
        all_fields = {
            'timestamp': 'Timestamp',
            'camera_id': 'Camera ID',
            'track_id': 'Track ID',
            'vehicle_type': 'Vehicle Type',
            'direction': 'Direction',
            'confidence': 'Confidence',
            'weight': 'Weight'
        }
        
        if fields:
            requested_fields = [f.strip() for f in fields.split(',') if f.strip()]
            valid_fields = [(f, all_fields[f]) for f in requested_fields if f in all_fields]
            if not valid_fields:
                raise HTTPException(status_code=400, detail="No valid fields selected")
        else:
            valid_fields = [(f, n) for f, n in all_fields.items()]
        
        # Формируем данные для Excel
        report_data = []
        for item in data:
            row = {display_name: item.get(field, 'N/A') 
                  for field, display_name in valid_fields}
            report_data.append(row)
        
        # Создаем Excel файл
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            report_df = pd.DataFrame(report_data)
            
            if 'Timestamp' in report_df.columns:
                try:
                    report_df['Timestamp'] = pd.to_datetime(report_df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    report_df['Timestamp'] = report_df['Timestamp'].astype(str)
            
            report_df.to_excel(writer, index=False, sheet_name='Report')
            worksheet = writer.sheets['Report']
            
            for i, (_, display_name) in enumerate(valid_fields):
                max_len = max(
                    report_df[display_name].astype(str).map(len).max(),
                    len(display_name)
                ) + 2
                worksheet.set_column(i, i, max_len)
        
        output.seek(0)
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_datetime(dt_str):
    """Парсит строку даты в datetime"""
    try:
        try:
            return datetime.fromisoformat(dt_str)
        except ValueError:
            try:
                return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S%z")
            except ValueError:
                return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f%z")
    except Exception as e:
        logger.error(f"Error parsing datetime {dt_str}: {str(e)}")
        return datetime.min


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)