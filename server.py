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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
analyzer = VideoAnalyzer()

# Fixed the middleware name (was CORS_middleware)
app.add_middleware(
    CORSMiddleware,  # Corrected the class name
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class VideoProcessor:
    def __init__(self):
        self.current_frame = None
        self.detections = []
        self.processing = False
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.frame_skip = 2
        self.target_size = (1280, 720)
        self.cap = None  # Добавляем ссылку на VideoCapture

    def process_video(self, file_path: str, camera_id: int):
        self.processing = True
        self.cap = cv2.VideoCapture(file_path)  # Сохраняем в поле класса
        
        try:
            frame_count = 0
            while self.processing and self.cap.isOpened():
                # Добавляем таймаут для чтения кадра
                ret, frame = self.cap.read()
                if not ret or not self.processing:  # Явная проверка флага
                    break
                
                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue

                # Проверка и логирование входного кадра
                logger.debug(f"Input frame shape: {frame.shape}")
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    logger.error(f"Некорректный формат кадра: {frame.shape}")
                    continue
                
                try:
                    # Ресайз и конвертация цвета
                    frame = cv2.resize(frame, self.target_size)
                    if frame.dtype != np.uint8:
                        frame = frame.astype(np.uint8)
                    
                    # Анализ кадра
                    result = analyzer.analyze_frame(frame, camera_id)
                    
                    # Обновление результатов
                    with self.lock:
                        self.current_frame = result['frame']
                        self.detections = result['detections']
                    
                    # Контроль FPS (максимум 30 кадров/сек)
                    elapsed = time.time() - self.last_frame_time
                    sleep_time = max(0, 0.033 - elapsed)
                    time.sleep(sleep_time)
                    self.last_frame_time = time.time()
                    
                except Exception as e:
                    logger.error(f"Ошибка обработки кадра: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Ошибка в процессе видео: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            if os.path.exists(file_path):
                os.remove(file_path)
            self.processing = False
            logger.info("Обработка видео завершена")

processor = VideoProcessor()

@app.post("/start_analysis/{camera_id}")
async def start_analysis(camera_id: int, file: UploadFile = File(...)):
    try:
        if processor.processing:
            return {"status": "already_processing"}

        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            file_path = tmp.name

        # Start processing in a separate thread
        thread = threading.Thread(
            target=processor.process_video,
            args=(file_path, camera_id),
            daemon=True  # Added daemon=True for proper thread cleanup
        )
        thread.start()

        return {"status": "processing_started"}
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/current_frame")
async def get_current_frame():
    if processor.current_frame is None:
        raise HTTPException(status_code=404, detail="No frame data available")
    
    # Use lock when accessing shared variables
    with processor.lock:
        _, jpeg = cv2.imencode('.jpg', processor.current_frame)
    
    return StreamingResponse(
        iter([jpeg.tobytes()]),
        media_type="image/jpeg"
    )

@app.get("/current_detections")
async def get_current_detections():
    # Use lock when accessing shared variables
    with processor.lock:
        return processor.detections

@app.get("/cameras")
async def get_cameras():
    try:
        cameras = list(analyzer.cameras.values())
        return JSONResponse(content=cameras)  # Явно указываем JSON-ответ
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_analysis")
async def stop_analysis():
    if processor.processing:
        processor.processing = False
        if processor.cap:  # Принудительно останавливаем VideoCapture
            processor.cap.release()
        logger.info("Обработка видео принудительно остановлена")
    return {"status": "processing_stopped"}

@app.get("/test_data")
async def test_data():
    return JSONResponse(
        content={"test": "success", "message": "API is working"},
        media_type="application/json"
    )

@app.get("/export_report")
async def export_report(
    period: str = None,
    start_date: str = None,
    end_date: str = None
):
    try:
        # Загружаем данные
        if not os.path.exists('data/analysis.json'):
            raise HTTPException(status_code=404, detail="No data available")
        
        with open('data/analysis.json', 'r') as f:
            data = json.load(f)
        
        if not data:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Фильтрация по дате (остается без изменений)
        filtered_data = []
        now = datetime.utcnow()
        
        if period == "5min":
            cutoff = now - timedelta(minutes=5)
            filtered_data = [item for item in data if datetime.fromisoformat(item['timestamp']) > cutoff]
        elif period == "10min":
            cutoff = now - timedelta(minutes=10)
            filtered_data = [item for item in data if datetime.fromisoformat(item['timestamp']) > cutoff]
        elif period == "1h":
            cutoff = now - timedelta(hours=1)
            filtered_data = [item for item in data if datetime.fromisoformat(item['timestamp']) > cutoff]
        elif period == "week":
            cutoff = now - timedelta(weeks=1)
            filtered_data = [item for item in data if datetime.fromisoformat(item['timestamp']) > cutoff]
        elif period == "month":
            cutoff = now - timedelta(days=30)
            filtered_data = [item for item in data if datetime.fromisoformat(item['timestamp']) > cutoff]
        elif period == "custom" and start_date and end_date:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            filtered_data = [
                item for item in data 
                if start <= datetime.fromisoformat(item['timestamp']) <= end
            ]
        else:
            filtered_data = data
        
        if not filtered_data:
            raise HTTPException(status_code=404, detail="No data for selected period")
        
        # Создаем DataFrame
        df = pd.DataFrame(filtered_data)
        
        # Функция проверки наличия поля
        def has_valid_field(field, item):
            return field in item and item[field] is not None and item[field] != ''

        # Собираем только существующие и валидные поля
        columns_to_export = []
        field_mapping = [
            ('timestamp', 'Timestamp'),
            ('camera_id', 'Camera ID'),
            ('track_id', 'Track ID'),
            ('vehicle_type', 'Vehicle Type'),
            ('direction', 'Direction'),
            ('confidence', 'Confidence'),
            ('weight', 'Weight')
        ]
        
        for field, display_name in field_mapping:
            if any(has_valid_field(field, item) for item in filtered_data):
                columns_to_export.append((field, display_name))
        
        if not columns_to_export:
            raise HTTPException(status_code=400, detail="No valid data fields found")
        
        # Создаем Excel файл
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Создаем DataFrame только с нужными колонками
            report_data = []
            for item in filtered_data:
                row = {}
                for field, display_name in columns_to_export:
                    row[display_name] = item.get(field, 'N/A')
                report_data.append(row)
            
            report_df = pd.DataFrame(report_data)
            
            # Форматируем дату, если есть
            if 'Timestamp' in report_df.columns:
                report_df['Timestamp'] = pd.to_datetime(report_df['Timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            report_df.to_excel(writer, index=False, sheet_name='Report')
            worksheet = writer.sheets['Report']
            
            # Настраиваем ширину колонок
            for i, (field, display_name) in enumerate(columns_to_export):
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


@app.get("/analysis_data", response_model=dict)
async def get_analysis_data(
    camera_id: str = None,
    search: str = None,
    page: int = 1,
    limit: int = 20
):
    try:
        logger.info(f"Request to /analysis_data with params: camera_id={camera_id}, search={search}, page={page}, limit={limit}")
        
        if not os.path.exists('data/analysis.json'):
            return JSONResponse(
                content={
                    "data": [],
                    "total": 0,
                    "page": page,
                    "limit": limit,
                    "has_more": False
                },
                media_type="application/json"
            )
        
        with open('data/analysis.json', 'r') as f:
            data = json.load(f)
        
        # Фильтрация
        if camera_id:
            camera_ids = [int(id) for id in camera_id.split(',')]
            data = [item for item in data if item['camera_id'] in camera_ids]
        
        if search:
            search = search.lower()
            data = [
                item for item in data
                if (search in item['track_id'].lower() or 
                    search in item['vehicle_type'].lower())
            ]
        
        # Сортировка и пагинация
        data.sort(key=lambda x: x['timestamp'], reverse=True)
        total = len(data)
        start = (page - 1) * limit
        end = start + limit
        paginated_data = data[start:end]
        
        return JSONResponse(
            content={
                "data": paginated_data,
                "total": total,
                "page": page,
                "limit": limit,
                "has_more": end < total
            },
            media_type="application/json"
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in analysis file: {e}")
        raise HTTPException(
            status_code=500,
            detail="Invalid data format in analysis file",
            headers={"Content-Type": "application/json"}
        )
    except Exception as e:
        logger.error(f"Error in /analysis_data: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
            headers={"Content-Type": "application/json"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")