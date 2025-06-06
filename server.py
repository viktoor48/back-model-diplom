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
    return list(analyzer.cameras.values())

@app.post("/stop_analysis")
async def stop_analysis():
    if processor.processing:
        processor.processing = False
        if processor.cap:  # Принудительно останавливаем VideoCapture
            processor.cap.release()
        logger.info("Обработка видео принудительно остановлена")
    return {"status": "processing_stopped"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")