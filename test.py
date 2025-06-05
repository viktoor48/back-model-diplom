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
        self.lock = threading.Lock()  # Added lock for thread safety

    def process_video(self, file_path: str, camera_id: int):
        self.processing = True
        cap = cv2.VideoCapture(file_path)
        
        try:
            while self.processing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                result = analyzer.analyze_frame(frame, camera_id)
                
                # Use lock when updating shared variables
                with self.lock:
                    self.current_frame = result['frame']
                    self.detections = result['detections']
                
                time.sleep(0.033)  # ~30 FPS
        finally:
            cap.release()
            if os.path.exists(file_path):
                os.remove(file_path)
            self.processing = False

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")