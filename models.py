from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, scoped_session
from datetime import datetime
import os

Base = declarative_base()

class Camera(Base):
    __tablename__ = 'cameras'
    
    id = Column(Integer, primary_key=True)
    external_id = Column(Integer, nullable=True)
    name = Column(String)
    source_type = Column(String, default='stream')  # 'stream' | 'file' | 'api'
    stream_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    last_sync_time = Column(DateTime, nullable=True)
    
    sessions = relationship("VideoStreamSession", back_populates="camera")
    polygons = relationship("Polygon", back_populates="camera")
    zones = relationship("Zone", back_populates="camera")

class VideoStreamSession(Base):
    __tablename__ = 'video_stream_sessions'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    source_type = Column(String)  # 'stream' | 'file'
    
    camera = relationship("Camera", back_populates="sessions")
    detections = relationship("Detection", back_populates="session")

class Detection(Base):
    __tablename__ = 'detections'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('video_stream_sessions.id'))
    track_id = Column(Integer, ForeignKey('tracks.id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pixel_coords = Column(JSON)  # {'x': ..., 'y': ..., 'w': ..., 'h': ...} или {'lat': ..., 'lon': ...}
    confidence = Column(Float)
    frame_index = Column(Integer, nullable=True)
    vehicle_type = Column(String)
    direction = Column(String)
    frame_size = Column(JSON, nullable=True)  # [width, height]
    
    session = relationship("VideoStreamSession", back_populates="detections")
    track = relationship("Track", back_populates="detections")

class Track(Base):
    __tablename__ = 'tracks'
    
    id = Column(Integer, primary_key=True)
    vehicle_type = Column(String)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    confidence_avg = Column(Float)
    direction = Column(String)
    pixel_coords_start = Column(JSON)
    pixel_coords_end = Column(JSON)
    
    detections = relationship("Detection", back_populates="track")

class Polygon(Base):
    __tablename__ = 'polygons'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'))
    direction = Column(String)
    geometry = Column(JSON)
    
    camera = relationship("Camera", back_populates="polygons")
    zones = relationship("Zone", back_populates="polygon")

class Zone(Base):
    __tablename__ = 'zones'
    
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.id'), nullable=True)
    polygon_id = Column(Integer, ForeignKey('polygons.id'), nullable=True)
    name = Column(String)
    points = Column(JSON)
    
    camera = relationship("Camera", back_populates="zones")
    polygon = relationship("Polygon", back_populates="zones")

def init_db(connection_string=None):
    """Инициализация подключения к БД"""
    db_url = connection_string or os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:your_strong_password@localhost:5432/video_analysis"
    )
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

# Глобальная сессия для использования в приложении
Session = init_db()
db_session = scoped_session(Session)