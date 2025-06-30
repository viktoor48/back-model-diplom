import json
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from server import Base, Camera, Polygon, AnalysisResult  # Импорт из вашего server.py

# Конфигурация
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://postgres:your_strong_password@db:5432/video_analysis')

def init_db(db_url):
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    return Session

def migrate_cameras(session):
    """Миграция данных камер"""
    with open(os.path.join(DATA_DIR, 'cameras.json')) as f:
        cameras = json.load(f)
        
        for cam in cameras:
            try:
                camera = Camera(
                    id=cam['id'],
                    name=cam['name'],
                    location=cam.get('location', ''),
                    stream_url=cam['stream_url'],
                    is_active=cam.get('is_active', True)
                )
                session.add(camera)
                session.commit()
                print(f"Добавлена камера: {cam['name']}")
            except Exception as e:
                session.rollback()
                print(f"Ошибка при добавлении камеры {cam['name']}: {str(e)}")

def migrate_polygons(session):
    """Миграция полигонов с обработкой разных форматов"""
    try:
        with open(os.path.join(DATA_DIR, 'polygons.geojson')) as f:
            data = json.load(f)
            
            # Если это FeatureCollection
            if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
                features = data.get('features', [])
            # Если это просто массив полигонов
            elif isinstance(data, list):
                features = data
            else:
                features = [data]
            
            for item in features:
                try:
                    # Обработка разных форматов
                    if isinstance(item, dict):
                        if 'geometry' in item:  # GeoJSON feature
                            props = item.get('properties', {})
                            polygon = Polygon(
                                id=props.get('id', 0),
                                camera_id=props.get('camera_id', 1),
                                name=props.get('name', 'Unnamed'),
                                coordinates=item['geometry']
                            )
                        else:  # Простой словарь
                            polygon = Polygon(
                                id=item.get('id'),
                                camera_id=item.get('camera_id'),
                                name=item.get('name'),
                                coordinates=item.get('coordinates', {})
                            )
                    else:  # Неподдерживаемый формат
                        continue
                        
                    session.add(polygon)
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"Ошибка при добавлении полигона: {str(e)}")
                    
    except Exception as e:
        print(f"Ошибка чтения файла полигонов: {str(e)}")

def migrate_analysis_results(session):
    """Миграция результатов анализа"""
    with open(os.path.join(DATA_DIR, 'analysis.json')) as f:
        results = json.load(f)
        
        for res in results:
            try:
                result = AnalysisResult(
                    camera_id=res['camera_id'],
                    track_id=res['track_id'],
                    vehicle_type=res['vehicle_type'],
                    direction=res['direction'],
                    confidence=res.get('confidence', 0.9),
                    weight=res.get('weight', 0),
                    timestamp=datetime.fromisoformat(res['timestamp'].replace('Z', '+00:00'))
                )
                session.add(result)
                session.commit()
            except Exception as e:
                session.rollback()
                print(f"Ошибка при добавлении результата: {str(e)}")

if __name__ == '__main__':
    print("🔄 Начало миграции данных...")
    
    # Инициализация подключения
    engine = create_engine(DATABASE_URL)
    Session = init_db(DATABASE_URL)
    session = Session()
    
    try:
        # Создание таблиц (если они еще не созданы)
        Base.metadata.create_all(engine)
        
        print("🔄 Миграция камер...")
        migrate_cameras(session)
        print("✅ Камеры успешно мигрированы")
        
        print("🔄 Миграция полигонов...")
        migrate_polygons(session)
        print("✅ Полигоны успешно мигрированы")
        
        print("🔄 Миграция результатов анализа...")
        migrate_analysis_results(session)
        print("✅ Результаты анализа успешно мигрированы")
        
        print("\n🎉 Все миграции успешно завершены!")
    except Exception as e:
        session.rollback()
        print(f"\n❌ Критическая ошибка: {str(e)}")
    finally:
        session.close()