import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "Tripwire CCTV Zone Intrusion"
    DEBUG: bool = False
    
    # Paths & Storage
    DATA_DIR: Path = BASE_DIR / "data"
    SNAPSHOTS_DIR: Path = BASE_DIR / "data" / "snapshots"
    CLIPS_DIR: Path = BASE_DIR / "data" / "clips"
    RECORDINGS_DIR: Path = BASE_DIR / "data" / "recordings"
    UPLOADS_DIR: Path = BASE_DIR / "data" / "uploads"
    MODELS_DIR: Path = BASE_DIR / "data" / "models"
    
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/tripwire.db"
    
    # Computer Vision & Tracking
    MODEL_NAME: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.40
    IOU_THRESHOLD: float = 0.45
    IMGSZ: int = 640
    FRAME_SKIP: int = 0
    TARGET_FPS: int = 25
    
    # Detection & Event Logic
    DEBOUNCE_FRAMES: int = 2
    COOLDOWN_SECONDS: float = 8.0
    LOITERING_SECONDS: float = 6.0
    
    # Clip Recorder Buffer (seconds)
    PRE_BUFFER_SECONDS: float = 3.0
    POST_BUFFER_SECONDS: float = 3.0
    
    # Alert Channels
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    WEBHOOK_URL: Optional[str] = None
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    ALERT_EMAIL_TO: Optional[str] = None
    ALERT_EMAIL_FROM: Optional[str] = None
    ENABLE_AUDIO_ALERT: bool = True

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False
    }

settings = Settings()

# Ensure directories exist
for folder in [
    settings.DATA_DIR,
    settings.SNAPSHOTS_DIR,
    settings.CLIPS_DIR,
    settings.RECORDINGS_DIR,
    settings.UPLOADS_DIR,
    settings.MODELS_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)
