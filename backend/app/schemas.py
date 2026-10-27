import json
from datetime import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, Field, field_validator

# Normalized point [x, y] in range [0.0, 1.0]
Point2D = List[float]

# Zone Schemas
class ZoneBase(BaseModel):
    name: str = "Zone 1"
    zone_type: str = Field(default="polygon", description="'polygon' or 'tripwire'")
    coordinates: List[Point2D] = Field(..., description="List of [x, y] points normalized 0.0 to 1.0")
    armed: bool = True
    color: str = "#EF4444"
    schedule_start: Optional[str] = None
    schedule_end: Optional[str] = None

    @field_validator("coordinates", mode="before")
    @classmethod
    def parse_coordinates(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return []
        return v

class ZoneCreate(ZoneBase):
    pass

class ZoneResponse(ZoneBase):
    id: str
    stream_id: str
    created_at: datetime

    class Config:
        from_attributes = True

# Stream Schemas
class StreamBase(BaseModel):
    name: str
    source_type: str = Field(..., description="'webcam', 'rtsp', 'file'")
    source_url: str
    is_active: bool = True
    loop_file: bool = True

class StreamCreate(StreamBase):
    pass

class StreamResponse(StreamBase):
    id: str
    fps: float
    width: int
    height: int
    created_at: datetime
    zones: List[ZoneResponse] = []

    class Config:
        from_attributes = True

# Event Schemas
class EventResponse(BaseModel):
    id: str
    stream_id: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = None
    track_id: int
    event_type: str
    timestamp: datetime
    confidence: float
    snapshot_url: Optional[str] = None
    clip_url: Optional[str] = None
    details: Optional[Any] = None

    class Config:
        from_attributes = True

# Settings Schema
class SettingsSchema(BaseModel):
    confidence_threshold: float = Field(0.40, ge=0.05, le=1.0)
    cooldown_seconds: float = Field(8.0, ge=0.5, le=120.0)
    loitering_seconds: float = Field(6.0, ge=1.0, le=300.0)
    debounce_frames: int = Field(2, ge=1, le=20)
    frame_skip: int = Field(0, ge=0, le=10)
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    webhook_url: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email_to: Optional[str] = None
    alert_email_from: Optional[str] = None
    enable_audio_alert: bool = True

class TestAlertRequest(BaseModel):
    channel: str = Field(..., description="'telegram', 'webhook', 'email'")

# WebSocket Alert Message Schema
class WSAlertMessage(BaseModel):
    type: str = "ALERT"
    event_id: str
    stream_id: str
    zone_id: Optional[str] = None
    zone_name: Optional[str] = "Zone"
    track_id: int
    event_type: str
    timestamp: str
    snapshot_url: Optional[str] = None
    clip_url: Optional[str] = None
    confidence: float
    message: str
