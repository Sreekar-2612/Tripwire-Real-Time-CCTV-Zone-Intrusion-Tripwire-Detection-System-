import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from backend.app.db import Base

class Stream(Base):
    __tablename__ = "streams"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    source_type = Column(String(20), nullable=False)  # 'webcam', 'rtsp', 'file'
    source_url = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=True)
    loop_file = Column(Boolean, default=True)
    fps = Column(Float, default=25.0)
    width = Column(Integer, default=1280)
    height = Column(Integer, default=720)
    created_at = Column(DateTime, default=datetime.utcnow)

    zones = relationship("Zone", back_populates="stream", cascade="all, delete-orphan")
    events = relationship("Event", back_populates="stream", cascade="all, delete-orphan")

class Zone(Base):
    __tablename__ = "zones"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_id = Column(String(36), ForeignKey("streams.id"), nullable=False)
    name = Column(String(100), nullable=False)
    zone_type = Column(String(20), default="polygon")  # 'polygon' or 'tripwire'
    coordinates = Column(Text, nullable=False)  # JSON string of [[x, y], ...] normalized 0..1
    armed = Column(Boolean, default=True)
    color = Column(String(20), default="#EF4444")  # hex color for rendering
    schedule_start = Column(String(10), nullable=True)  # 'HH:MM' or null
    schedule_end = Column(String(10), nullable=True)    # 'HH:MM' or null
    created_at = Column(DateTime, default=datetime.utcnow)

    stream = relationship("Stream", back_populates="zones")
    events = relationship("Event", back_populates="zone", cascade="all, delete-orphan")

class Event(Base):
    __tablename__ = "events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    stream_id = Column(String(36), ForeignKey("streams.id"), nullable=False)
    zone_id = Column(String(36), ForeignKey("zones.id"), nullable=True)
    track_id = Column(Integer, nullable=False)
    event_type = Column(String(20), nullable=False)  # 'ENTER', 'EXIT', 'LOITER', 'CROSS_IN', 'CROSS_OUT'
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, default=0.0)
    snapshot_path = Column(String(500), nullable=True)
    clip_path = Column(String(500), nullable=True)
    details = Column(Text, nullable=True)  # JSON extra metadata

    stream = relationship("Stream", back_populates="events")
    zone = relationship("Zone", back_populates="events")

class SystemSetting(Base):
    __tablename__ = "system_settings"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=False)
