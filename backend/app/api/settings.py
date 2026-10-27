import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.config import settings
from backend.app.db import get_db
from backend.app.models import SystemSetting
from backend.app.schemas import SettingsSchema, TestAlertRequest
from backend.app.core.alerts import AlertManager
from backend.app.api.streams import active_pipelines

logger = logging.getLogger("tripwire.api.settings")
router = APIRouter(prefix="/api/settings", tags=["settings"])

@router.get("", response_model=SettingsSchema)
def get_settings():
    return SettingsSchema(
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        cooldown_seconds=settings.COOLDOWN_SECONDS,
        loitering_seconds=settings.LOITERING_SECONDS,
        debounce_frames=settings.DEBOUNCE_FRAMES,
        frame_skip=settings.FRAME_SKIP,
        telegram_bot_token=settings.TELEGRAM_BOT_TOKEN,
        telegram_chat_id=settings.TELEGRAM_CHAT_ID,
        webhook_url=settings.WEBHOOK_URL,
        smtp_host=settings.SMTP_HOST,
        smtp_port=settings.SMTP_PORT,
        smtp_user=settings.SMTP_USER,
        smtp_password=settings.SMTP_PASSWORD,
        alert_email_to=settings.ALERT_EMAIL_TO,
        alert_email_from=settings.ALERT_EMAIL_FROM,
        enable_audio_alert=settings.ENABLE_AUDIO_ALERT
    )

@router.put("", response_model=SettingsSchema)
def update_settings(cfg: SettingsSchema, db: Session = Depends(get_db)):
    # Update runtime settings
    settings.CONFIDENCE_THRESHOLD = cfg.confidence_threshold
    settings.COOLDOWN_SECONDS = cfg.cooldown_seconds
    settings.LOITERING_SECONDS = cfg.loitering_seconds
    settings.DEBOUNCE_FRAMES = cfg.debounce_frames
    settings.FRAME_SKIP = cfg.frame_skip
    settings.TELEGRAM_BOT_TOKEN = cfg.telegram_bot_token
    settings.TELEGRAM_CHAT_ID = cfg.telegram_chat_id
    settings.WEBHOOK_URL = cfg.webhook_url
    settings.SMTP_HOST = cfg.smtp_host
    settings.SMTP_PORT = cfg.smtp_port
    settings.SMTP_USER = cfg.smtp_user
    settings.SMTP_PASSWORD = cfg.smtp_password
    settings.ALERT_EMAIL_TO = cfg.alert_email_to
    settings.ALERT_EMAIL_FROM = cfg.alert_email_from
    settings.ENABLE_AUDIO_ALERT = cfg.enable_audio_alert

    # Propagate confidence threshold to all running detectors
    for p in active_pipelines.values():
        if hasattr(p, "detector"):
            p.detector.set_confidence(cfg.confidence_threshold)

    # Persist into DB
    for k, v in cfg.model_dump().items():
        row = db.query(SystemSetting).filter(SystemSetting.key == k).first()
        val_str = json.dumps(v)
        if row:
            row.value = val_str
        else:
            db.add(SystemSetting(key=k, value=val_str))
    db.commit()

    return cfg

@router.post("/test-alert")
async def send_test_alert(req: TestAlertRequest):
    test_event = {
        "event_type": "TEST_ALERT",
        "zone_name": "Test Zone",
        "track_id": 999,
        "timestamp": "2026-10-27 22:00:00",
        "confidence": 0.99
    }

    channel = req.channel.lower()
    success = False
    message = ""

    if channel == "telegram":
        if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
            raise HTTPException(status_code=400, detail="Telegram bot token and chat ID must be configured.")
        success = await AlertManager.send_telegram_alert(
            settings.TELEGRAM_BOT_TOKEN,
            settings.TELEGRAM_CHAT_ID,
            test_event
        )
        message = "Telegram test alert sent successfully!" if success else "Failed to send Telegram test alert. Check token and chat ID."

    elif channel == "webhook":
        if not settings.WEBHOOK_URL:
            raise HTTPException(status_code=400, detail="Webhook URL must be configured.")
        success = await AlertManager.send_webhook_alert(settings.WEBHOOK_URL, test_event)
        message = "Webhook ping sent successfully!" if success else "Failed to send Webhook ping. Check endpoint URL."

    elif channel == "email":
        if not settings.SMTP_HOST or not settings.ALERT_EMAIL_TO or not settings.ALERT_EMAIL_FROM:
            raise HTTPException(status_code=400, detail="SMTP Host, Recipient, and Sender email must be configured.")
        success = await AlertManager.send_email_alert(
            host=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            user=settings.SMTP_USER,
            password=settings.SMTP_PASSWORD,
            to_email=settings.ALERT_EMAIL_TO,
            from_email=settings.ALERT_EMAIL_FROM,
            event_data=test_event
        )
        message = "Email test alert sent successfully!" if success else "Failed to send Email. Check SMTP settings."

    else:
        raise HTTPException(status_code=400, detail=f"Unknown alert channel: {channel}")

    return {"status": "success" if success else "failed", "message": message}
