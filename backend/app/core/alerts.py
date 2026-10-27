import os
import smtplib
import logging
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from typing import Optional, Dict, Any
import httpx

from backend.app.config import settings

logger = logging.getLogger("tripwire.alerts")

class AlertManager:
    """Dispatches notifications across Telegram, Webhook, and Email channels."""

    @staticmethod
    async def dispatch_alert(
        event_data: Dict[str, Any],
        snapshot_path: Optional[str] = None
    ):
        """Sends notifications to all configured active channels concurrently."""
        tasks = []

        # Telegram
        if settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID:
            tasks.append(
                AlertManager.send_telegram_alert(
                    settings.TELEGRAM_BOT_TOKEN,
                    settings.TELEGRAM_CHAT_ID,
                    event_data,
                    snapshot_path
                )
            )

        # Webhook
        if settings.WEBHOOK_URL:
            tasks.append(
                AlertManager.send_webhook_alert(
                    settings.WEBHOOK_URL,
                    event_data,
                    snapshot_path
                )
            )

        # Email
        if (
            settings.SMTP_HOST
            and settings.ALERT_EMAIL_TO
            and settings.ALERT_EMAIL_FROM
        ):
            tasks.append(
                AlertManager.send_email_alert(
                    host=settings.SMTP_HOST,
                    port=settings.SMTP_PORT,
                    user=settings.SMTP_USER,
                    password=settings.SMTP_PASSWORD,
                    to_email=settings.ALERT_EMAIL_TO,
                    from_email=settings.ALERT_EMAIL_FROM,
                    event_data=event_data,
                    snapshot_path=snapshot_path
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def send_telegram_alert(
        bot_token: str,
        chat_id: str,
        event_data: Dict[str, Any],
        snapshot_path: Optional[str] = None
    ) -> bool:
        caption = (
            f"🚨 *TRIPWIRE ALARM TRIGGERED!*\n"
            f"• *Event:* {event_data.get('event_type')}\n"
            f"• *Zone:* {event_data.get('zone_name')}\n"
            f"• *Track ID:* #{event_data.get('track_id')}\n"
            f"• *Time:* {event_data.get('timestamp')}\n"
            f"• *Confidence:* {event_data.get('confidence', 0.0):.2f}"
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if snapshot_path and os.path.exists(snapshot_path):
                    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
                    with open(snapshot_path, "rb") as f:
                        files = {"photo": (os.path.basename(snapshot_path), f, "image/jpeg")}
                        data = {"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"}
                        resp = await client.post(url, data=data, files=files)
                else:
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    data = {"chat_id": chat_id, "text": caption, "parse_mode": "Markdown"}
                    resp = await client.post(url, json=data)

                if resp.status_code == 200:
                    logger.info("Telegram alert dispatched successfully")
                    return True
                else:
                    logger.error(f"Telegram alert failed: {resp.status_code} - {resp.text}")
                    return False
        except Exception as e:
            logger.error(f"Telegram notification exception: {e}")
            return False

    @staticmethod
    async def send_webhook_alert(
        webhook_url: str,
        event_data: Dict[str, Any],
        snapshot_path: Optional[str] = None
    ) -> bool:
        payload = {
            "source": "Tripwire CCTV Intrusion System",
            "event": event_data,
            "has_snapshot": bool(snapshot_path and os.path.exists(snapshot_path))
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(webhook_url, json=payload)
                if 200 <= resp.status_code < 300:
                    logger.info("Webhook alert dispatched successfully")
                    return True
                else:
                    logger.error(f"Webhook alert returned status: {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Webhook exception: {e}")
            return False

    @staticmethod
    async def send_email_alert(
        host: str,
        port: int,
        user: Optional[str],
        password: Optional[str],
        to_email: str,
        from_email: str,
        event_data: Dict[str, Any],
        snapshot_path: Optional[str] = None
    ) -> bool:
        def _send():
            msg = MIMEMultipart()
            msg["Subject"] = f"[ALARM] Tripwire Intrusion Detected: {event_data.get('zone_name')}"
            msg["From"] = from_email
            msg["To"] = to_email

            body = (
                f"Tripwire Security Alert\n\n"
                f"Intrusion detected at zone: {event_data.get('zone_name')}\n"
                f"Event Type: {event_data.get('event_type')}\n"
                f"Track ID: {event_data.get('track_id')}\n"
                f"Timestamp: {event_data.get('timestamp')}\n"
                f"Confidence: {event_data.get('confidence', 0.0):.2f}\n"
            )
            msg.attach(MIMEText(body, "plain"))

            if snapshot_path and os.path.exists(snapshot_path):
                with open(snapshot_path, "rb") as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(snapshot_path))
                    msg.attach(img)

            with smtplib.SMTP(host, port) as server:
                server.ehlo()
                server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(msg)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _send)
            logger.info("Email alert dispatched successfully")
            return True
        except Exception as e:
            logger.error(f"Email dispatch exception: {e}")
            return False
