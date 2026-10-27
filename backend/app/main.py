import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.config import settings
from backend.app.db import init_db
from backend.app.api import streams, zones, events, ws, settings as settings_api
from backend.app.core.logger import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger("tripwire.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Tripwire CCTV Intrusion System...")
    init_db()
    streams.init_active_streams()
    yield
    logger.info("Shutting down Tripwire... Stopping active stream pipelines.")
    for sid, pipeline in list(streams.active_pipelines.items()):
        pipeline.stop()

app = FastAPI(
    title=settings.APP_NAME,
    description="Real-time CCTV Tripwire & Zone Intrusion Detection API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(streams.router)
app.include_router(zones.router)
app.include_router(events.router)
app.include_router(settings_api.router)
app.include_router(ws.router)

# Mount static files for direct snapshot / recording assets
app.mount("/static", StaticFiles(directory=str(settings.DATA_DIR)), name="static")

@app.get("/api/health")
def health_check():
    from datetime import datetime
    import torch
    import cv2
    from ultralytics import YOLO
    
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "active_streams": len(streams.active_pipelines),
        "system": {
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
            "opencv_version": cv2.__version__,
            "pytorch_version": torch.__version__
        }
    }

# Mount frontend production build if available
frontend_dist = settings.DATA_DIR.parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    from fastapi.responses import FileResponse

    @app.get("/")
    @app.get("/monitor")
    @app.get("/sources")
    @app.get("/events")
    @app.get("/settings")
    async def serve_frontend():
        return FileResponse(frontend_dist / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
