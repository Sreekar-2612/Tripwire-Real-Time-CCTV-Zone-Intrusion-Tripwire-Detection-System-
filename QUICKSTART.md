# Tripwire Quick Start Guide

Get Tripwire up and running in minutes!

## Prerequisites

- Python 3.10+
- Node.js 18+
- Git

## Option 1: Local Development (Recommended for Development)

### Backend

```bash
# Clone repository
git clone <repo-url>
cd TRIP-WIRE/backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run server
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
# Open new terminal
cd TRIP-WIRE/frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

Access the app at:
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Option 2: Docker (Recommended for Production)

```bash
# Clone repository
git clone <repo-url>
cd TRIP-WIRE

# Copy environment file
cp .env.example backend/.env

# Edit backend/.env with your settings (optional)

# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Access the app at http://localhost

## First Time Setup

1. **Open the app** - Navigate to http://localhost:5173
2. **Add a camera source** - Go to "Sources" tab → "Add Source"
   - Webcam: `0` or camera index
   - RTSP: `rtsp://username:password@ip:port/stream`
   - Video file: Upload a video file
3. **Define zones** - Go to "Monitor" tab → Select stream → Draw zones/tripwires
4. **Configure alerts** - Go to "Settings" tab to set up notifications
5. **Start monitoring** - Watch for intrusions in real-time!

## Tips

- **Performance**: Use GPU if available (automatically detected)
- **False positives**: Increase `DEBOUNCE_FRAMES` or adjust `CONFIDENCE_THRESHOLD`
- **Alert spam**: Increase `COOLDOWN_SECONDS`
- **Testing**: Use a video file for testing without real cameras

## Troubleshooting

### Backend won't start
- Check if port 8000 is in use
- Ensure all dependencies are installed
- Check if YOLO model is accessible

### No video feed
- Verify source URL is correct
- Check camera permissions
- For RTSP, ensure network connectivity

### High CPU usage
- Set `FRAME_SKIP` > 0
- Lower `IMGSZ` (e.g., 416 instead of 640)
- Use GPU if available

## Need Help?

- Check [README.md](README.md) for detailed documentation
- Open an issue on GitHub
- Check API docs at http://localhost:8000/docs
