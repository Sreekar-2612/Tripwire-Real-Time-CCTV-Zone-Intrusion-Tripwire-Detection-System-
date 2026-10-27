# Tripwire - AI-Powered CCTV Zone Intrusion Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.dev/)
[![React](https://img.shields.io/badge/React-19.3+-61DAFB.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-7.0+-3178C6.svg)](https://www.typescriptlang.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red.svg)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)

A real-time, AI-powered video surveillance system that detects and alerts on intrusions in defined zones using computer vision. Built with YOLOv8 object detection, ByteTrack multi-object tracking, and a modern web interface for monitoring and management.

## 🎯 Overview

Tripwire is an intelligent CCTV intrusion detection system designed for perimeter security, restricted area monitoring, and smart surveillance applications. It combines state-of-the-art computer vision with a user-friendly web dashboard to provide real-time monitoring, automated alerts, and comprehensive event logging.

## ✨ Key Features

### Computer Vision & Detection
- **Real-time Person Detection** - Powered by YOLOv8n for accurate human detection
- **Multi-Object Tracking** - ByteTrack algorithm maintains consistent person identities across frames
- **Zone-Based Monitoring** - Define custom polygon zones or tripwire lines for targeted monitoring
- **Tripwire Crossing Detection** - Directional crossing detection (IN/OUT) with precise boundary crossing analysis
- **Smart State Machine** - Debounced entry/exit detection with loitering alerts to minimize false positives
- **Track History Visualization** - Motion trails for better situational awareness

### Real-Time Monitoring
- **Live MJPEG Streaming** - Low-latency video feed with real-time annotations
- **Interactive Zone Editor** - Draw and configure zones directly from the web interface
- **WebSocket Alerts** - Instant browser notifications with audio alerts
- **Multi-Stream Support** - Monitor multiple cameras/sources simultaneously
- **Live Dashboard** - Real-time statistics, FPS monitoring, and active alerts

### Event Management & Recording
- **Automated Event Logging** - All intrusions, entries, exits, and loitering events are recorded
- **Snapshot Capture** - Automatic snapshot on security events
- **Pre/Post Event Recording** - Ring buffer captures footage before and after events
- **Event History** - Searchable, filterable event log with media playback
- **Clip Management** - Review and download event recordings

### Alerts & Notifications
- **In-App Alerts** - Toast notifications with event details
- **Audio Alerts** - Configurable alarm sounds
- **Multi-Channel Notifications** - Support for Telegram, Email (SMTP), and Webhooks
- **Cooldown Periods** - Prevent alert spam with configurable cooldowns
- **Scheduled Arming** - Time-based zone activation

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Video Sources │────▶│  CV Pipeline    │────▶│   Event Engine  │
│  (RTSP/File/Webcam)│   │ YOLOv8+ByteTrack│   │ Zone Logic+Alerts│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │                        │
                                ▼                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  MJPEG Stream   │     │   Database      │
                        │  Ring Buffer    │     │   SQLite/Events │
                        └─────────────────┘     └─────────────────┘
                                │                        │
                                └──────────┬─────────────┘
                                           ▼
                                    ┌─────────────────┐
                                    │   FastAPI Server │
                                    │   REST + WebSocket│
                                    └─────────────────┘
                                           │
                                           ▼
                                    ┌─────────────────┐
                                    │  React Frontend │
                                    │  TypeScript+Vite│
                                    └─────────────────┘
```

## 🛠️ Tech Stack

### Backend
- **[FastAPI](https://fastapi.dev/)** - Modern, high-performance async web framework
- **[SQLAlchemy](https://www.sqlalchemy.org/)** - SQL toolkit and ORM
- **[OpenCV](https://opencv.org/)** - Computer vision library
- **[Ultralytics YOLOv8](https://ultralytics.com/)** - State-of-the-art object detection
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)** - Multi-object tracking
- **[Shapely](https://shapely.readthedocs.io/)** - Geometric operations for zone detection
- **[PyTorch](https://pytorch.org/)** - Deep learning framework

### Frontend
- **[React 19](https://react.dev/)** - UI library
- **[TypeScript](https://www.typescriptlang.org/)** - Type-safe JavaScript
- **[Vite](https://vitejs.dev/)** - Fast build tool
- **[Tailwind CSS 4](https://tailwindcss.com/)** - Utility-first CSS framework
- **[Lucide React](https://lucide.dev/)** - Icon library

## 📋 Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm
- CUDA-capable GPU (recommended) or CPU
- For RTSP streams: network access to cameras

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/tripwire.git
cd tripwire
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model (will auto-download on first run, or download manually)
# The model yolov8n.pt should be in project root or models dir
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

### 4. Environment Configuration

Create a `.env` file in the `backend` directory (or use defaults):

```env
# Application
APP_NAME=Tripwire CCTV Zone Intrusion
DEBUG=False

# Alert Configuration
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
WEBHOOK_URL=your_webhook_url

# SMTP Email Alerts
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
ALERT_EMAIL_TO=recipient@example.com
ALERT_EMAIL_FROM=your_email@gmail.com

# Detection Settings
CONFIDENCE_THRESHOLD=0.40
IOU_THRESHOLD=0.45
DEBOUNCE_FRAMES=2
COOLDOWN_SECONDS=8.0
LOITERING_SECONDS=6.0
```

## 🎮 Usage

### Start Backend Server

```bash
cd backend
# Activate venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Run server
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- Interactive API docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### Start Frontend Development Server

```bash
cd frontend
npm run dev
```

The web interface will be available at `http://localhost:5173`

### Using the System

1. **Add a Stream** - Go to Sources tab and add a webcam, RTSP URL, or upload a video file
2. **Define Zones** - Select a stream in Monitor tab, use the zone editor to draw polygons or tripwires
3. **Configure Settings** - Adjust detection sensitivity, alert cooldowns, and notification channels
4. **Monitor** - Watch live feed with real-time detection and alerts
5. **Review Events** - Check Events tab for history, snapshots, and recordings

## 📁 Project Structure

```
TRIP-WIRE/
├── backend/
│   ├── app/
│   │   ├── api/          # FastAPI route handlers
│   │   ├── core/         # CV pipeline, detection, zones, alerts
│   │   ├── config.py     # Configuration settings
│   │   ├── db.py         # Database setup
│   │   ├── main.py       # Application entry point
│   │   ├── models.py     # SQLAlchemy models
│   │   └── schemas.py    # Pydantic schemas
│   ├── data/             # Runtime data (DB, snapshots, clips)
│   ├── tests/            # Unit tests
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── api/          # API client
│   │   ├── components/   # React components
│   │   ├── hooks/        # Custom React hooks
│   │   ├── pages/        # Page components
│   │   ├── types.ts      # TypeScript types
│   │   └── App.tsx       # Main app component
│   ├── package.json
│   └── vite.config.ts
└── yolov8n.pt           # YOLO model weights
```

## 🧪 Testing

### Backend Tests

```bash
cd backend
pytest tests/ -v
```

Current tests cover:
- Zone geometry and containment
- Tripwire crossing detection
- Zone scheduling
- State machine logic (enter/exit/debounce)
- Loitering detection

## 🔧 Configuration Options

| Setting | Default | Description |
|---------|---------|-------------|
| `CONFIDENCE_THRESHOLD` | 0.40 | Minimum confidence for person detection |
| `IOU_THRESHOLD` | 0.45 | IoU threshold for NMS |
| `DEBOUNCE_FRAMES` | 2 | Frames required to confirm zone entry |
| `COOLDOWN_SECONDS` | 8.0 | Minimum time between alerts for same zone/track |
| `LOITERING_SECONDS` | 6.0 | Time threshold for loitering alert |
| `PRE_BUFFER_SECONDS` | 3.0 | Seconds of footage before event |
| `POST_BUFFER_SECONDS` | 3.0 | Seconds of footage after event |
| `FRAME_SKIP` | 0 | Skip frames for performance (0 = no skip) |

## 🚧 Roadmap

- [ ] User authentication and role-based access control
- [ ] Docker containerization
- [ ] Database migrations with Alembic
- [ ] Enhanced analytics dashboard with charts
- [ ] Face recognition capabilities
- [ ] Mobile app support
- [ ] Cloud storage integration
- [ ] RTSP stream health monitoring
- [ ] AI-powered false alarm reduction
- [ ] Kubernetes deployment support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking
- [FastAPI](https://fastapi.dev/) and [React](https://react.dev/) communities
- All open-source contributors

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub.

---

**Built with ❤️ for intelligent surveillance and security applications**
