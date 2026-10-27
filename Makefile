.PHONY: help install dev build test clean docker-up docker-down

help:
	@echo "Tripwire - Available Commands"
	@echo "================================"
	@echo "install        Install all dependencies"
	@echo "dev            Run development servers (backend + frontend)"
	@echo "backend        Run only backend server"
	@echo "frontend       Run only frontend server"
	@echo "test           Run all tests"
	@echo "test-backend   Run backend tests"
	@echo "build          Build frontend for production"
	@echo "clean          Clean build artifacts and cache"
	@echo "docker-up      Start with Docker Compose"
	@echo "docker-down    Stop Docker Compose"
	@echo "lint           Lint code"

install:
	@echo "Installing backend dependencies..."
	cd backend && python -m pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Installation complete!"

backend:
	@echo "Starting backend server..."
	cd backend && uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	@echo "Starting frontend server..."
	cd frontend && npm run dev

dev:
	@echo "Starting both servers..."
	@echo "Run 'make backend' and 'make frontend' in separate terminals"

test:
	@echo "Running all tests..."
	cd backend && pytest tests/ -v

test-backend:
	@echo "Running backend tests..."
	cd backend && pytest tests/ -v --tb=short

build:
	@echo "Building frontend..."
	cd frontend && npm run build

clean:
	@echo "Cleaning..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf frontend/dist frontend/node_modules/.cache
	rm -rf backend/.pytest_cache backend/htmlcov

docker-up:
	@echo "Starting with Docker..."
	docker-compose up -d
	@echo "App available at http://localhost"

docker-down:
	@echo "Stopping Docker..."
	docker-compose down

lint:
	@echo "Linting frontend..."
	cd frontend && npm run lint
	@echo "Linting backend..."
	cd backend && python -m flake8 app/ --max-line-length=100 2>/dev/null || echo "flake8 not installed"

status:
	@echo "Project Status:"
	@echo "- Backend: FastAPI on port 8000"
	@echo "- Frontend: Vite dev server on port 5173"
	@echo "- Database: SQLite"
	@echo "- Model: YOLOv8n"
