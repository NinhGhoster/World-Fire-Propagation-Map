# World Fire Propagation Map - Development Commands

.PHONY: help install dev test lint clean run docker

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make dev         - Install development dependencies"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run code quality checks"
	@echo "  make format      - Format code with black and isort"
	@echo "  make clean       - Clean up cache files"
	@echo "  make run         - Run the application"
	@echo "  make docker      - Build and run with Docker"
	@echo "  make docker-prod - Build for production"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt -r requirements-dev.txt

test:
	pytest tests/ -v --cov=modules --cov-report=term-missing

test-coverage:
	pytest tests/ --cov=modules --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

lint:
	flake8 modules/ --max-line-length=100 --extend-ignore=E203
	mypy modules/ --ignore-missing-imports

format:
	black modules/ tests/ --line-length 100
	isort modules/ tests/ --profile black

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".pytest_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/

run:
	python app.py

docker:
	docker-compose up --build

docker-prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

docker-stop:
	docker-compose down

check:
	@echo "Checking configuration..."
	@python -c "from config import Config; Config.validate()"
	@echo "✓ Configuration valid"
