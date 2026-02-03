# Contributing to World Fire Propagation Map

## Quick Start

```bash
# Clone the repository
git clone https://github.com/NinhGhoster/World-Fire-Propagation-Map.git
cd World-Fire-Propagation-Map

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## Project Structure

- `app.py` - Main Dash application entry point
- `modules/` - Core modules (callbacks, layout, data fetcher, etc.)
- `moving_firefighter_problem_generator/` - MFP optimization algorithms
- `movingff_paper/` - Research paper implementations
- `assets/` - Static assets (CSS, images)

## Adding New Features

1. For UI changes: Edit `modules/layout.py`
2. For interactivity: Edit `modules/callbacks.py`
3. For data processing: Edit `modules/data_fetcher.py`
4. For optimization: Add to `moving_firefighter_problem_generator/`

## Coding Standards

- Follow PEP 8
- Add docstrings to new functions
- Use type hints where helpful

## Testing

```bash
pytest tests/  # If tests exist
```
