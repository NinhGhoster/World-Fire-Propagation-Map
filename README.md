# 🔥 World Fire Propagation Map

Real-time wildfire tracking and Moving Firefighter Problem (MFP) optimization dashboard.

## Features

- **Real-time Fire Tracking**: NASA FIRMS satellite data (MODIS & VIIRS)
- **Interactive Map**: Dash + Plotly visualization with zoom, pan, and point selection
- **Fire Propagation Modeling**: Grid-based fire spread simulation
- **MFP Optimization**: Solve the Moving Firefighter Problem with SCIP/MIQCP
- **Multi-parameter Analysis**: Adjust D (spread rate), B (budget), and λ (speed ratio)
- **Cloud Deployment**: Azure App Service with GitHub Actions CI/CD

## Quick Start

### Prerequisites

- Python 3.11+
- NASA FIRMS API key (free from [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/api/))

### Installation

```bash
# Clone the repository
git clone https://github.com/NinhGhoster/World-Fire-Propagation-Map.git
cd World-Fire-Propagation-Map

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your FIRMS_API_KEY

# Run the application
python app.py
```

The app will be available at `http://localhost:8050`

### Docker

```bash
# Build and run with Docker
docker-compose up --build

# Or for production with gunicorn
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```

## Usage

### 1. Select Country & Date

Choose a country from the dropdown and select an analysis date.

### 2. Select Fire Point

Click on a fire hotspot on the map to select it for analysis.

### 3. Configure Parameters

- **Grid Graph Size**: 3x3 to 9x9 grid overlay
- **Grid Spacing**: Distance between grid nodes
- **D Value**: Fire spread rate (1-10)
- **B Value**: Firefighter budget (1-10)
- **Lambda**: Firefighter speed ratio (0.1-5.0)

### 4. Run Analysis

Click "Analyze Selected Fire Point" to fetch data and generate the grid.

### 5. Deploy Firefighters

- Toggle the grid graph
- Click on grid nodes to select firefighter stations
- Save the graph data
- Run the MFF solver to optimize deployment

## Project Structure

```
World-Fire-Propagation-Map/
├── app.py                      # Main Dash application entry point
├── config.py                   # Configuration management
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
├── .env.example               # Environment template
├── .gitignore                 # Git ignore rules
├── CONTRIBUTING.md            # Contribution guidelines
│
├── modules/                   # Core application modules
│   ├── __init__.py
│   ├── callbacks.py          # Dash callbacks (interactivity)
│   ├── layout.py             # Dash layout (UI components)
│   ├── data_fetcher.py       # FIRMS API data fetching
│   ├── analysis_pipeline.py  # Data processing pipeline
│   ├── mff_integration.py    # MFP solver integration
│   ├── plotly_visuals.py     # Plotly visualization helpers
│   ├── simulation.py         # Fire spread simulation
│   └── logger.py             # Logging configuration
│
├── assets/                    # Static assets (CSS, images)
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_data_fetcher.py
│   └── test_analysis_pipeline.py
│
├── moving_firefighter_problem_generator/  # MFP algorithm
├── movingff_paper/              # Research paper implementations
├── mff_solution_*.json         # Solution outputs
└── mfp_*.json                  # Problem instances
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FIRMS_API_KEY` | NASA FIRMS API key | Required |
| `DEBUG` | Enable debug mode | `false` |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

### Azure Deployment

The app is configured for Azure App Service deployment:

```bash
# Deploy via GitHub Actions (automatic on push to main)
# Or manually:
az webapp up --name fire-propagation-map --resource-group your-rg --runtime "PYTHON:3.12"
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=modules --cov-report=html

# Run specific test
pytest tests/test_config.py -v
```

## The Science Behind It

### Moving Firefighter Problem (MFP)

The MFP is a classic combinatorial optimization problem:

> Given a spreading fire on a graph, where should you place firefighters each turn to protect the most area?

**Problem Parameters:**
- **n**: Number of nodes in the graph
- **λ (lambda)**: Fire spread probability
- **B**: Number of firefighters available per turn
- **D**: Fire spread rate

**Key Insight:** Protecting high-degree nodes (chokepoints) is optimal for grid graphs. This is known as the "betweenness centrality heuristic."

### References

- Hartnell, B. (1995). Firefighter problem: A survey. *Congressus Numerantium*.
- Gutiérrez-De-La-Paz, B. R., & García-Díaz, J. (2022). The moving firefighter problem. *Mathematics*, 11(1), 179.
- Finney, M. A. (2003). Fire spread algorithms for the FARSITE fire area simulator.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- NASA FIRMS for satellite fire data
- SCIP Optimization Suite for MILP solver
- Dash/Plotly team for the visualization framework

---

Built with ❤️ for wildfire research and emergency management
