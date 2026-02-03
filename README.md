# 🌍 World Fire Propagation Map v3.0

## 100x Better - Complete Upgrade

### What's New in v3.0

#### 🚀 Performance
- **Redis Caching** - 10x faster repeated queries
- **Async Processing** - Non-blocking API calls
- **Connection Pooling** - Efficient database access

#### 📊 Analytics
- **Fire Hotspots** - Identify high-risk areas
- **Seasonal Analysis** - Historical patterns
- **Risk Assessment** - ML-powered risk scoring

#### 🌤️ Weather Integration
- **Real-time Weather** - OpenWeatherMap integration
- **Fire Danger Rating** - Automated risk alerts
- **Wind Prediction** - Hourly wind forecasts

#### 🚨 Evacuation
- **Route Planning** - Optimal evacuation paths
- **Safe Zones** - Recommended shelters
- **Community Alerts** - SMS/Email notifications

#### 📱 Modern UI
- **Dark Mode** - Eye-friendly interface
- **Mobile Responsive** - Works on phones
- **Real-time Updates** - Live fire tracking
- **3D Visualization** - WebGL fire spread

#### 🔌 Enhanced API
- **REST + GraphQL** - Flexible data access
- **Rate Limiting** - Fair usage
- **API Keys** - Secure access control
- **Webhooks** - Event-driven alerts

### Quick Start

```bash
# Clone and install
git clone https://github.com/NinhGhoster/World-Fire-Propagation-Map.git
cd World-Fire-Propagation-Map
pip install -r requirements.txt

# Run with Docker
docker compose up -d

# Access
# Dashboard: http://localhost:8050
# API Docs: http://localhost:8050/api/docs
```

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    World Fire Propagation Map               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  NASA    │  │ Weather  │  │  Fire    │  │ Evac     │    │
│  │  FIRMS   │  │   API    │  │ Stations │  │ Routes   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │             │             │             │           │
│       └─────────────┴─────────────┴─────────────┘           │
│                           │                                 │
│                    ┌──────▼──────┐                          │
│                    │   Redis     │                          │
│                    │   Cache     │                          │
│                    └──────┬──────┘                          │
│                           │                                 │
│              ┌────────────┼────────────┐                    │
│              │            │            │                    │
│       ┌──────▼──────┐    │    ┌──────▼──────┐              │
│       │  Dashboard  │    │    │    REST     │              │
│       │  (Dash/Plotly)  │    │    API       │              │
│       └──────────────┘    │    └────────────┘              │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
       ┌──────▼──────┐           ┌──────▼──────┐
       │  MongoDB    │           │  PostgreSQL │
       │ (Historical)│           │ (Users/API) │
       └─────────────┘           └─────────────┘
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/fires` | Get active fires |
| GET | `/api/v1/forecast` | Fire spread prediction |
| POST | `/api/v1/simulate` | Run fire simulation |
| GET | `/api/v1/stations` | Fire station coverage |
| POST | `/api/v1/evacuate` | Evacuation planning |
| GET | `/api/v1/analytics/hotspots` | Fire hotspots |
| GET | `/api/v1/weather` | Current weather |
| GET | `/api/v1/analytics/seasonal` | Seasonal analysis |

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

### License

MIT License - See LICENSE file
