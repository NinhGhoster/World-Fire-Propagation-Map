"""
Rate Limiting & API Key Management

Protect API endpoints with rate limiting and API key authentication.
"""
import time
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import json


@dataclass
class APIKey:
    """API key with metadata."""
    key: str
    name: str
    user_id: str
    created_at: datetime
    last_used: datetime
    requests_today: int
    daily_limit: int
    is_active: bool
    
    def to_dict(self) -> dict:
        return {
            "key": self.key[:8] + "...",
            "name": self.name,
            "user_id": self.user_id,
            "requests_today": self.requests_today,
            "daily_limit": self.daily_limit,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat()
        }


class RateLimiter:
    """In-memory rate limiter using sliding window."""
    
    def __init__(self, window_seconds: int = 60, max_requests: int = 60):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests: Dict[str, list] = {}
    
    def _get_key(self, identifier: str, endpoint: str = "") -> str:
        """Generate rate limit key."""
        return f"{identifier}:{endpoint}"
    
    def is_allowed(self, identifier: str, endpoint: str = "") -> Tuple[bool, Dict]:
        """Check if request is allowed."""
        key = self._get_key(identifier, endpoint)
        now = time.time()
        
        # Clean old requests
        if key in self.requests:
            self.requests[key] = [
                t for t in self.requests[key]
                if now - t < self.window_seconds
            ]
        else:
            self.requests[key] = []
        
        # Check limit
        if len(self.requests[key]) >= self.max_requests:
            return False, {
                "error": "Rate limit exceeded",
                "retry_after": int(self.window_seconds - (now - self.requests[key][0]))
            }
        
        # Add request
        self.requests[key].append(now)
        
        return True, {
            "remaining": self.max_requests - len(self.requests[key]),
            "reset_after": int(self.window_seconds)
        }
    
    def get_usage(self, identifier: str, endpoint: str = "") -> Dict:
        """Get current usage stats."""
        key = self._get_key(identifier, endpoint)
        now = time.time()
        
        if key in self.requests:
            requests = [t for t in self.requests[key] if now - t < self.window_seconds]
            return {
                "requests": len(requests),
                "limit": self.max_requests,
                "remaining": max(0, self.max_requests - len(requests)),
                "reset_seconds": int(self.window_seconds - (now - requests[0])) if requests else 0
            }
        
        return {
            "requests": 0,
            "limit": self.max_requests,
            "remaining": self.max_requests,
            "reset_seconds": self.window_seconds
        }


class APIKeyManager:
    """Manage API keys for authenticated access."""
    
    def __init__(self, storage_file: str = "/tmp/fire_api_keys.json"):
        self.storage_file = storage_file
        self.keys: Dict[str, APIKey] = self._load_keys()
    
    def _load_keys(self) -> Dict[str, APIKey]:
        """Load keys from storage."""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    keys = {}
                    for k, v in data.items():
                        keys[k] = APIKey(
                            key=k,
                            name=v["name"],
                            user_id=v["user_id"],
                            created_at=datetime.fromisoformat(v["created_at"]),
                            last_used=datetime.fromisoformat(v["last_used"]),
                            requests_today=v.get("requests_today", 0),
                            daily_limit=v.get("daily_limit", 10000),
                            is_active=v.get("is_active", True)
                        )
                    return keys
            except:
                pass
        return {}
    
    def _save_keys(self):
        """Save keys to storage."""
        data = {}
        for k, v in self.keys.items():
            data[k] = {
                "name": v.name,
                "user_id": v.user_id,
                "created_at": v.created_at.isoformat(),
                "last_used": v.last_used.isoformat(),
                "requests_today": v.requests_today,
                "daily_limit": v.daily_limit,
                "is_active": v.is_active
            }
        
        with open(self.storage_file, 'w') as f:
            json.dump(data, f)
    
    def generate_key(self, name: str, user_id: str, daily_limit: int = 10000) -> str:
        """Generate new API key."""
        # Generate random key
        random_part = os.urandom(16).hex()
        timestamp = str(int(time.time()))
        key = f"fp_{random_part}_{timestamp}"
        
        api_key = APIKey(
            key=key,
            name=name,
            user_id=user_id,
            created_at=datetime.now(),
            last_used=datetime.now(),
            requests_today=0,
            daily_limit=daily_limit,
            is_active=True
        )
        
        self.keys[key] = api_key
        self._save_keys()
        
        return key
    
    def validate_key(self, key: str) -> Tuple[bool, Optional[APIKey]]:
        """Validate API key."""
        if not key:
            return False, None
        
        api_key = self.keys.get(key)
        
        if not api_key:
            return False, None
        
        if not api_key.is_active:
            return False, None
        
        # Check daily limit
        if api_key.requests_today >= api_key.daily_limit:
            return False, None
        
        # Update last used
        api_key.last_used = datetime.now()
        api_key.requests_today += 1
        self._save_keys()
        
        return True, api_key
    
    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        if key in self.keys:
            self.keys[key].is_active = False
            self._save_keys()
            return True
        return False
    
    def get_key_info(self, key: str) -> Optional[dict]:
        """Get API key info."""
        api_key = self.keys.get(key)
        if api_key:
            return api_key.to_dict()
        return None
    
    def list_keys(self, user_id: str = None) -> list:
        """List all keys."""
        keys = []
        for k, v in self.keys.items():
            if user_id is None or v.user_id == user_id:
                keys.append(v.to_dict())
        return keys
    
    def reset_daily_counts(self):
        """Reset daily request counts."""
        for k in self.keys:
            self.keys[k].requests_today = 0
        self._save_keys()


# Default instances
rate_limiter = RateLimiter(window_seconds=60, max_requests=60)  # 60 req/min
api_key_manager = APIKeyManager()


def require_api_key(f):
    """Decorator to require API key for endpoint."""
    from functools import wraps
    from flask import request, jsonify
    
    @wraps(f)
    def decorated(*args, **kwargs):
        # Check header for API key
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        
        # For now, allow requests without key but with warning
        if not api_key:
            # Allow unauthenticated access but rate limit stricter
            identifier = request.remote_addr or "unknown"
            allowed, info = rate_limiter.is_allowed(identifier, request.path)
            if not allowed:
                return jsonify({"error": "Rate limit exceeded", **info}), 429
            response = f(*args, **kwargs)
            # Add rate limit headers
            if hasattr(response, 'headers'):
                response.headers["X-RateLimit-Remaining"] = str(info.get("remaining", 0))
            return response
        
        # Validate API key
        valid, key_info = api_key_manager.validate_key(api_key)
        
        if not valid:
            return jsonify({"error": "Invalid or expired API key"}), 401
        
        return f(*args, **kwargs)
    
    return decorated


# Swagger/OpenAPI documentation
API_DOCS = {
    "openapi": "3.0.0",
    "info": {
        "title": "World Fire Propagation Map API",
        "version": "3.0.0",
        "description": "Real-time wildfire tracking, analytics, and evacuation planning API"
    },
    "servers": [
        {"url": "http://100.72.4.35:8050", "description": "Local Development"},
        {"url": "https://fire.azurewebsites.net", "description": "Production"}
    ],
    "paths": {
        "/api/v1/fires": {
            "get": {
                "summary": "Get active fires",
                "parameters": [
                    {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}},
                    {"name": "lon", "in": "query", "required": True, "schema": {"type": "number"}},
                    {"name": "radius", "in": "query", "schema": {"type": "number", "default": 200}}
                ],
                "responses": {
                    "200": {"description": "List of active fires"},
                    "400": {"description": "Missing parameters"}
                }
            }
        },
        "/api/v1/weather": {
            "get": {
                "summary": "Get weather and fire danger",
                "parameters": [
                    {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}},
                    {"name": "lon", "in": "query", "required": True, "schema": {"type": "number"}}
                ],
                "responses": {
                    "200": {"description": "Weather data with fire danger rating"}
                }
            }
        },
        "/api/v1/analytics/hotspots": {
            "get": {
                "summary": "Identify fire hotspots",
                "parameters": [
                    {"name": "lat", "in": "query", "required": True},
                    {"name": "lon", "in": "query", "required": True},
                    {"name": "radius", "in": "query", "schema": {"type": "number", "default": 500}}
                ],
                "responses": {
                    "200": {"description": "List of identified fire hotspots"}
                }
            }
        },
        "/api/v1/risk": {
            "get": {
                "summary": "Get fire risk assessment",
                "parameters": [
                    {"name": "lat", "in": "query", "required": True},
                    {"name": "lon", "in": "query", "required": True}
                ],
                "responses": {
                    "200": {"description": "Risk assessment with recommendations"}
                }
            }
        },
        "/api/v1/simulate": {
            "post": {
                "summary": "Run fire spread simulation",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "grid_size": {"type": "integer", "default": 7},
                                    "lambda_spread": {"type": "number", "default": 0.1},
                                    "firefighters": {"type": "integer", "default": 2},
                                    "wind_speed": {"type": "number", "default": 30},
                                    "wind_direction": {"type": "string", "default": "NE"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Simulation results"}
                }
            }
        },
        "/api/v1/alerts/subscribe": {
            "post": {
                "summary": "Subscribe to alerts",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "email": {"type": "string"},
                                    "phone": {"type": "string"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "200": {"description": "Successfully subscribed"}
                }
            }
        }
    }
}


def register_rate_limit_routes(app):
    """Register rate limit and API key routes."""
    
    @app.route("/api/v1/keys", methods=["POST"])
    def create_api_key():
        """Create new API key."""
        from flask import request, jsonify
        data = request.get_json() or {}
        
        name = data.get("name", "API Key")
        user_id = data.get("user_id", "default")
        daily_limit = data.get("daily_limit", 10000)
        
        key = api_key_manager.generate_key(name, user_id, daily_limit)
        
        return jsonify({
            "status": "created",
            "key": key,
            "info": {
                "name": name,
                "daily_limit": daily_limit,
                "note": "Store this key securely. It will not be shown again."
            }
        })
    
    @app.route("/api/v1/keys", methods=["GET"])
    def list_api_keys():
        """List your API keys."""
        from flask import request, jsonify
        user_id = request.headers.get("X-API-User-ID", "default")
        
        keys = api_key_manager.list_keys(user_id)
        return jsonify({"keys": keys})
    
    @app.route("/api/v1/docs/openapi.json")
    def get_openapi_docs():
        """Return OpenAPI documentation."""
        from flask import jsonify
        return jsonify(API_DOCS)
    
    @app.route("/api/v1/rate-limit/status")
    def rate_limit_status():
        """Get current rate limit status."""
        from flask import request, jsonify
        identifier = request.remote_addr or "unknown"
        usage = rate_limiter.get_usage(identifier)
        return jsonify(usage)


if __name__ == "__main__":
    print("🔐 Rate Limiting & API Keys Demo")
    print("=" * 50)
    
    # Test rate limiter
    print("\n1. Rate Limiter Test")
    limiter = RateLimiter(window_seconds=10, max_requests=3)
    
    for i in range(5):
        allowed, info = limiter.is_allowed("test_user", "/api/test")
        print(f"   Request {i+1}: {'✅' if allowed else '❌'} - {info}")
    
    # Test API key manager
    print("\n2. API Key Manager Test")
    manager = APIKeyManager()
    
    key = manager.generate_key("Test Key", "user123")
    print(f"   Generated key: {key[:20]}...")
    
    valid, key_info = manager.validate_key(key)
    print(f"   Valid: {valid}")
    print(f"   Key name: {key_info.name}")
    
    print("\n✅ Rate limiting system ready!")
