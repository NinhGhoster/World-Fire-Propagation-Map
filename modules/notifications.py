"""
Notification System

Email and SMS alerts for fire danger and evacuation warnings.
Integrates with Twilio for SMS and SMTP for email.
"""
import os
import smtplib
from email.mime.text import MIMEText
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import json


@dataclass
class Alert:
    """Fire alert notification."""
    alert_type: str  # FIRE_WARNING, EVACUATION, DANGER_UPDATE
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    title: str
    message: str
    location: dict  # lat, lon, radius
    affected_people: int = 0
    created_at: datetime = None
    
    def to_dict(self) -> dict:
        return {
            "type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "location": self.location,
            "affected_people": self.affected_people,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class EmailNotifier:
    """Email notification service."""
    
    def __init__(self, smtp_host: str = None, smtp_port: int = 587,
                 smtp_user: str = None, smtp_password: str = None,
                 from_email: str = None, from_name: str = "Fire Alert System"):
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", 587))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("FROM_EMAIL", "alerts@firemap.local")
        self.from_name = from_name
        
        # For development, log instead of sending
        self.dev_mode = not all([self.smtp_host, self.smtp_user, self.smtp_password])
        if self.dev_mode:
            print("📧 Email notifier running in DEV MODE (no SMTP configured)")
    
    def send_alert(self, alert: Alert, recipients: List[str]) -> bool:
        """Send fire alert email."""
        try:
            # Create message
            msg = MIMEText(alert.message, 'plain', 'utf-8')
            msg['Subject'] = f"🚨 {alert.severity}: {alert.title}"
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = ", ".join(recipients)
            
            if self.dev_mode:
                print(f"\n📧 [DEV] Email Alert:")
                print(f"   To: {recipients}")
                print(f"   Subject: {msg['Subject']}")
                print(f"   Body: {alert.message[:200]}...")
                return True
            
            # Send via SMTP
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            
            print(f"✅ Email sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            print(f"❌ Email send failed: {e}")
            return False
    
    def send_daily_summary(self, summary: dict, recipients: List[str]) -> bool:
        """Send daily fire activity summary."""
        message = f"""
🌍 World Fire Propagation Map - Daily Summary
========================================
Date: {datetime.now().strftime('%Y-%m-%d')}

📊 Statistics:
- Active Fires Worldwide: {summary.get('total_fires', 0)}
- New Fires (24h): {summary.get('new_fires', 0)}
- High Risk Regions: {summary.get('high_risk_regions', 0)}

🔥 Largest Fires:
"""
        for i, fire in enumerate(summary.get('largest_fires', [])[:5], 1):
            message += f"{i}. {fire.get('region', 'Unknown')}: {fire.get('frp', 0)} MW\n"
        
        message += f"""
🌤️ Weather Outlook:
{summary.get('weather_outlook', 'No significant weather events')}

Stay safe!
- Fire Alert System
"""
        
        alert = Alert(
            alert_type="DAILY_SUMMARY",
            severity="LOW",
            title="Daily Fire Activity Summary",
            message=message,
            location={}
        )
        
        return self.send_alert(alert, recipients)


class SMSNotifier:
    """SMS notification via Twilio."""
    
    def __init__(self, account_sid: str = None, auth_token: str = None,
                 from_number: str = None):
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.getenv("TWILIO_FROM_NUMBER")
        
        self.dev_mode = not all([self.account_sid, self.auth_token, self.from_number])
        if self.dev_mode:
            print("📱 SMS notifier running in DEV MODE (no Twilio configured)")
    
    def send_alert(self, alert: Alert, recipients: List[str]) -> dict:
        """Send SMS alert via Twilio."""
        result = {"sent": 0, "failed": 0, "errors": []}
        
        # Truncate message for SMS
        sms_body = f"🚨 {alert.severity}: {alert.title}\n{alert.message[:100]}..."
        
        if self.dev_mode:
            print(f"\n📱 [DEV] SMS Alert:")
            print(f"   To: {recipients}")
            print(f"   Body: {sms_body}")
            result["sent"] = len(recipients)
            return result
        
        try:
            from twilio.rest import Client
            
            client = Client(self.account_sid, self.auth_token)
            
            for recipient in recipients:
                try:
                    message = client.messages.create(
                        body=sms_body,
                        from_=self.from_number,
                        to=recipient
                    )
                    result["sent"] += 1
                    print(f"✅ SMS sent to {recipient}")
                except Exception as e:
                    result["failed"] += 1
                    result["errors"].append({"to": recipient, "error": str(e)})
            
        except ImportError:
            result["errors"].append("twilio not installed")
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    def send_evacuation_warning(self, location: dict, radius_km: float, 
                                 recipients: List[str]) -> dict:
        """Send evacuation SMS warning."""
        alert = Alert(
            alert_type="EVACUATION",
            severity="CRITICAL",
            title="EVACUATION WARNING",
            message=f"Immediate evacuation required within {radius_km}km of your location. Check local emergency services for details.",
            location=location,
            affected_people=len(recipients) * 3  # Estimate
        )
        
        return self.send_alert(alert, recipients)


class AlertManager:
    """Central alert management system."""
    
    def __init__(self):
        self.email_notifier = EmailNotifier()
        self.sms_notifier = SMSNotifier()
        self.alert_history = []
        self.subscribers = self._load_subscribers()
    
    def _load_subscribers(self) -> dict:
        """Load subscriber preferences."""
        subs_file = "/tmp/fire_alerts_subscribers.json"
        if os.path.exists(subs_file):
            with open(subs_file, 'r') as f:
                return json.load(f)
        return {"email": [], "sms": []}
    
    def _save_subscribers(self):
        """Save subscriber preferences."""
        subs_file = "/tmp/fire_alerts_subscribers.json"
        with open(subs_file, 'w') as f:
            json.dump(self.subscribers, f)
    
    def subscribe_email(self, email: str) -> bool:
        """Subscribe email for alerts."""
        if email not in self.subscribers["email"]:
            self.subscribers["email"].append(email)
            self._save_subscribers()
            print(f"✅ {email} subscribed for email alerts")
            return True
        return False
    
    def subscribe_sms(self, phone: str) -> bool:
        """Subscribe phone for SMS alerts."""
        if phone not in self.subscribers["sms"]:
            self.subscribers["sms"].append(phone)
            self._save_subscribers()
            print(f"✅ {phone} subscribed for SMS alerts")
            return True
        return False
    
    def check_and_alert(self, risk_data: dict) -> dict:
        """
        Check risk data and send appropriate alerts.
        
        Args:
            risk_data: Risk assessment data from /api/v1/risk
            
        Returns:
            Alert results
        """
        results = {"email": None, "sms": None}
        
        risk = risk_data.get("overall_risk", "LOW")
        score = risk_data.get("risk_score", 0)
        
        # Determine alert level
        if risk in ["EXTREME"] or score >= 80:
            alert_type = "CRITICAL_FIRE"
            severity = "CRITICAL"
            title = "EXTREME Fire Danger Alert"
            message = f"Extreme fire danger in your area. Risk score: {score}/100. Take immediate precautions."
        elif risk in ["VERY_HIGH"] or score >= 60:
            alert_type = "HIGH_FIRE"
            severity = "HIGH"
            title = "HIGH Fire Danger Alert"
            message = f"Very high fire danger. Risk score: {score}/100. Avoid outdoor activities."
        elif risk in ["HIGH"] or score >= 40:
            alert_type = "FIRE_WARNING"
            severity = "MEDIUM"
            title = "Fire Danger Warning"
            message = f"Elevated fire danger in your area. Score: {score}/100."
        else:
            # No alert needed
            return {"status": "no_alert", "reason": "Low risk"}
        
        # Create alert
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            location=risk_data.get("location", {}),
            created_at=datetime.now()
        )
        
        self.alert_history.append(alert.to_dict())
        
        # Send email alerts
        if self.subscribers["email"]:
            results["email"] = self.email_notifier.send_alert(
                alert, self.subscribers["email"]
            )
        
        # Send SMS alerts for critical/high only
        if severity in ["CRITICAL", "HIGH"] and self.subscribers["sms"]:
            results["sms"] = self.sms_notifier.send_alert(
                alert, self.subscribers["sms"]
            )
        
        return {
            "status": "alert_sent",
            "alert": alert.to_dict(),
            "results": results
        }
    
    def send_evacuation(self, location: dict, radius_km: float, 
                        affected_people: int = 0) -> dict:
        """Send evacuation warning to all subscribers."""
        alert = Alert(
            alert_type="EVACUATION",
            severity="CRITICAL",
            title="EVACUATION ORDER",
            message=f"IMMEDIATE EVACUATION required within {radius_km}km of your location. Follow official evacuation routes.",
            location=location,
            affected_people=affected_people,
            created_at=datetime.now()
        )
        
        results = {"email": None, "sms": None}
        
        if self.subscribers["email"]:
            results["email"] = self.email_notifier.send_alert(
                alert, self.subscribers["email"]
            )
        
        if self.subscribers["sms"]:
            results["sms"] = self.sms_notifier.send_evacuation_warning(
                location, radius_km, self.subscribers["sms"]
            )
        
        return {
            "status": "evacuation_sent",
            "results": results
        }
    
    def get_alert_history(self, limit: int = 50) -> List[dict]:
        """Get recent alerts."""
        return self.alert_history[-limit:]


# API endpoints for alert management
def register_alert_routes(app):
    """Register alert management API routes."""
    
    @app.route("/api/v1/alerts/subscribe", methods=["POST"])
    def subscribe_alert():
        """Subscribe to alerts."""
        from flask import request
        data = request.get_json()
        
        manager = AlertManager()
        
        if data.get("email"):
            manager.subscribe_email(data["email"])
        if data.get("phone"):
            manager.subscribe_sms(data["phone"])
        
        return {"status": "subscribed", "subscribers": len(manager.subscribers["email"]) + len(manager.subscribers["sms"])}
    
    @app.route("/api/v1/alerts/check", methods=["POST"])
    def check_alert():
        """Check risk and send alert if needed."""
        from flask import request
        data = request.get_json()
        
        manager = AlertManager()
        result = manager.check_and_alert(data)
        
        return result
    
    @app.route("/api/v1/alerts/evacuate", methods=["POST"])
    def send_evacuation():
        """Send evacuation warning."""
        from flask import request
        data = request.get_json()
        
        manager = AlertManager()
        result = manager.send_evacuation(
            data.get("location", {}),
            data.get("radius_km", 10),
            data.get("affected_people", 0)
        )
        
        return result
    
    @app.route("/api/v1/alerts/history", methods=["GET"])
    def get_alert_history():
        """Get alert history."""
        manager = AlertManager()
        return {"alerts": manager.get_alert_history()}


if __name__ == "__main__":
    print("🔥 Notification System Demo")
    print("=" * 50)
    
    # Create alert
    alert = Alert(
        alert_type="FIRE_WARNING",
        severity="HIGH",
        title="High Fire Danger",
        message="Very high fire danger in your area. Avoid outdoor activities.",
        location={"lat": -25.0, "lon": 133.0},
        affected_people=5000
    )
    
    print(f"\nAlert: {alert.title}")
    print(f"Type: {alert.alert_type}")
    print(f"Severity: {alert.severity}")
    print(f"Message: {alert.message}")
    
    # Test email (dev mode)
    email = EmailNotifier()
    email.send_alert(alert, ["test@example.com"])
    
    # Test SMS (dev mode)
    sms = SMSNotifier()
    sms.send_alert(alert, ["+1234567890"])
    
    print("\n✅ Notification system ready!")
