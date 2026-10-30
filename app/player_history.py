"""
Player violation history (Repeat Offenders) module for Grand Mobile Support Copilot.

IMPORTANT: 
- Unconfirmed complaints are NOT treated as proven violations.
- History cards clearly distinguish: confirmed violations vs. unconfirmed complaints.
- If real game DB is unavailable, DEMO data is clearly marked.
"""
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("grand_mobile_copilot.player_history")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Demo player history data — clearly marked as DEMO, not real Grand Mobile data
_DEMO_HISTORY: Dict[str, Dict[str, Any]] = {
    "Player_777": {
        "nickname": "Player_777",
        "is_demo": True,
        "demo_notice": "DEMO DATA — не является реальными данными Grand Mobile",
        "total_complaints": 8,
        "confirmed_violations": 3,
        "rejected_complaints": 2,
        "pending_complaints": 3,
        "punishment_history": [
            {"date": "2024-10-15", "type": "Mute 60 min", "rule": "2.1", "admin": "Ерлан Б.", "confirmed": True},
            {"date": "2024-10-22", "type": "Ban 1 day", "rule": "3.2", "admin": "Мария С.", "confirmed": True},
            {"date": "2024-11-01", "type": "Warn", "rule": "2.2", "admin": "Ерлан Б.", "confirmed": True},
        ],
        "repeat_category": "insult_toxicity",
        "last_complaint_date": "2024-11-10",
        "risk_level": "high",
    },
    "Player_228": {
        "nickname": "Player_228",
        "is_demo": True,
        "demo_notice": "DEMO DATA — не является реальными данными Grand Mobile",
        "total_complaints": 2,
        "confirmed_violations": 0,
        "rejected_complaints": 1,
        "pending_complaints": 1,
        "punishment_history": [],
        "repeat_category": None,
        "last_complaint_date": "2024-11-05",
        "risk_level": "low",
    },
    "ScammerPro": {
        "nickname": "ScammerPro",
        "is_demo": True,
        "demo_notice": "DEMO DATA — не является реальными данными Grand Mobile",
        "total_complaints": 12,
        "confirmed_violations": 5,
        "rejected_complaints": 3,
        "pending_complaints": 4,
        "punishment_history": [
            {"date": "2024-09-10", "type": "Ban 3 days", "rule": "4.1", "admin": "Ерлан Б.", "confirmed": True},
            {"date": "2024-09-25", "type": "Ban 7 days", "rule": "4.1", "admin": "Ерлан Б.", "confirmed": True},
            {"date": "2024-10-15", "type": "Ban 30 days", "rule": "4.2", "admin": "Ерлан Б.", "confirmed": True},
            {"date": "2024-11-01", "type": "Ban 30 days", "rule": "4.1", "admin": "Мария С.", "confirmed": True},
            {"date": "2024-11-08", "type": "Permanent Ban WARNING", "rule": "4.3", "admin": "Ерлан Б.", "confirmed": True},
        ],
        "repeat_category": "scam_fraud",
        "last_complaint_date": "2024-11-12",
        "risk_level": "critical",
    },
}


class PlayerHistoryStore:
    """
    In-memory player history storage.
    Tracks complaints against players and confirmed violations.
    Distinguishes confirmed violations from unconfirmed complaints.
    """

    def __init__(self) -> None:
        # Runtime records for newly filed complaints in this session
        self._reported_players: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def record_complaint_against(
        self,
        offender_nickname: str,
        ticket_id: str,
        category: str,
        confirmed: bool = False,
    ) -> None:
        """Record a complaint against a player (not confirmed violation)."""
        nick_lower = offender_nickname.lower()
        with self._lock:
            if nick_lower not in self._reported_players:
                self._reported_players[nick_lower] = {
                    "nickname": offender_nickname,
                    "is_demo": False,
                    "complaints": [],
                    "confirmed_violations": 0,
                    "total_complaints": 0,
                }
            
            player_data = self._reported_players[nick_lower]
            player_data["complaints"].append({
                "ticket_id": ticket_id,
                "category": category,
                "confirmed": confirmed,
                "date": _now(),
            })
            player_data["total_complaints"] = len(player_data["complaints"])
            if confirmed:
                player_data["confirmed_violations"] += 1

    def get_player_history(
        self,
        nickname: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Get player violation history by nickname.
        Falls back to demo data if nickname matches a demo player.
        Clearly marks demo data as DEMO.
        """
        if not nickname:
            return None

        nick_stripped = nickname.strip()
        nick_lower = nick_stripped.lower()

        # Check runtime records first
        with self._lock:
            runtime_data = self._reported_players.get(nick_lower)

        if runtime_data:
            return self._format_runtime_history(runtime_data)

        # Check demo data (case-insensitive)
        for demo_nick, demo_data in _DEMO_HISTORY.items():
            if demo_nick.lower() == nick_lower:
                return self._format_demo_history(demo_data)

        # No history found
        return {
            "nickname": nick_stripped,
            "found": False,
            "is_demo": False,
            "message": f"История по игроку {nick_stripped!r} не найдена в базе данных",
            "disclaimer": "Если игровой API не подключён, история может быть неполной",
        }

    def _format_runtime_history(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format runtime (in-session) player history."""
        complaints = data.get("complaints", [])
        confirmed = [c for c in complaints if c.get("confirmed")]
        unconfirmed = [c for c in complaints if not c.get("confirmed")]
        
        # Category frequency
        cat_counts: Dict[str, int] = {}
        for c in complaints:
            cat_counts[c.get("category", "unknown")] = cat_counts.get(c.get("category", "unknown"), 0) + 1
        most_common = max(cat_counts, key=cat_counts.get) if cat_counts else None

        return {
            "nickname": data["nickname"],
            "found": True,
            "is_demo": False,
            "total_complaints": data["total_complaints"],
            "confirmed_violations": data["confirmed_violations"],
            "unconfirmed_complaints": len(unconfirmed),
            "repeat_category": most_common,
            "risk_level": self._calculate_risk_level(data["total_complaints"], data["confirmed_violations"]),
            "recent_complaints": complaints[-5:],
            "punishment_history": [],
            "disclaimer": "История формируется в рамках текущей сессии. Для полных данных подключите игровой API.",
        }

    def _format_demo_history(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format demo history with clear DEMO marking."""
        return {
            "nickname": data["nickname"],
            "found": True,
            "is_demo": True,
            "demo_notice": data.get("demo_notice", "DEMO DATA"),
            "total_complaints": data["total_complaints"],
            "confirmed_violations": data["confirmed_violations"],
            "rejected_complaints": data.get("rejected_complaints", 0),
            "pending_complaints": data.get("pending_complaints", 0),
            "unconfirmed_complaints": data.get("pending_complaints", 0),
            "repeat_category": data.get("repeat_category"),
            "risk_level": data.get("risk_level", "low"),
            "punishment_history": data.get("punishment_history", []),
            "last_complaint_date": data.get("last_complaint_date"),
            "disclaimer": "DEMO DATA: Показаны тестовые данные, не являющиеся реальной историей Grand Mobile",
        }

    def _calculate_risk_level(self, total: int, confirmed: int) -> str:
        """Calculate player risk level based on complaint history."""
        if confirmed >= 5 or total >= 10:
            return "critical"
        elif confirmed >= 3 or total >= 6:
            return "high"
        elif confirmed >= 1 or total >= 3:
            return "medium"
        else:
            return "low"

    def get_repeat_offenders(self, min_complaints: int = 3, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of repeat offenders (for admin dashboard)."""
        results = []
        
        # Runtime offenders
        with self._lock:
            runtime_players = list(self._reported_players.values())
        
        for player in runtime_players:
            if player["total_complaints"] >= min_complaints:
                results.append({
                    "nickname": player["nickname"],
                    "total_complaints": player["total_complaints"],
                    "confirmed_violations": player["confirmed_violations"],
                    "is_demo": False,
                })
        
        # Demo offenders
        for nick, data in _DEMO_HISTORY.items():
            if data["total_complaints"] >= min_complaints:
                results.append({
                    "nickname": data["nickname"],
                    "total_complaints": data["total_complaints"],
                    "confirmed_violations": data["confirmed_violations"],
                    "risk_level": data.get("risk_level", "medium"),
                    "is_demo": True,
                })
        
        results.sort(key=lambda x: x["total_complaints"], reverse=True)
        return results[:limit]


# Singleton instance
player_history_store = PlayerHistoryStore()
