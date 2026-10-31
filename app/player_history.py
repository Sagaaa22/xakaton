"""
Player violation history (Repeat Offenders) module for Grand Mobile Support Copilot.

Features:
- Lookup & indexing by BOTH Player Nickname (`nickname`) AND In-Game ID (`player_id`).
- Strict separation between Confirmed Violations (`confirmed_violations`) and
  Unconfirmed Complaints (`unconfirmed_complaints`) / Rejected Complaints (`rejected_complaints`).
- Repeat pattern detection (`repeat_violations`, `is_repeat_offender`, `repeat_category`).
- Punishment history (`punishment_history`).
- Pluggable Grand Mobile Game Server API connector (`GrandMobileAPIConnector`):
  - If `GRAND_MOBILE_API_ENABLED=true` and `GRAND_MOBILE_API_URL` is configured in `.env`,
    queries live game server database via HTTP.
  - Otherwise, honestly reports offline/local mode (`game_api_connected: false`) and uses
    session records + clearly marked DEMO profiles.
- Persistent JSON storage in `data/player_history.json`.
"""
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import re
import threading
from typing import Any, Dict, List, Optional, Union

import httpx

from app.config import settings

logger = logging.getLogger("grand_mobile_copilot.player_history")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Demo player history data — clearly marked as DEMO, indexed by both nickname and player_id
_DEMO_HISTORY: Dict[str, Dict[str, Any]] = {
    "Player_777": {
        "nickname": "Player_777",
        "player_id": 777,
        "server": 5,
        "is_demo": True,
        "demo_notice": "DEMO DATA — тестовый профиль (не является реальными данными сервера Grand Mobile)",
        "total_complaints": 8,
        "confirmed_violations": 3,
        "rejected_complaints": 2,
        "pending_complaints": 3,
        "punishment_history": [
            {"date": "2026-10-05", "type": "Mute 60 min", "rule": "2.1", "category": "insult_toxicity", "admin": "Ерлан Б.", "confirmed": True},
            {"date": "2026-10-12", "type": "Jail 90 min (DB)", "rule": "3.2", "category": "rp_violation", "admin": "Мария С.", "confirmed": True},
            {"date": "2026-10-21", "type": "Mute 180 min", "rule": "2.2", "category": "insult_toxicity", "admin": "Ерлан Б.", "confirmed": True},
        ],
        "complaints_breakdown": {
            "confirmed": [
                {"ticket_id": "tkt_h01", "category": "insult_toxicity", "rule": "2.1", "date": "2026-10-05", "status": "confirmed"},
                {"ticket_id": "tkt_h02", "category": "rp_violation", "rule": "3.2", "date": "2026-10-12", "status": "confirmed"},
                {"ticket_id": "tkt_h03", "category": "insult_toxicity", "rule": "2.2", "date": "2026-10-21", "status": "confirmed"},
            ],
            "unconfirmed": [
                {"ticket_id": "tkt_h04", "category": "insult_toxicity", "date": "2026-10-25", "status": "pending_review", "note": "Ожидает проверки скриншота"},
                {"ticket_id": "tkt_h05", "category": "rp_violation", "date": "2026-10-27", "status": "pending_review", "note": "Нет /time на скриншоте"},
                {"ticket_id": "tkt_h06", "category": "insult_toxicity", "date": "2026-10-29", "status": "pending_review", "note": "На рассмотрении модератора"},
            ],
            "rejected": [
                {"ticket_id": "tkt_h07", "category": "cheating_exploits", "date": "2026-09-18", "status": "rejected", "note": "Античит-логи чистые, рассинхрон"},
                {"ticket_id": "tkt_h08", "category": "rp_violation", "date": "2026-10-01", "status": "rejected", "note": "Недостаточно доказательств"},
            ],
        },
        "repeat_category": "insult_toxicity",
        "repeat_violations": ["Систематические оскорбления в чате (п. 2.1, 2.2) — 2 подтверждённых случая"],
        "last_complaint_date": "2026-10-29",
        "risk_level": "high",
    },
    "Player_228": {
        "nickname": "Player_228",
        "player_id": 228,
        "server": 3,
        "is_demo": True,
        "demo_notice": "DEMO DATA — тестовый профиль (не является реальными данными сервера Grand Mobile)",
        "total_complaints": 2,
        "confirmed_violations": 0,
        "rejected_complaints": 1,
        "pending_complaints": 1,
        "punishment_history": [],
        "complaints_breakdown": {
            "confirmed": [],
            "unconfirmed": [
                {"ticket_id": "tkt_p01", "category": "rp_violation", "date": "2026-10-28", "status": "pending_review", "note": "Проверяется"},
            ],
            "rejected": [
                {"ticket_id": "tkt_p02", "category": "insult_toxicity", "date": "2026-10-14", "status": "rejected", "note": "Нарушение не подтверждено"},
            ],
        },
        "repeat_category": None,
        "repeat_violations": [],
        "last_complaint_date": "2026-10-28",
        "risk_level": "low",
    },
    "ScammerPro": {
        "nickname": "ScammerPro",
        "player_id": 1042,
        "server": 1,
        "is_demo": True,
        "demo_notice": "DEMO DATA — тестовый профиль (не является реальными данными сервера Grand Mobile)",
        "total_complaints": 12,
        "confirmed_violations": 5,
        "rejected_complaints": 3,
        "pending_complaints": 4,
        "punishment_history": [
            {"date": "2026-09-10", "type": "Ban 3 days", "rule": "4.1", "category": "scam_fraud", "admin": "Ерлан Б.", "confirmed": True},
            {"date": "2026-09-25", "type": "Ban 7 days", "rule": "4.1", "category": "scam_fraud", "admin": "Ерлан Б.", "confirmed": True},
            {"date": "2026-10-08", "type": "Ban 30 days", "rule": "4.2", "category": "scam_fraud", "admin": "Ерлан Б.", "confirmed": True},
            {"date": "2026-10-19", "type": "Ban 30 days", "rule": "4.1", "category": "scam_fraud", "admin": "Мария С.", "confirmed": True},
            {"date": "2026-10-26", "type": "Permaban Warning", "rule": "4.1", "category": "scam_fraud", "admin": "Ерлан Б.", "confirmed": True},
        ],
        "complaints_breakdown": {
            "confirmed": [
                {"ticket_id": "tkt_s01", "category": "scam_fraud", "rule": "4.1", "date": "2026-09-10", "status": "confirmed"},
                {"ticket_id": "tkt_s02", "category": "scam_fraud", "rule": "4.1", "date": "2026-09-25", "status": "confirmed"},
                {"ticket_id": "tkt_s03", "category": "scam_fraud", "rule": "4.2", "date": "2026-10-08", "status": "confirmed"},
                {"ticket_id": "tkt_s04", "category": "scam_fraud", "rule": "4.1", "date": "2026-10-19", "status": "confirmed"},
                {"ticket_id": "tkt_s05", "category": "scam_fraud", "rule": "4.1", "date": "2026-10-26", "status": "confirmed"},
            ],
            "unconfirmed": [
                {"ticket_id": "tkt_s06", "category": "scam_fraud", "date": "2026-10-28", "status": "pending_review"},
                {"ticket_id": "tkt_s07", "category": "scam_fraud", "date": "2026-10-29", "status": "pending_review"},
                {"ticket_id": "tkt_s08", "category": "scam_fraud", "date": "2026-10-29", "status": "pending_review"},
                {"ticket_id": "tkt_s09", "category": "scam_fraud", "date": "2026-10-30", "status": "pending_review"},
            ],
            "rejected": [
                {"ticket_id": "tkt_s10", "category": "scam_fraud", "date": "2026-09-15", "status": "rejected"},
                {"ticket_id": "tkt_s11", "category": "insult_toxicity", "date": "2026-09-20", "status": "rejected"},
                {"ticket_id": "tkt_s12", "category": "rp_violation", "date": "2026-10-02", "status": "rejected"},
            ],
        },
        "repeat_category": "scam_fraud",
        "repeat_violations": [
            "Рецидив NonRP обмана при сделках (п. 4.1) — 4 подтверждённых нарушения",
            "Нарушение экономических правил (п. 4.2) — 1 подтверждённое нарушение",
        ],
        "last_complaint_date": "2026-10-30",
        "risk_level": "critical",
    },
    "Ivan_Grozny": {
        "nickname": "Ivan_Grozny",
        "player_id": 142,
        "server": 5,
        "is_demo": True,
        "demo_notice": "DEMO DATA — тестовый профиль (не является реальными данными сервера Grand Mobile)",
        "total_complaints": 5,
        "confirmed_violations": 2,
        "rejected_complaints": 1,
        "pending_complaints": 2,
        "punishment_history": [
            {"date": "2026-10-10", "type": "Mute 60 min", "rule": "2.1", "category": "insult_toxicity", "admin": "Айдар К.", "confirmed": True},
            {"date": "2026-10-18", "type": "Mute 120 min", "rule": "2.1", "category": "insult_toxicity", "admin": "Мария С.", "confirmed": True},
        ],
        "complaints_breakdown": {
            "confirmed": [
                {"ticket_id": "tkt_ig1", "category": "insult_toxicity", "rule": "2.1", "date": "2026-10-10", "status": "confirmed"},
                {"ticket_id": "tkt_ig2", "category": "insult_toxicity", "rule": "2.1", "date": "2026-10-18", "status": "confirmed"},
            ],
            "unconfirmed": [
                {"ticket_id": "tkt_ig3", "category": "insult_toxicity", "date": "2026-10-29", "status": "pending_review"},
                {"ticket_id": "tkt_ig4", "category": "rp_violation", "date": "2026-10-30", "status": "pending_review"},
            ],
            "rejected": [
                {"ticket_id": "tkt_ig5", "category": "cheating_exploits", "date": "2026-10-04", "status": "rejected"},
            ],
        },
        "repeat_category": "insult_toxicity",
        "repeat_violations": ["Повторные оскорбления в чате (п. 2.1) — 2 подтверждённых мута"],
        "last_complaint_date": "2026-10-30",
        "risk_level": "high",
    },
}


class GrandMobileAPIConnector:
    """
    Connector for external Grand Mobile Game Server API (punishments, player IDs, logs).
    Honestly reports connection status when external credentials are not provided.
    """

    @property
    def is_connected(self) -> bool:
        return bool(
            settings.grand_mobile_api_enabled
            and settings.grand_mobile_api_url
            and settings.grand_mobile_api_key
        )

    def fetch_player_profile(
        self,
        nickname: Optional[str] = None,
        player_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        if not self.is_connected:
            return None
        try:
            url = f"{settings.grand_mobile_api_url}/v1/players/history"
            params: Dict[str, Any] = {}
            if nickname:
                params["nickname"] = nickname
            if player_id is not None:
                params["player_id"] = player_id
            headers = {"Authorization": f"Bearer {settings.grand_mobile_api_key}"}
            with httpx.Client(timeout=settings.grand_mobile_api_timeout) as client:
                resp = client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("Grand Mobile API request failed (falling back to local store): %s", exc)
            return None

    def get_status_info(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected,
            "api_url": settings.grand_mobile_api_url or None,
            "mode": "live_game_api" if self.is_connected else "local_and_demo_fallback",
            "disclosure": (
                "Подключено к игровому API Grand Mobile."
                if self.is_connected
                else "Прямой доступ к закрытому игровому API Grand Mobile не подключён (GRAND_MOBILE_API_URL не задан в .env). "
                     "Используется локальная база истории текущей сессии и помеченные DEMO-профили. "
                     "Для подключения реальной БД укажите GRAND_MOBILE_API_ENABLED=true, GRAND_MOBILE_API_URL и GRAND_MOBILE_API_KEY в .env."
            ),
        }


class PlayerHistoryStore:
    """
    In-memory + pluggable Game API player history storage with disk persistence.
    Supports lookup by nickname OR in-game player_id.
    Strictly separates confirmed violations from unconfirmed complaints.
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._reported_players: Dict[str, Dict[str, Any]] = {}  # key: lower nickname
        self._id_to_nick: Dict[int, str] = {}                   # player_id -> lower nickname
        self._lock = threading.RLock()
        self.persist_path: Path = persist_path or (settings.root_dir / "data" / "player_history.json")
        self.game_api = GrandMobileAPIConnector()

        # Pre-index demo player IDs
        for nick, data in _DEMO_HISTORY.items():
            pid = data.get("player_id")
            if isinstance(pid, int):
                self._id_to_nick[pid] = nick.lower()

        self._load()

    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            raw = json.loads(self.persist_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                self._reported_players = raw
                for k, v in self._reported_players.items():
                    pid = v.get("player_id")
                    if isinstance(pid, int):
                        self._id_to_nick[pid] = k.lower()
        except Exception as exc:
            logger.warning("Could not load player history: %s", exc)

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.persist_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._reported_players, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self.persist_path)
        except Exception as exc:
            logger.warning("Could not persist player history: %s", exc)

    def reset(self) -> None:
        with self._lock:
            self._reported_players = {}
            self._id_to_nick = {}
            for nick, data in _DEMO_HISTORY.items():
                pid = data.get("player_id")
                if isinstance(pid, int):
                    self._id_to_nick[pid] = nick.lower()
            self._save()

    def record_complaint_against(
        self,
        offender_nickname: Optional[str],
        ticket_id: str,
        category: str,
        offender_id: Optional[int] = None,
        server: Any = None,
        confirmed: bool = False,
        rule_id: Optional[str] = None,
        punishment_type: Optional[str] = None,
        admin_name: Optional[str] = None,
    ) -> None:
        """Record a complaint or confirmed violation against a player by nickname and/or player_id."""
        if not offender_nickname and offender_id is None:
            return

        nick_display = offender_nickname or f"Player_ID_{offender_id}"
        nick_lower = nick_display.strip().lower()

        with self._lock:
            if offender_id is not None:
                if offender_id in self._id_to_nick and not offender_nickname:
                    nick_lower = self._id_to_nick[offender_id]
                else:
                    self._id_to_nick[offender_id] = nick_lower

            if nick_lower not in self._reported_players:
                # Seed from demo history if this player exists in _DEMO_HISTORY
                demo_match = None
                for dn, dd in _DEMO_HISTORY.items():
                    if dn.lower() == nick_lower or (offender_id is not None and dd.get("player_id") == offender_id):
                        demo_match = dd
                        break

                if demo_match:
                    self._reported_players[nick_lower] = {
                        "nickname": demo_match["nickname"],
                        "player_id": demo_match.get("player_id") or offender_id,
                        "server": demo_match.get("server") or server,
                        "is_demo": True,
                        "demo_notice": demo_match.get("demo_notice"),
                        "complaints": (
                            list(demo_match.get("complaints_breakdown", {}).get("confirmed", []))
                            + list(demo_match.get("complaints_breakdown", {}).get("unconfirmed", []))
                            + list(demo_match.get("complaints_breakdown", {}).get("rejected", []))
                        ),
                        "punishment_history": list(demo_match.get("punishment_history", [])),
                        "confirmed_violations": int(demo_match.get("confirmed_violations", 0)),
                        "rejected_complaints": int(demo_match.get("rejected_complaints", 0)),
                        "total_complaints": int(demo_match.get("total_complaints", 0)),
                    }
                else:
                    self._reported_players[nick_lower] = {
                        "nickname": nick_display.strip(),
                        "player_id": offender_id,
                        "server": server,
                        "is_demo": False,
                        "complaints": [],
                        "punishment_history": [],
                        "confirmed_violations": 0,
                        "rejected_complaints": 0,
                        "total_complaints": 0,
                    }

            player_data = self._reported_players[nick_lower]
            if offender_id is not None and not player_data.get("player_id"):
                player_data["player_id"] = offender_id
            if server is not None and not player_data.get("server"):
                player_data["server"] = server

            # Avoid duplicate ticket recording for the same ticket_id unless upgrading to confirmed
            existing = next((c for c in player_data["complaints"] if c.get("ticket_id") == ticket_id), None)
            if existing:
                if confirmed and existing.get("status") != "confirmed":
                    existing["status"] = "confirmed"
                    existing["confirmed"] = True
                    existing["rule"] = rule_id
                    player_data["confirmed_violations"] += 1
                    if punishment_type:
                        player_data["punishment_history"].append({
                            "date": _now()[:10],
                            "type": punishment_type,
                            "rule": rule_id or "—",
                            "category": category,
                            "admin": admin_name or "Администратор",
                            "confirmed": True,
                        })
                    self._save()
                return

            status_str = "confirmed" if confirmed else "pending_review"
            player_data["complaints"].append({
                "ticket_id": ticket_id,
                "category": category,
                "confirmed": confirmed,
                "status": status_str,
                "rule": rule_id,
                "date": _now()[:10],
            })
            player_data["total_complaints"] += 1
            if confirmed:
                player_data["confirmed_violations"] += 1
                if punishment_type:
                    player_data["punishment_history"].append({
                        "date": _now()[:10],
                        "type": punishment_type,
                        "rule": rule_id or "—",
                        "category": category,
                        "admin": admin_name or "Администратор",
                        "confirmed": True,
                    })
            self._save()

    def get_player_history(
        self,
        identifier: Optional[Union[str, int]] = None,
        player_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Look up player history by nickname OR in-game player ID.
        """
        if identifier is None and player_id is None:
            return None

        # Parse identifier if it looks like an integer ID or "ID: 777"
        query_nick: Optional[str] = None
        query_id: Optional[int] = player_id

        if isinstance(identifier, int):
            query_id = identifier
        elif isinstance(identifier, str):
            clean = identifier.strip()
            if not clean:
                return None
            id_m = re.match(r"^(?:id\s*[:#-]?\s*)?(\d{1,6})$", clean, re.IGNORECASE)
            if id_m and query_id is None:
                query_id = int(id_m.group(1))
            else:
                query_nick = clean

        # 1. Try external Grand Mobile Game API if enabled
        live_profile = self.game_api.fetch_player_profile(nickname=query_nick, player_id=query_id)
        if live_profile:
            live_profile["game_api_connected"] = True
            return live_profile

        # 2. Check runtime store (by nick or ID)
        with self._lock:
            target_key = None
            if query_nick and query_nick.lower() in self._reported_players:
                target_key = query_nick.lower()
            elif query_id is not None and query_id in self._id_to_nick:
                mapped_nick = self._id_to_nick[query_id]
                if mapped_nick in self._reported_players:
                    target_key = mapped_nick

            if target_key:
                return self._format_history_card(self._reported_players[target_key])

        # 3. Check static DEMO store (by nick or ID)
        for demo_nick, demo_data in _DEMO_HISTORY.items():
            if (query_nick and demo_nick.lower() == query_nick.lower()) or (
                query_id is not None and demo_data.get("player_id") == query_id
            ):
                return self._format_demo_history(demo_data)

        label = query_nick or f"ID:{query_id}"
        return {
            "nickname": query_nick or f"ID #{query_id}",
            "player_id": query_id,
            "found": False,
            "is_demo": False,
            "game_api_connected": self.game_api.is_connected,
            "total_complaints": 0,
            "confirmed_violations": 0,
            "unconfirmed_complaints": 0,
            "rejected_complaints": 0,
            "is_repeat_offender": False,
            "repeat_violations": [],
            "punishment_history": [],
            "message": f"История нарушений по игроку {label!r} не найдена",
            "disclaimer": self.game_api.get_status_info()["disclosure"],
        }

    def _format_history_card(self, data: Dict[str, Any]) -> Dict[str, Any]:
        complaints = data.get("complaints", [])
        confirmed_list = [c for c in complaints if c.get("confirmed") or c.get("status") == "confirmed"]
        rejected_list = [c for c in complaints if c.get("status") == "rejected"]
        unconfirmed_list = [
            c for c in complaints
            if not c.get("confirmed") and c.get("status") not in ("confirmed", "rejected")
        ]

        confirmed_cnt = max(int(data.get("confirmed_violations", 0)), len(confirmed_list))
        rejected_cnt = max(int(data.get("rejected_complaints", 0)), len(rejected_list))
        total_cnt = max(int(data.get("total_complaints", 0)), len(complaints))
        unconfirmed_cnt = max(0, total_cnt - confirmed_cnt - rejected_cnt)

        # Detect repeat violation patterns across categories
        cat_counts: Dict[str, int] = {}
        confirmed_cat_counts: Dict[str, int] = {}
        for c in complaints:
            cat = c.get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if c.get("confirmed") or c.get("status") == "confirmed":
                confirmed_cat_counts[cat] = confirmed_cat_counts.get(cat, 0) + 1

        most_common = max(cat_counts, key=cat_counts.get) if cat_counts else None
        repeat_patterns: List[str] = []
        for cat, cnt in confirmed_cat_counts.items():
            if cnt >= 2:
                repeat_patterns.append(f"Повторяющееся подтверждённое нарушение в категории '{cat}' ({cnt} раз)")
        for cat, cnt in cat_counts.items():
            if cnt >= 3 and cat not in confirmed_cat_counts:
                repeat_patterns.append(f"Серия неподтверждённых жалоб в категории '{cat}' ({cnt} обращений — требует проверки)")

        risk_level = self._calculate_risk_level(total_cnt, confirmed_cnt)

        return {
            "nickname": data["nickname"],
            "player_id": data.get("player_id"),
            "server": data.get("server"),
            "found": True,
            "is_demo": bool(data.get("is_demo", False)),
            "demo_notice": data.get("demo_notice"),
            "game_api_connected": self.game_api.is_connected,
            "total_complaints": total_cnt,
            "confirmed_violations": confirmed_cnt,
            "unconfirmed_complaints": unconfirmed_cnt,
            "rejected_complaints": rejected_cnt,
            "is_repeat_offender": confirmed_cnt >= 2 or total_cnt >= 4,
            "repeat_category": most_common,
            "repeat_violations": repeat_patterns,
            "risk_level": risk_level,
            "punishment_history": data.get("punishment_history", []),
            "confirmed_list": confirmed_list[-10:],
            "unconfirmed_list": unconfirmed_list[-10:],
            "rejected_list": rejected_list[-5:],
            "recent_complaints": complaints[-8:],
            "disclaimer": (
                "ВАЖНО: Неподтверждённые жалобы не являются доказательством вины. "
                "Учитываются только подтверждённые администрацией нарушения."
            ),
        }

    def _format_demo_history(self, data: Dict[str, Any]) -> Dict[str, Any]:
        breakdown = data.get("complaints_breakdown", {})
        return {
            "nickname": data["nickname"],
            "player_id": data.get("player_id"),
            "server": data.get("server"),
            "found": True,
            "is_demo": True,
            "demo_notice": data.get("demo_notice", "DEMO DATA"),
            "game_api_connected": self.game_api.is_connected,
            "total_complaints": data["total_complaints"],
            "confirmed_violations": data["confirmed_violations"],
            "unconfirmed_complaints": data.get("pending_complaints", 0),
            "rejected_complaints": data.get("rejected_complaints", 0),
            "pending_complaints": data.get("pending_complaints", 0),
            "is_repeat_offender": data["confirmed_violations"] >= 2,
            "repeat_category": data.get("repeat_category"),
            "repeat_violations": data.get("repeat_violations", []),
            "risk_level": data.get("risk_level", "low"),
            "punishment_history": data.get("punishment_history", []),
            "confirmed_list": breakdown.get("confirmed", []),
            "unconfirmed_list": breakdown.get("unconfirmed", []),
            "rejected_list": breakdown.get("rejected", []),
            "last_complaint_date": data.get("last_complaint_date"),
            "disclaimer": (
                "DEMO DATA: Тестовые данные (прямой API Grand Mobile не подключён). "
                "Подтверждённые нарушения и неподтверждённые жалобы строго разделены."
            ),
        }

    def _calculate_risk_level(self, total: int, confirmed: int) -> str:
        if confirmed >= 5 or (confirmed >= 3 and total >= 8):
            return "critical"
        elif confirmed >= 2 or total >= 5:
            return "high"
        elif confirmed >= 1 or total >= 3:
            return "medium"
        else:
            return "low"

    def get_api_status(self) -> Dict[str, Any]:
        """Return external Grand Mobile Game Server API connection status."""
        return self.game_api.get_status_info()

    def get_repeat_offenders(self, min_complaints: int = 2, limit: int = 15) -> List[Dict[str, Any]]:
        """Get list of repeat offenders with separated confirmed vs unconfirmed counts."""
        results_by_nick: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            runtime_keys = list(self._reported_players.keys())

        for k in runtime_keys:
            card = self._format_history_card(self._reported_players[k])
            if card["total_complaints"] >= min_complaints or card["confirmed_violations"] >= 1:
                results_by_nick[card["nickname"].lower()] = card

        for nick, data in _DEMO_HISTORY.items():
            if nick.lower() not in results_by_nick:
                if data["total_complaints"] >= min_complaints or data["confirmed_violations"] >= 1:
                    results_by_nick[nick.lower()] = self._format_demo_history(data)

        items = list(results_by_nick.values())
        items.sort(
            key=lambda x: (x.get("confirmed_violations", 0), x.get("total_complaints", 0)),
            reverse=True,
        )
        return items[:limit]


# Singleton instance
player_history_store = PlayerHistoryStore()
