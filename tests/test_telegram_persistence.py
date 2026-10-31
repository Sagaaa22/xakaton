"""
Tests for Telegram connectivity endpoints and disk persistence layer.
"""
import json
import unittest
from pathlib import Path
from app.main import app
from app.admin_requests import admin_request_store
from app.fraud_detector import fraud_detector
from app.player_history import player_history_store
from app.telegram_notifier import telegram_notifier
from tests.test_triage import call_asgi


class TestTelegramAndPersistence(unittest.TestCase):
    def test_telegram_test_connection_endpoint(self):
        status, data, _ = call_asgi(app, "POST", "/api/telegram/test-connection")
        self.assertEqual(status, 200)
        self.assertIn("ok", data)
        self.assertIn("message", data)
        self.assertIn("configured", data)

    def test_telegram_test_send_endpoint(self):
        status, data, _ = call_asgi(app, "POST", "/api/telegram/test-send", body={"text": "Тестовый пинг администраторам"})
        self.assertEqual(status, 200)
        self.assertIn("ok", data)
        self.assertIn("message", data)

    def test_telegram_webhook_command_handling(self):
        # Test /queue command
        payload = {
            "message": {
                "message_id": 999,
                "chat": {"id": 12345678},
                "from": {"id": 12345678, "username": "senior_admin"},
                "text": "/queue"
            }
        }
        status, data, _ = call_asgi(app, "POST", "/api/telegram/webhook", body=payload)
        self.assertEqual(status, 200)
        self.assertTrue(data.get("ok"))
        self.assertEqual(data.get("event"), "command")
        self.assertEqual(data.get("command"), "/queue")

    def test_disk_persistence_files_exist_and_valid(self):
        data_dir = Path("data")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Trigger an action on each store to ensure data is saved
        admin_request_store.create(
            player_nickname="Persistent_Test_Player",
            server=5,
            context_messages=[{"sender": "player", "text": "Тестовый вызов для проверки сохранения на диск"}]
        )
        self.assertTrue((data_dir / "admin_requests.json").exists())
        with open(data_dir / "admin_requests.json", "r", encoding="utf-8") as f:
            admin_data = json.load(f)
            self.assertIsInstance(admin_data, dict)
            self.assertTrue(any(r.get("player_nickname") == "Persistent_Test_Player" for r in admin_data.values()))

        self.assertTrue((data_dir / "telegram_outbox.json").exists())
        self.assertTrue((data_dir / "fraud_store.json").exists())
        self.assertTrue((data_dir / "player_history.json").exists())


if __name__ == "__main__":
    unittest.main()
