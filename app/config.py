"""Configuration loader for Grand Mobile Support Copilot."""
import os
from pathlib import Path
from typing import Dict


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    try:
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception:
        pass


ROOT_DIR = Path(__file__).resolve().parent.parent
_load_dotenv(ROOT_DIR / ".env")


class Settings:
    def __init__(self) -> None:
        self.root_dir: Path = ROOT_DIR
        self.rules_path: Path = self._resolve_path(
            os.getenv("RULES_PATH", "rules.json"),
            fallback="rules.json",
        )
        self.taxonomy_path: Path = self._resolve_path(
            os.getenv("TAXONOMY_PATH", "taxonomy.json"),
            fallback="taxonomy.json",
        )
        self.train_data_path: Path = self._resolve_path(
            os.getenv("TRAIN_DATA_PATH", "tickets_train.jsonl.gz"),
            fallback=os.getenv("SYNTHETIC_TRAIN_PATH", "data/synthetic_tickets_train.jsonl.gz"),
        )
        self.valid_data_path: Path = self._resolve_path(
            os.getenv("VALID_DATA_PATH", "tickets_valid.jsonl"),
            fallback=os.getenv("SYNTHETIC_VALID_PATH", "data/synthetic_tickets_valid.jsonl"),
        )
        self.model_weights_path: Path = ROOT_DIR / os.getenv(
            "MODEL_WEIGHTS_PATH", "models/classifier_weights.json"
        )

        # LLM Configuration
        self.llm_enabled: bool = os.getenv("LLM_ENABLED", "false").lower() in ("1", "true", "yes")
        self.llm_api_key: str = os.getenv("LLM_API_KEY", "").strip()
        self.llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "3.5"))
        self.llm_input_cost_per_1k: float = float(os.getenv("LLM_INPUT_COST_PER_1K", "0.00015"))
        self.llm_output_cost_per_1k: float = float(os.getenv("LLM_OUTPUT_COST_PER_1K", "0.00060"))

        # Telegram Admin Notifications Configuration
        self.telegram_enabled: bool = os.getenv("TELEGRAM_ENABLED", "true").lower() in ("1", "true", "yes")
        self.telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.telegram_admin_ids: Dict[str, str] = self._parse_admin_ids(os.getenv("TELEGRAM_ADMIN_IDS", ""))
        self.telegram_reminder_minutes: int = int(os.getenv("TELEGRAM_REMINDER_MINUTES", "15"))
        self.dashboard_base_url: str = os.getenv("DASHBOARD_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

        # Grand Mobile Game Server API Configuration
        self.grand_mobile_api_enabled: bool = os.getenv("GRAND_MOBILE_API_ENABLED", "false").lower() in ("1", "true", "yes")
        self.grand_mobile_api_url: str = os.getenv("GRAND_MOBILE_API_URL", "").strip().rstrip("/")
        self.grand_mobile_api_key: str = os.getenv("GRAND_MOBILE_API_KEY", "").strip()
        self.grand_mobile_api_timeout: float = float(os.getenv("GRAND_MOBILE_API_TIMEOUT", "3.0"))

        # Evidence & OCR Configuration
        self.max_evidence_size_mb: int = int(os.getenv("MAX_EVIDENCE_SIZE_MB", "25"))
        self.ocr_enabled: bool = os.getenv("OCR_ENABLED", "true").lower() in ("1", "true", "yes")
        self.tesseract_cmd: str = os.getenv("TESSERACT_CMD", "").strip()

        self.host: str = os.getenv("HOST", "127.0.0.1")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @staticmethod
    def _parse_admin_ids(raw: str) -> Dict[str, str]:
        result: Dict[str, str] = {}
        if not raw or not raw.strip():
            return result
        for pair in raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                k, v = pair.split(":", 1)
                if k.strip() and v.strip():
                    result[k.strip()] = v.strip()
        return result

    def _resolve_path(self, primary: str, fallback: str) -> Path:
        p = Path(primary)
        if not p.is_absolute():
            p = self.root_dir / p
        if p.exists():
            return p
        alt = self.root_dir / "data" / Path(primary).name
        if alt.exists():
            return alt
        fb = Path(fallback)
        if not fb.is_absolute():
            fb = self.root_dir / fb
        return fb

    @property
    def is_using_synthetic_train(self) -> bool:
        return "synthetic" in self.train_data_path.name.lower()

    @property
    def is_using_synthetic_valid(self) -> bool:
        return "synthetic" in self.valid_data_path.name.lower()


settings = Settings()
