# Grand Mobile Support Copilot (Decentrathon 6.0)

Автономный AI-сервис поддержки операторов **Grand Mobile** для триажа жалоб и внутриигровых репортов, извлечения структурированных фактов, поиска подтверждающих пунктов правил (`rules.json`) и подготовки безопасных черновиков ответов.

> **Ключевой принцип:** Сервис **не является автономным модератором**. Окончательные решения о наказаниях, разбане, возврате средств и спорных случаях всегда принимает человек (`needs_human: true`). Каждая ссылка на правило строго верифицируется по базе `rules.json`.

---

## 1. Соответствие официальному заданию хакатона

В проекте подключены и используются официальные файлы трека Grand Mobile Support Copilot:
- [`rules.json`](rules.json) — 11 разделов правил проекта Grand Mobile (198 пунктов и RP-терминов: «Общие правила игры» `1.1`–`5.9`, «Правила подачи жалоб» `ЖБ 1.1.1`–`ЖБ 1.3.1`, «Правила нахождения в Зелёной Зоне» `GZ 1.1`–`GZ 2.3`, «Правила государственных организаций», «RP термины и их описание» и др.);
- [`taxonomy.json`](taxonomy.json) — 14 официальных категорий (`complaint.insult`, `complaint.cheat`, `complaint.dm`, `complaint.scam`, `complaint.rmt_ads`, `complaint.admin`, `complaint.other`, `complaint.invalid`, `report.question`, `report.help`, `report.complaint`, `report.account`, `report.payment`, `report.offtopic`), 4 уровня приоритета (`p0`, `p1`, `p2`, `p3`) и список обязательной эскалации `needs_human_always`;
- **Защита данных датасетов (`.gitignore`)**: реальные файлы `tickets_train.jsonl.gz`, `tickets_valid.jsonl` и файлы предсказаний `pred*.jsonl` исключены из Git.

---

## 2. Быстрый старт и команды запуска

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск веб-сервера и панели оператора
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
# или
make run
# на Windows:
.\make.bat run
```
- **Панель оператора (тёмная тема, RU)**: http://127.0.0.1:8000/
- **Проверка состояния (`GET /health`)**: http://127.0.0.1:8000/health
- **Swagger UI (`POST /triage`)**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Пакетная обработка JSONL (`make predict`)
На Linux / macOS:
```bash
make predict IN=examples/sample_tickets.jsonl OUT=pred.jsonl
```
На Windows (PowerShell / CMD):
```powershell
.\make.bat predict IN=examples/sample_tickets.jsonl OUT=pred.jsonl
```
Или напрямую через Python CLI:
```bash
python -m app.cli predict --in examples/sample_tickets.jsonl --out pred.jsonl
```

### Обучение классификатора и оценка метрик
```bash
# Обучение на tickets_train.jsonl.gz
python -m app.cli train

# Оценка Accuracy (категории), Macro-F1 (приоритеты), p95 задержки и стоимости
python -m app.cli evaluate --out pred_valid.jsonl
```

### Запуск автотестов
```bash
python -m unittest discover -s tests -p "test_*.py" -v
# или
make test
# на Windows:
.\make.bat test
```

---

## 3. Примеры входа и выхода (`POST /triage`)

### Пример входа
```json
{
  "kind": "complaint",
  "messages": [
    {
      "role": "player",
      "text": "Игрок Denis_Voronov оскорблял моих родных в голосовом чате на 2 сервере, есть скриншот /time",
      "attachments": 1
    }
  ]
}
```

### Пример ответа
```json
{
  "ticket_id": "tkt_507951",
  "language": "ru",
  "category": "complaint.insult",
  "category_name": "Оскорбление игроков / родных",
  "priority": "p1",
  "is_spam": false,
  "is_duplicate": false,
  "facts": {
    "nicknames": ["Denis_Voronov"],
    "offender_nickname": "Denis_Voronov",
    "reporter_nickname": null,
    "mentioned_rules": [],
    "valid_mentioned_rules": [],
    "unknown_mentioned_rules": [],
    "has_evidence": true,
    "attachments_count": 1,
    "evidence_types": ["attachments", "screenshot_mention"],
    "server": 2,
    "amounts": [],
    "commands_mentioned": ["/time"],
    "is_mixed_language": false,
    "detected_languages": ["ru"],
    "extra_fields": {}
  },
  "draft_response": "Здравствуйте! Ваша жалоба на игрока Denis_Voronov принята к рассмотрению. Упоминание или оскорбление родственников регламентируется п. 2.6 Общих правил игры (санкция: Ban 7–30 дней). Администрация проверит предоставленные доказательства и логи сервера. Итоговый вердикт будет вынесен в этой теме.",
  "rule_ids": ["2.1", "2.4", "2.6"],
  "cited_rules": [
    {
      "id": "2.6",
      "section": "2. Правила общения (текстовый и голосовой чат)",
      "title": "Упоминание / оскорбление родственников",
      "quote": "Запрещено оскорбление, неуважительное отношение или упоминание родственников в любой форме (OOC/IC/голосовой чат).",
      "punishment": "Блокировка аккаунта (Ban от 7 до 30 дней)",
      "relevance_score": 0.92
    }
  ],
  "needs_human": true,
  "needs_human_reason": "Жалоба на игрока требует проверки доказательств и логов модератором",
  "confidence": 0.94,
  "reasons": [
    "Определен язык обращения: 'ru'",
    "Классификатор определил категорию 'complaint.insult' с уверенностью 0.94",
    "Приоритет 'p1' назначен согласно критериям taxonomy.json для оскорблений родных",
    "Найдены подтверждающие пункты rules.json: 2.1, 2.4, 2.6",
    "Требуется оператор: Жалоба на игрока требует проверки доказательств и логов модератором"
  ],
  "processing_time_ms": 4.5,
  "cost_usd": 0.0
}
```

---

## 4. Структура проекта

- `app/main.py` — FastAPI-сервис (`POST /triage`, `GET /health`, статика панели оператора).
- `app/triage.py` — Оркестратор триажа, логика `needs_human` и сбор метрик.
- `app/rag.py` — Индексация 11 разделов `rules.json`, поиск (BM25 + n-grams + RP-словарь) и валидация `rule_ids`.
- `app/classifier.py` — Загрузчик `taxonomy.json` и ML-классификатор категорий/приоритетов (`models/classifier_weights.json`).
- `app/extractor.py` — Детекция языка (`ru`, `kk`, `uz`, `en`, `und`), сущностей (ники, сервер, /time, улики), спама и дубликатов.
- `app/generator.py` & `app/safety.py` — Генератор черновиков на языке игрока + защитный фильтр `SafetyGuard`.
- `app/workspace.py` — Роли Junior Admin и Senior Admin, жизненный цикл тикетов, передача и история действий.
- `app/static/index.html` — Веб-панель оператора поддержки с переключателем Demo Mode, уведомлениями и фильтрами.
- `tests/test_triage.py` — 17 автоматических интеграционных тестов по всем сценариям.
- `REPORT.md` — Отчёт с измеренными метриками, латентностью, стоимостью и примерами черновиков.
