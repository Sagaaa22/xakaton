# Grand Mobile Support Copilot (Decentrathon 6.0)

Автономный AI-сервис поддержки операторов **Grand Mobile** для триажа жалоб и внутриигровых репортов, извлечения структурированных фактов, поиска подтверждающих пунктов правил (`rules.json`) и подготовки безопасных черновиков ответов.

> **Ключевой принцип:** Сервис **не является автономным модератором**. Окончательные решения о наказаниях, разбане, возврате средств и спорных случаях всегда принимает человек (`needs_human: true`). Каждая ссылка на правило строго верифицируется по базе `rules.json`.

---

## 1. Результаты обследования проекта и принятые допущения

1. При первичном анализе репозитория файлы `tickets_train.jsonl.gz`, `tickets_valid.jsonl`, `rules.json` и `taxonomy.json` отсутствовали.
2. В соответствии с инструкцией кейса созданы демонстрационные конфигурации и синтетические наборы данных (явно помеченные флагом `"is_synthetic_demo": true`):
   - [`rules.json`](rules.json) — 24 пункта правил сервера Grand Mobile (разделы 1–8: аккаунт, чат, RolePlay, мошенничество, читы/багоюз, донат/возвраты, апелляции, техподдержка);
   - [`taxonomy.json`](taxonomy.json) — 10 категорий и 4 уровня приоритета (`critical`, `high`, `medium`, `low`) с критериями SLA;
   - `data/synthetic_tickets_train.jsonl.gz` и `data/synthetic_tickets_valid.jsonl` — синтетические обучающая и валидационная выборки на `ru`, `kk`, `uz`, `en`.
3. **Защита предоставленных данных (`.gitignore`)**: реальные файлы `tickets_train.jsonl.gz`, `tickets_valid.jsonl` и `data/raw/` добавлены в `.gitignore` и не попадают в публичный Git-репозиторий. Как только реальные файлы размещаются в корне проекта, сервис автоматически загружает их вместо демонстрационных.

---

## 2. Быстрый старт и команды запуска

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск веб-сервера и панели оператора
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
- **Панель оператора (тёмная тема, RU)**: http://127.0.0.1:8000/
- **Проверка состояния (`GET /health`)**: http://127.0.0.1:8000/health
- **Swagger UI (`POST /triage`)**: http://127.0.0.1:8000/docs

### Пакетная обработка JSONL (`make predict`)
На Linux / macOS / Git Bash:
```bash
make predict IN=examples/sample_tickets.jsonl OUT=pred.jsonl
```
На Windows (PowerShell / CMD без установленного GNU Make):
```powershell
.\make.bat predict IN=examples/sample_tickets.jsonl OUT=pred.jsonl
```
Или напрямую через Python CLI:
```bash
python -m app.cli predict --in examples/sample_tickets.jsonl --out pred.jsonl
```

### Обучение классификатора и оценка метрик
```bash
# Обучение на tickets_train.jsonl.gz (или синтетическом демо-наборе, если реального файла нет)
python -m app.cli train

# Оценка Accuracy (категории), Macro-F1 (приоритеты), p95 задержки и стоимости
python -m app.cli evaluate --out examples/sample_outputs.jsonl
```

### Запуск автотестов
```bash
python -m unittest discover -s tests -p "test_*.py" -v
# или
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
      "text": "Игрок Ivan_Grozny оскорблял меня в чате на 5 сервере, скриншоты есть",
      "attachments": 2
    }
  ]
}
```

### Пример ответа
```json
{
  "ticket_id": "tkt_f542936a",
  "language": "ru",
  "category": "insult_toxicity",
  "category_name": "Оскорбления и токсичное поведение",
  "priority": "medium",
  "is_spam": false,
  "is_duplicate": false,
  "facts": {
    "nicknames": ["Ivan_Grozny"],
    "offender_nickname": "Ivan_Grozny",
    "reporter_nickname": null,
    "mentioned_rules": [],
    "valid_mentioned_rules": [],
    "unknown_mentioned_rules": [],
    "has_evidence": true,
    "attachments_count": 2,
    "evidence_types": ["attachments", "screenshot_mention"],
    "server": 5,
    "amounts": [],
    "commands_mentioned": [],
    "is_mixed_language": false,
    "detected_languages": ["ru"],
    "extra_fields": {}
  },
  "draft_response": "Здравствуйте! Благодарим за обращение (сервер #5). Приложенные вами материалы по заявленной ситуации в отношении игрока Ivan_Grozny приняты. Потенциально применимый регламент — п. 2.1, п. 2.2 (Оскорбления игроков и неадекватное поведение (OOC/IC)): «Запрещены любые оскорбления игроков, унижение чести и достоинства, грубое неадекватное поведение в OOC-чатах (/n, семейный, вип-чат), а также злоупотребление оскорблениями в голосовом и IC-чате без веской RP-причины. Для рассмотрения жалобы необходимы доказательства (скриншот или видеозапись с видимым временем /time и ID/никнеймом нарушителя).». Мы не выносим автоматических решений и не гарантируем применение санкций до завершения расследования: обращение передано модератору для проверки доказательств и логов, после чего администратор примет окончательное решение.",
  "rule_ids": ["2.1", "2.2"],
  "cited_rules": [
    {
      "id": "2.1",
      "section": "2. Правила общения (текстовый и голосовой чат)",
      "title": "Оскорбления игроков и неадекватное поведение (OOC/IC)",
      "quote": "Запрещены любые оскорбления игроков, унижение чести и достоинства, грубое неадекватное поведение в OOC-чатах (/n, семейный, вип-чат), а также злоупотребление оскорблениями в голосовом и IC-чате без веской RP-причины. Для рассмотрения жалобы необходимы доказательства (скриншот или видеозапись с видимым временем /time и ID/никнеймом нарушителя).",
      "punishment": "Блокировка чата (Mute от 30 до 120 минут) / Ban до 3 дней при рецидиве",
      "relevance_score": 0.613
    }
  ],
  "needs_human": true,
  "needs_human_reason": "Окончательное решение о выдаче наказания игроку по приложенным доказательствам принимает модератор",
  "confidence": 0.98,
  "reasons": [
    "Определен язык обращения: 'ru'",
    "Классификатор определил категорию 'insult_toxicity' (Оскорбления и токсичное поведение) с уверенностью 0.98",
    "Приоритет 'medium' назначен согласно критериям taxonomy.json для 'insult_toxicity'",
    "Найдены подтверждающие пункты rules.json: 2.1, 2.2",
    "Требуется оператор: Окончательное решение о выдаче наказания игроку по приложенным доказательствам принимает модератор"
  ],
  "processing_time_ms": 1.62,
  "cost_usd": 0.0
}
```

---

## 4. Структура проекта

- `app/main.py` — FastAPI-приложение (`POST /triage`, `GET /health`, `GET /`, `GET /api/tickets`, `GET /api/analytics`, `POST /api/escalate`).
- `app/triage.py` — Оркестратор триажа, логика `needs_human` и безопасный сбор обезличенных метрик.
- `app/rag.py` — Индексация фрагментов `rules.json`, гибридный поиск (BM25 + char n-grams + мультиязычное расширение) и жёсткая верификация `rule_ids`.
- `app/classifier.py` — Загрузчик `taxonomy.json` и классификатор категорий/приоритетов, обучаемый на `tickets_train.jsonl.gz` (включая извлечение меток из ответов администраторов).
- `app/extractor.py` — Определение языка (`ru`, `kk`, `uz`, `en`, `und`), смешения языков, спама/дубликатов и извлечение фактов (никнеймы, сервер, вложения, правила).
- `app/generator.py` & `app/safety.py` — Генератор черновиков на языке игрока (с опциональным подключением LLM и автоматическим fallback) + защитный фильтр `SafetyGuard`.
- `app/static/index.html` — Тёмная панель оператора поддержки на русском языке со всеми 9 разделами.
- `tests/test_triage.py` — Полный набор автотестов по всем 9 обязательным сценариям.
- `REPORT.md` — Отчёт с метриками, стоимостью и задержкой.

