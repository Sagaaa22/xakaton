"""
Generator for synthetic demonstration datasets (Grand Mobile Support Copilot).
Explicitly marks every generated record with `"is_synthetic_demo": true`.
Do NOT present metrics on this dataset as real production dataset metrics.
"""
import gzip
import json
import random
from pathlib import Path

RANDOM_SEED = 42


TRAIN_TEMPLATES = [
    # insult_toxicity (medium / high when parents insulted with evidence)
    {
        "kind": "complaint",
        "category": "insult_toxicity",
        "priority": "medium",
        "lang": "ru",
        "player_texts": [
            "Игрок {offender} оскорблял меня в чате на {server} сервере, скриншоты есть",
            "На {server} сервере {offender} матерится в вип чате и обзывает всех, приложил скрин с /time",
            "Жалоба на игрока {offender} (ID 412). Неадекват в голосовом чате, орет матом, п. 2.1",
            "Меня крыл матом в /n чате игрок {offender}, сервер {server}, доказательства прикрепил",
            "Оскорбление в семейном чате от {offender} на сервере {server}, 2 скриншота",
        ],
        "admin_reply": "Здравствуйте! Передано модерации для проверки по п. 2.1 правил (Оскорбления игроков). Приоритет: medium.",
        "attachments": [1, 2, 3],
    },
    {
        "kind": "complaint",
        "category": "insult_toxicity",
        "priority": "high",
        "lang": "ru",
        "player_texts": [
            "Игрок {offender} оскорбил мою мать в голосовом чате на {server} сервере! Есть видео с /time, нарушение п. 2.2",
            "Упоминание родных и розжиг от {offender} в общем чате, сервер {server}, скрины приложил!",
            "Жесткое оскорбление родителей от игрока {offender} на {server} сервере, прошу проверить по п. 2.2 и 2.3",
        ],
        "admin_reply": "Здравствуйте! Обращение с высоким приоритетом (high) передано администрации для проверки по п. 2.2 (Упоминание родных).",
        "attachments": [1, 2],
    },
    {
        "kind": "complaint",
        "category": "insult_toxicity",
        "priority": "medium",
        "lang": "kk",
        "player_texts": [
            "Сәлем! {server} серверде {offender} деген ойыншы мені чатта балағаттады, скриншот бар",
            "Ойыншы {offender} дауыстық чатта боқтап жатыр, {server} сервер, дәлел скриншот тіркедім",
        ],
        "admin_reply": "Сәлеметсіз бе! Шағым п. 2.1 бойынша тексеруге жіберілді. Категория: insult_toxicity, приоритет: medium.",
        "attachments": [1, 2],
    },
    {
        "kind": "complaint",
        "category": "insult_toxicity",
        "priority": "medium",
        "lang": "uz",
        "player_texts": [
            "Salom, {server}-serverda {offender} meni chatda haqorat qildi va so'kdi, skrinshot bor",
            "O'yinchi {offender} ovozli chatda haqorat qilyapti, {server} server, skrinshot ilova qildim",
        ],
        "admin_reply": "Assalomu alaykum! Murojaat 2.1-band bo'yicha moderatorga yuborildi (insult_toxicity, medium).",
        "attachments": [1, 2],
    },
    # rp_violation (medium)
    {
        "kind": "complaint",
        "category": "rp_violation",
        "priority": "medium",
        "lang": "ru",
        "player_texts": [
            "Игрок {offender} убил меня без причины (ДМ) около больницы на {server} сервере, фрапс приложил, п. 3.1",
            "На {server} сервере {offender} специально задавил меня машиной (ДБ) на авторынке, видео есть",
            "НонРП коп {offender} надел наручники без отыгровки и посадил в КПЗ на {server} сервере! Пункт 3.5",
            "Жалоба на {offender}: СК на спавне нашей семьи, убивает сразу при выходе из дома, сервер {server}",
            "Игрок {offender} вышел из игры во время ареста (уход от РП) на {server} сервере, откат прикреплен",
        ],
        "admin_reply": "Здравствуйте! Материалы переданы следящей администрации для проверки нарушения RP-процесса (п. 3.1 / 3.2 / 3.5). Приоритет: medium.",
        "attachments": [1, 2],
    },
    {
        "kind": "complaint",
        "category": "rp_violation",
        "priority": "medium",
        "lang": "en",
        "player_texts": [
            "Player {offender} killed me without reason (DM) near spawn on server {server}, video attached",
            "NonRP cop {offender} arrested me without any RP dialogue on server {server}, rule 3.5 violation",
        ],
        "admin_reply": "Hello! Your report regarding RP violation (rule 3.1 / 3.5) has been forwarded to moderators. Priority: medium.",
        "attachments": [1],
    },
    # scam_fraud (high)
    {
        "kind": "complaint",
        "category": "scam_fraud",
        "priority": "high",
        "lang": "ru",
        "player_texts": [
            "Игрок {offender} обманул меня при обмене машинами на {server} сервере! Кинул на BMW M5 и 5кк, фрапс сделки есть",
            "НонРП обман от {offender} на сервере {server}: договорились на продажу дома с гаражом, а он подсунул квартиру, п. 4.1",
            "Меня развел мошенник {offender} на скин и 2 миллиона виртов при трейде на {server} сервере, видео приложил",
            "Скам на авторынке! Игрок {offender} кинул обмен с моей доплатой вместо своей, сервер {server}",
        ],
        "admin_reply": "Здравствуйте! Жалоба на NonRP обман (п. 4.1, scam_fraud, high) передана техническому администратору для проверки логов.",
        "attachments": [1, 2],
    },
    {
        "kind": "complaint",
        "category": "scam_fraud",
        "priority": "high",
        "lang": "kk",
        "player_texts": [
            "{server} серверде {offender} көлік алмасу кезінде мені алдап кетті (скам), видео дәлел бар",
        ],
        "admin_reply": "Сәлеметсіз бе! Алаяқтық бойынша шағым (п. 4.1, scam_fraud, high) логтарды тексеру үшін жіберілді.",
        "attachments": [1],
    },
    # cheating_exploits (critical)
    {
        "kind": "complaint",
        "category": "cheating_exploits",
        "priority": "critical",
        "lang": "ru",
        "player_texts": [
            "Срочно! Игрок {offender} на {server} сервере летает на спидхаке и телепортируется по карте, читер! Видео есть",
            "Читер {offender} с аимом и ГМ (бессмертием) разносит военную базу на {server} сервере, п. 5.1",
            "Игрок {offender} использует багоюз и дюпает вирты через багажник на {server} сервере, проверьте логи срочно! п. 5.2",
            "На {server} сервере {offender} стреляет сквозь стены с софтом, приложил фрапс",
        ],
        "admin_reply": "Здравствуйте! Критическое обращение (critical, cheating_exploits) по п. 5.1/5.2 немедленно передано администрации для проверки античит-логов.",
        "attachments": [1, 2],
    },
    # ban_appeal (high)
    {
        "kind": "appeal",
        "category": "ban_appeal",
        "priority": "high",
        "lang": "ru",
        "player_texts": [
            "Меня забанили за читы на {server} сервере, хотя я играл без софта! Ник {reporter}, админ {offender}, прошу разбан",
            "Апелляция на блокировку аккаунта {reporter} на {server} сервере. Выдали бан на 30 дней по ошибке, скрин окна бана приложил",
            "Не согласен с деморганом и варном от администратора {offender} на {server} сервере, разбаньте пожалуйста, п. 7.1",
            "Забанили аккаунт {reporter} до выяснений на {server} сервере, готов пройти проверку, снимите блокировку",
        ],
        "admin_reply": "Здравствуйте! Ваша апелляция (ban_appeal, high) оформлена по п. 7.1 и передана Главной Администрации сервера.",
        "attachments": [1],
    },
    {
        "kind": "appeal",
        "category": "ban_appeal",
        "priority": "high",
        "lang": "uz",
        "player_texts": [
            "Mening {reporter} akkauntimni {server}-serverda xato ban qilishdi, iltimos bandan chiqaring, skrinshot bor",
        ],
        "admin_reply": "Assalomu alaykum! Apellyatsiya 7.1-band bo'yicha bosh ma'muriyatga yuborildi (ban_appeal, high).",
        "attachments": [1],
    },
    # billing_refund (high)
    {
        "kind": "complaint",
        "category": "billing_refund",
        "priority": "high",
        "lang": "ru",
        "player_texts": [
            "Я задонатил 1500 рублей на {server} сервер (ник {reporter}), деньги с карты списались, а GC не пришли! Чек приложил",
            "Прошу вернуть деньги за донат 5000 тенге, случайно пополнил не тот сервер ({server}), квитанция оплаты есть",
            "Двойное списание при покупке VIP-статуса на {server} сервере, ник {reporter}, верните средства пожалуйста, п. 6.1",
            "Не пришел донат Х2 на аккаунт {reporter} (сервер {server}), оплатил час назад по СБП, чек в вложении",
        ],
        "admin_reply": "Здравствуйте! Запрос по платежам (billing_refund, high, п. 6.1/6.2) передан в финансовый отдел поддержки.",
        "attachments": [1, 2],
    },
    # account_recovery (critical)
    {
        "kind": "complaint",
        "category": "account_recovery",
        "priority": "critical",
        "lang": "ru",
        "player_texts": [
            "Помогите срочно! Мой аккаунт {reporter} на {server} сервере взломали, сменили пароль и почту!",
            "Украли аккаунт {reporter} на {server} сервере, кто-то сейчас сидит на нем в игре! Все рег. данные и чеки есть",
            "Потерял телефон с Google Authenticator (2FA), не могу войти в аккаунт {reporter} на {server} сервере, п. 1.3",
            "Забыл пароль от аккаунта {reporter} на {server} сервере, почта не приходит, помогите восстановить доступ",
        ],
        "admin_reply": "Здравствуйте! Обращение по безопасности аккаунта (account_recovery, critical, п. 1.2/1.3) передано техническому специалисту.",
        "attachments": [0, 1],
    },
    # technical_bug (medium)
    {
        "kind": "bug",
        "category": "technical_bug",
        "priority": "medium",
        "lang": "ru",
        "player_texts": [
            "После утреннего рестарта на {server} сервере у меня пропал тюнинг с машины, ник {reporter}, скрин ПТС есть",
            "Постоянно вылетает игра (краш) при въезде в Арзамас на {server} сервере, телефон Xiaomi Android 14",
            "Баг с квестом автошколы на {server} сервере: маркер не появляется, перезаход не помогает, п. 8.1",
            "Не заходит на {server} сервер, пишет ошибка подключения, хотя интернет отличный, ник {reporter}",
        ],
        "admin_reply": "Здравствуйте! Информация о технической ошибке (technical_bug, medium, п. 8.1) зафиксирована.",
        "attachments": [0, 1],
    },
    # gameplay_question (low)
    {
        "kind": "question",
        "category": "gameplay_question",
        "priority": "low",
        "lang": "ru",
        "player_texts": [
            "Подскажите пожалуйста, как открыть меню персонажа и навигатор по работам на сервере?",
            "С какого уровня доступна работа дальнобойщика и где получить права категории С?",
            "Какой командой показать паспорт, лицензии и медкарту другому игроку?",
            "Где находится военкомат и как получить военный билет на Grand Mobile?",
        ],
        "admin_reply": "Здравствуйте! Вы можете использовать команды /mm (меню), /gps (навигатор), /pass и /lic согласно п. 8.2 и п. 1.1. (gameplay_question, low)",
        "attachments": [0],
    },
    {
        "kind": "question",
        "category": "gameplay_question",
        "priority": "low",
        "lang": "en",
        "player_texts": [
            "Hello, how to open GPS navigation and find jobs on the server? Which command shows licenses?",
        ],
        "admin_reply": "Hello! Use /gps for navigation, /mm for menu, and /lic for licenses per rule 8.2 (gameplay_question, low).",
        "attachments": [0],
    },
    # spam_offtopic (low)
    {
        "kind": "question",
        "category": "spam_offtopic",
        "priority": "low",
        "lang": "ru",
        "player_texts": [
            "ауууууууууууу админы вы тут???? 111111111",
            "продам гараж в реале недорого пишите в лс аааааа",
            "тест тест проверка связи фывапролджэ",
            "дайте миллион виртов просто так пж пж пж пж",
        ],
        "admin_reply": "Здравствуйте! Пожалуйста, не оффтопьте и не спамьте в поддержку (п. 7.3, spam_offtopic, low).",
        "attachments": [0],
    },
]

OFFENDERS = ["Ivan_Grozny", "Maksim_Killer", "Tommy_Vercetti", "Baha_Kazakh", "Sardor_Uz", "Dark_Stalker", "Kirill_Major", "Artem_Boss"]
REPORTERS = ["Alex_Mercer", "Danil_Volkov", "Nurlan_Almaty", "Jasur_Tashkent", "Vlad_Smirnov", "Egor_Kreed", "Timur_Khan"]
SERVERS = [1, 2, 3, 5, 7, 10, 12, 15, 22, 28]


def generate_train_records(n: int = 240) -> list[dict]:
    rng = random.Random(RANDOM_SEED)
    records = []
    for i in range(n):
        tpl = TRAIN_TEMPLATES[i % len(TRAIN_TEMPLATES)]
        text_tpl = rng.choice(tpl["player_texts"])
        offender = rng.choice(OFFENDERS)
        reporter = rng.choice(REPORTERS)
        server = rng.choice(SERVERS)
        text = text_tpl.format(offender=offender, reporter=reporter, server=server)
        attachments = rng.choice(tpl["attachments"])

        messages = [
            {"role": "player", "text": text, "attachments": attachments}
        ]
        # In ~50% of training samples, include admin reply in messages and omit top-level explicit labels
        # so the training pipeline tests both explicit labels and admin-reply label extraction!
        include_explicit_labels = (i % 2 == 0)
        messages.append({"role": "admin", "text": tpl["admin_reply"]})

        rec = {
            "id": f"train_{i+1:04d}",
            "is_synthetic_demo": True,
            "kind": tpl["kind"],
            "server": server,
            "messages": messages,
        }
        if include_explicit_labels:
            rec["category"] = tpl["category"]
            rec["priority"] = tpl["priority"]
        records.append(rec)
    rng.shuffle(records)
    return records


VALID_RECORDS = [
    # Distinct validation phrasings covering all categories, priorities, and edge cases
    {
        "id": "val_0001",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "insult_toxicity",
        "priority": "medium",
        "messages": [{"role": "player", "text": "Игрок Roman_ov оскорблял меня в чате дальнобойщиков на 4 сервере, скриншоты есть", "attachments": 2}]
    },
    {
        "id": "val_0002",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "insult_toxicity",
        "priority": "high",
        "messages": [{"role": "player", "text": "На 8 сервере игрок Oleg_Mongol оскорбил моих родителей в войс-чате, видеозапись приложил, нарушение п. 2.2!", "attachments": 1}]
    },
    {
        "id": "val_0003",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "insult_toxicity",
        "priority": "medium",
        "messages": [{"role": "player", "text": "Меня вчера обзывали в чате возле казино, но скриншот я сделать не успел, проверьте пожалуйста", "attachments": 0}]
    },
    {
        "id": "val_0004",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "rp_violation",
        "priority": "medium",
        "messages": [{"role": "player", "text": "Игрок Pasha_Technik расстрелял мою машину без РП причины (ДМ) на 3 сервере, фрапс с /time прикреплен", "attachments": 1}]
    },
    {
        "id": "val_0005",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "rp_violation",
        "priority": "medium",
        "messages": [{"role": "player", "text": "Жалоба на сотрудника полиции Denis_Cop на 11 сервере: посадил за решетку без РП отыгровки (НонРП коп), п. 3.5", "attachments": 2}]
    },
    {
        "id": "val_0006",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "rp_violation",
        "priority": "medium",
        "messages": [{"role": "player", "text": "Player John_Wick killed me with a car (DB) at the hospital spawn on server 6, video attached", "attachments": 1}]
    },
    {
        "id": "val_0007",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "scam_fraud",
        "priority": "high",
        "messages": [{"role": "player", "text": "Меня обманул игрок Scammer_Pro при обмене Гелика на дом на 9 сервере! Кинул трейд и вышел, фрапс есть, п. 4.1", "attachments": 2}]
    },
    {
        "id": "val_0008",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "scam_fraud",
        "priority": "high",
        "messages": [{"role": "player", "text": "5 серверде Marat_Kz деген ойыншы көлік сату кезінде мені алдап кетті, видео дәлел бар", "attachments": 1}]
    },
    {
        "id": "val_0009",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "cheating_exploits",
        "priority": "critical",
        "messages": [{"role": "player", "text": "Срочно! Игрок Hack_Master летает по воздуху на спидхаке и убивает всех с аимом на 1 сервере! Видео приложил, п. 5.1", "attachments": 1}]
    },
    {
        "id": "val_0010",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "cheating_exploits",
        "priority": "critical",
        "messages": [{"role": "player", "text": "Обнаружил игрока Duper_Man, который багоюзит и дюпает предметы через склад семьи на 14 сервере, проверьте логи!", "attachments": 1}]
    },
    {
        "id": "val_0011",
        "is_synthetic_demo": True,
        "kind": "appeal",
        "category": "ban_appeal",
        "priority": "high",
        "messages": [{"role": "player", "text": "Здравствуйте, мой аккаунт Nikita_Bell на 2 сервере забанили на 15 дней по ошибке, прошу разбанить! Скрин окна блокировки есть", "attachments": 1}]
    },
    {
        "id": "val_0012",
        "is_synthetic_demo": True,
        "kind": "appeal",
        "category": "ban_appeal",
        "priority": "high",
        "messages": [{"role": "player", "text": "Mening Aziz_Uz akkauntimni 7-serverda nohaq ban qilishdi, iltimos bandan chiqaring, skrinshot bor", "attachments": 1}]
    },
    {
        "id": "val_0013",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "billing_refund",
        "priority": "high",
        "messages": [{"role": "player", "text": "Оплатил донат на 2000 рублей на 5 сервер (ник Anton_Cash), деньги списали, но коины не зачислились! Чек банка прикрепил", "attachments": 1}]
    },
    {
        "id": "val_0014",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "billing_refund",
        "priority": "high",
        "messages": [{"role": "player", "text": "Верните пожалуйста деньги за покупку донат-набора на 10 сервере, купил случайно, чек есть", "attachments": 1}]
    },
    {
        "id": "val_0015",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "account_recovery",
        "priority": "critical",
        "messages": [{"role": "player", "text": "Помогите срочно! Взломали мой аккаунт Sergey_Rich на 12 сервере, сменили пароль, не могу зайти!", "attachments": 1}]
    },
    {
        "id": "val_0016",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "account_recovery",
        "priority": "critical",
        "messages": [{"role": "player", "text": "Потерял доступ к Google Authenticator (2FA) от аккаунта Max_Payne на 3 сервере, помогите отвязать по рег. данным", "attachments": 0}]
    },
    {
        "id": "val_0017",
        "is_synthetic_demo": True,
        "kind": "bug",
        "category": "technical_bug",
        "priority": "medium",
        "messages": [{"role": "player", "text": "После обновления игра постоянно вылетает (крашит) при открытии инвентаря на 6 сервере, устройство Poco X5", "attachments": 1}]
    },
    {
        "id": "val_0018",
        "is_synthetic_demo": True,
        "kind": "bug",
        "category": "technical_bug",
        "priority": "medium",
        "messages": [{"role": "player", "text": "Из-за внезапного рестарта 15 сервера пропали вещи из багажника моего авто, ник Dmitry_Fox", "attachments": 0}]
    },
    {
        "id": "val_0019",
        "is_synthetic_demo": True,
        "kind": "question",
        "category": "gameplay_question",
        "priority": "low",
        "messages": [{"role": "player", "text": "Подскажите пожалуйста, какой командой открыть навигатор GPS и где устроиться на работу автобусника?", "attachments": 0}]
    },
    {
        "id": "val_0020",
        "is_synthetic_demo": True,
        "kind": "question",
        "category": "gameplay_question",
        "priority": "low",
        "messages": [{"role": "player", "text": "Как показать паспорт и медицинскую карту на собеседовании во фракцию?", "attachments": 0}]
    },
    {
        "id": "val_0021",
        "is_synthetic_demo": True,
        "kind": "question",
        "category": "spam_offtopic",
        "priority": "low",
        "messages": [{"role": "player", "text": "ааааааааа 111111111 админы дайте 10 миллионов виртов просто так пж пж", "attachments": 0}]
    },
    {
        "id": "val_0022",
        "is_synthetic_demo": True,
        "kind": "complaint",
        "category": "insult_toxicity",
        "priority": "medium",
        "messages": [{"role": "player", "text": "Игрок Bad_Boy нарушил правило 99.9 и обозвал меня в чате на 2 сервере, но скринов нет", "attachments": 0}]
    }
]


def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "data"
    examples_dir = root / "examples"
    data_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    train_records = generate_train_records(240)
    train_gz_path = data_dir / "synthetic_tickets_train.jsonl.gz"
    with gzip.open(train_gz_path, "wt", encoding="utf-8") as f:
        for rec in train_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    valid_path = data_dir / "synthetic_tickets_valid.jsonl"
    with open(valid_path, "w", encoding="utf-8") as f:
        for rec in VALID_RECORDS:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sample_in_path = examples_dir / "sample_tickets.jsonl"
    with open(sample_in_path, "w", encoding="utf-8") as f:
        for rec in VALID_RECORDS[:10]:
            # Strip ground-truth labels for sample input demonstration
            clean = {k: v for k, v in rec.items() if k not in ("category", "priority")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    print(f"Generated {len(train_records)} synthetic train tickets -> {train_gz_path}")
    print(f"Generated {len(VALID_RECORDS)} synthetic valid tickets -> {valid_path}")
    print(f"Generated 10 sample input tickets -> {sample_in_path}")


if __name__ == "__main__":
    main()

