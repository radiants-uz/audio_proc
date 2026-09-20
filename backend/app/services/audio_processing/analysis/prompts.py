"""Промпты анализа. Системный промпт — на языке отчёта; транскрипт — как есть."""

PROMPT_VERSION = "1.0"

_LANG_NAMES = {
    "ru": {"ru": "русском", "uz": "узбекском", "en": "английском"},
    "uz": {"ru": "rus", "uz": "o'zbek", "en": "ingliz"},
    "en": {"ru": "Russian", "uz": "Uzbek", "en": "English"},
}

SYSTEM = {
    "ru": """Ты — аналитик совещаний. Тебе дан транскрипт разговора нескольких участников (автоматическое распознавание речи, возможны ошибки). Участники говорят на узбекском, русском или английском.

Задача: составить структурированный анализ СТРОГО в JSON по заданной схеме.

Правила:
1. Все текстовые поля пиши ТОЛЬКО на {lang} языке. Цитаты участников (key_quotes.original) оставляй на языке оригинала, перевод — в key_quotes.translated.
2. Опирайся только на транскрипт. Ничего не додумывай. Если решений или задач не прозвучало — верни пустые списки.
3. Каждое решение (decisions) и задача (action_items) — со ссылкой на segment_ids реплик, где это прозвучало. Темы (topics) — тоже с segment_ids.
4. Явно неразборчивые или бессмысленные фрагменты (ошибки распознавания) не интерпретируй.
5. speakers: по каждому участнику — role_hint (только если роль явно следует из разговора, иначе null), position (его позиция/вклад), 1–3 ключевые цитаты.
6. title — короткое название встречи (до 10 слов). summary — 3–8 предложений.
7. open_questions — вопросы, оставшиеся без ответа.
{stats_hint}""",
    "uz": """Siz — yig'ilishlar tahlilchisisiz. Sizga bir nechta ishtirokchi suhbatining transkripti berilgan (avtomatik nutqni tanish, xatolar bo'lishi mumkin). Ishtirokchilar o'zbek, rus yoki ingliz tilida gapiradi.

Vazifa: berilgan sxema bo'yicha QAT'IY JSON ko'rinishida tuzilgan tahlil tayyorlash.

Qoidalar:
1. Barcha matn maydonlarini FAQAT {lang} tilida yozing. Ishtirokchilar iqtiboslarini (key_quotes.original) asl tilida qoldiring, tarjimasini key_quotes.translated ga yozing.
2. Faqat transkriptga tayaning. Hech narsani o'ylab topmang. Qarorlar yoki topshiriqlar aytilmagan bo'lsa — bo'sh ro'yxat qaytaring.
3. Har bir qaror (decisions) va topshiriq (action_items) — u aytilgan replikalarning segment_ids havolasi bilan. Mavzular (topics) ham segment_ids bilan.
4. Aniq tushunarsiz yoki ma'nosiz qismlarni (tanish xatolari) talqin qilmang.
5. speakers: har bir ishtirokchi bo'yicha — role_hint (faqat rol suhbatdan aniq kelib chiqsa, aks holda null), position (uning pozitsiyasi/hissasi), 1–3 asosiy iqtibos.
6. title — uchrashuvning qisqa nomi (10 so'zgacha). summary — 3–8 gap.
7. open_questions — javobsiz qolgan savollar.
{stats_hint}""",
    "en": """You are a meeting analyst. You are given a transcript of a multi-speaker conversation (automatic speech recognition, errors are possible). Speakers talk in Uzbek, Russian or English.

Task: produce a structured analysis STRICTLY as JSON matching the given schema.

Rules:
1. Write every text field ONLY in {lang}. Keep speaker quotes (key_quotes.original) in the original language and put the translation in key_quotes.translated.
2. Rely only on the transcript. Do not invent anything. If no decisions or action items were voiced, return empty lists.
3. Every decision and action item must reference segment_ids of the lines where it was said. Topics must reference segment_ids too.
4. Do not interpret clearly garbled or meaningless fragments (recognition errors).
5. speakers: for each participant — role_hint (only if the role clearly follows from the conversation, else null), position (their stance/contribution), 1–3 key quotes.
6. title — a short meeting title (up to 10 words). summary — 3–8 sentences.
7. open_questions — questions left unanswered.
{stats_hint}""",
}

STATS_HINT = {
    "ru": "Статистика (посчитана программно, используй как контекст): {stats}",
    "uz": "Statistika (dasturiy hisoblangan, kontekst sifatida foydalaning): {stats}",
    "en": "Statistics (computed programmatically, use as context): {stats}",
}

USER_FULL = {
    "ru": "Транскрипт (формат: [чч:мм:сс] СПИКЕР (язык): текст; [overlap] — наложение голосов):\n\n{transcript}\n\nВерни JSON по схеме.",
    "uz": "Transkript (format: [ss:dd:ss] SPIKER (til): matn; [overlap] — ovozlar ustma-ust):\n\n{transcript}\n\nSxema bo'yicha JSON qaytaring.",
    "en": "Transcript (format: [hh:mm:ss] SPEAKER (lang): text; [overlap] — overlapping speech):\n\n{transcript}\n\nReturn JSON per the schema.",
}

# map-reduce: анализ куска — та же схема; итог — по промежуточным анализам.
USER_CHUNK = {
    "ru": "Это часть {i} из {n} длинного транскрипта. Проанализируй ТОЛЬКО эту часть по схеме (segment_ids — из этой части).\n\n{transcript}",
    "uz": "Bu uzun transkriptning {n} qismidan {i}-qismi. FAQAT shu qismni sxema bo'yicha tahlil qiling (segment_ids — shu qismdan).\n\n{transcript}",
    "en": "This is part {i} of {n} of a long transcript. Analyse ONLY this part per the schema (segment_ids from this part).\n\n{transcript}",
}

USER_REDUCE = {
    "ru": "Ниже — промежуточные анализы {n} частей одного длинного разговора (JSON). Объедини их в ОДИН итоговый анализ по той же схеме: убери дубли, сохрани segment_ids из частей, summary — по всему разговору.\n\n{parts}",
    "uz": "Quyida bitta uzun suhbatning {n} qismi bo'yicha oraliq tahlillar (JSON). Ularni o'sha sxema bo'yicha BITTA yakuniy tahlilga birlashtiring: takrorlarni olib tashlang, qismlardagi segment_ids ni saqlang, summary — butun suhbat bo'yicha.\n\n{parts}",
    "en": "Below are intermediate analyses of {n} parts of one long conversation (JSON). Merge them into ONE final analysis with the same schema: remove duplicates, keep segment_ids from the parts, summary — for the whole conversation.\n\n{parts}",
}

REPAIR = {
    "ru": "Предыдущий ответ не прошёл валидацию схемы: {error}\nВерни исправленный JSON строго по схеме.",
    "uz": "Oldingi javob sxema tekshiruvidan o'tmadi: {error}\nSxemaga qat'iy mos tuzatilgan JSON qaytaring.",
    "en": "The previous answer failed schema validation: {error}\nReturn the corrected JSON strictly per the schema.",
}


def system_prompt(report_lang: str, stats_json: str) -> str:
    lang = report_lang if report_lang in SYSTEM else "ru"
    hint = STATS_HINT[lang].format(stats=stats_json)
    return SYSTEM[lang].format(lang=_LANG_NAMES[lang][lang], stats_hint=hint)


def t(table: dict, report_lang: str) -> str:
    return table.get(report_lang, table["ru"])
