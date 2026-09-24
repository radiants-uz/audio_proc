"""§5.8 п.7: статистика считается кодом, не LLM."""


def compute_stats(transcript: dict) -> dict:
    segments = transcript.get("segments", [])
    speakers = transcript.get("speakers", [])
    total = sum(s["end"] - s["start"] for s in segments)
    by_speaker = []
    for spk in speakers:
        secs = sum(s["end"] - s["start"] for s in segments if s["speaker"] == spk["id"])
        by_speaker.append({
            "id": spk["id"],
            "name": spk.get("name"),
            "language": spk.get("language", ""),
            "speech_sec": round(secs, 2),
            "share": round(secs / total, 3) if total else 0.0,
            "segments": sum(1 for s in segments if s["speaker"] == spk["id"]),
        })
    by_lang: dict[str, float] = {}
    for s in segments:
        # По кускам, если есть (внутри сегмента языки могут различаться), иначе по сегменту.
        # Куски режутся с паддингом и по VAD — считаем только их время внутри сегмента,
        # а доли нормируем на сумму по языкам, чтобы они сходились к 1.
        parts = s.get("chunks") or [{"language": s.get("language"), "start": s["start"], "end": s["end"]}]
        for c in parts:
            lang = c.get("language") or ""
            secs = max(0.0, min(c["end"], s["end"]) - max(c["start"], s["start"]))
            by_lang[lang] = by_lang.get(lang, 0.0) + secs
    lang_total = sum(by_lang.values())
    by_language = [
        {"language": lang, "speech_sec": round(secs, 2), "share": round(secs / lang_total, 3) if lang_total else 0.0}
        for lang, secs in sorted(by_lang.items(), key=lambda kv: -kv[1])
    ]
    return {
        "total_speech_sec": round(total, 2),
        "speakers_count": len(speakers),
        "segments_count": len(segments),
        "overlap_segments": sum(1 for s in segments if s.get("overlap")),
        "avg_segment_sec": round(total / len(segments), 2) if segments else 0.0,
        "by_speaker": by_speaker,
        "by_language": by_language,
    }


def dominant_language(transcript: dict) -> str | None:
    """Язык большинства по времени речи всего файла (по кускам, если они есть).
    Спикеры теперь могут быть двуязычными, поэтому язык «главного» спикера — не то же."""
    by_language = compute_stats(transcript)["by_language"]
    for item in by_language:
        if item["language"]:
            return item["language"]
    speakers = transcript.get("speakers") or []
    return speakers[0].get("language") or None if speakers else None
