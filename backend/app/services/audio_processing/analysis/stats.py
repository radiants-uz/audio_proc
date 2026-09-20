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
        by_lang[s.get("language") or ""] = by_lang.get(s.get("language") or "", 0.0) + (s["end"] - s["start"])
    by_language = [
        {"language": lang, "speech_sec": round(secs, 2), "share": round(secs / total, 3) if total else 0.0}
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
