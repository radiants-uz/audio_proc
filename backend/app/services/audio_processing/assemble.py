"""§5.7: транскрипт по схеме §7. Спикеры переименовываются в порядке
первого появления (SPEAKER_00 — первый заговоривший)."""

from app.services.audio_processing.langid import LangResult
from app.services.audio_processing.types import Segment

PIPELINE_VERSION = "1.1.0"


def rename_speakers(segments: list[Segment]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for seg in sorted(segments, key=lambda s: s.start):
        if seg.speaker not in mapping:
            mapping[seg.speaker] = f"SPEAKER_{len(mapping):02d}"
    for seg in segments:
        seg.speaker = mapping[seg.speaker]
        seg.overlap_with = sorted(mapping.get(o, o) for o in seg.overlap_with)
        for ch in seg.chunks:
            ch.speaker = mapping.get(ch.speaker, ch.speaker)
    return mapping


def assemble_transcript(
    job_id: str,
    filename: str,
    duration: float,
    segments: list[Segment],
    languages: dict[str, LangResult],
    models: dict,
    timings: dict[str, float],
    config: dict,
    warnings: list[str],
    title: str | None = None,
    speaker_names: dict[str, str] | None = None,
) -> dict:
    segments = sorted(segments, key=lambda s: (s.start, s.end))
    for i, seg in enumerate(segments):
        seg.id = i

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for seg in segments:
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + seg.duration
        counts[seg.speaker] = counts.get(seg.speaker, 0) + 1

    speaker_names = speaker_names or {}
    speakers = [
        {
            "id": spk,
            "name": speaker_names.get(spk),
            "language": languages[spk].language if spk in languages else "",
            "lang_confidence": round(languages[spk].confidence, 3) if spk in languages else 0.0,
            # раскладка по языкам (режим по кускам): [{language, speech_sec, share}]
            "languages": languages[spk].languages if spk in languages else None,
            "total_speech_sec": round(totals[spk], 2),
            "segments_count": counts[spk],
        }
        for spk in sorted(totals, key=lambda s: -totals[s])
    ]

    return {
        "job_id": job_id,
        "title": title,
        "audio": {"filename": filename, "duration_sec": round(duration, 2), "sample_rate": 16000},
        "speakers": speakers,
        "segments": [
            {
                "id": seg.id,
                "speaker": seg.speaker,
                "language": seg.language,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "overlap": seg.overlap,
                "overlap_with": seg.overlap_with,
                "text": seg.text,
                "lang_confidence": round(seg.lang_confidence, 3),
                "words": seg.words,
                # куски ASR с языком каждого (null в legacy-режиме)
                "chunks": [
                    {
                        "start": round(c.start, 2), "end": round(c.end, 2), "language": c.language,
                        "lang_confidence": round(c.lang_confidence, 3), "lang_source": c.lang_source,
                        "text": c.text, **({"warning": c.warning} if c.warning else {}),
                        **({"lid_language": c.lid_language} if c.lid_language else {}),
                    }
                    for c in seg.chunks
                ] if seg.chunks else None,
            }
            for seg in segments
        ],
        "meta": {
            "pipeline_version": PIPELINE_VERSION,
            "models": models,
            "timings_sec": {k: round(v, 2) for k, v in timings.items()},
            "config": config,
            "warnings": warnings,
        },
    }


def apply_speaker_names(transcript: dict, speaker_names: dict[str, str]) -> dict:
    for spk in transcript.get("speakers", []):
        if spk["id"] in speaker_names:
            spk["name"] = speaker_names[spk["id"]]
    return transcript
