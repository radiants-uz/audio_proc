"""§5.8: оркестрация анализа. Вход — транскрипт §7, выход — analysis.json §7.1
или None (analysis_failed), транскрипт из-за LLM не пропадает."""

import json
import logging
from collections.abc import Awaitable, Callable

from pydantic import ValidationError

from app.core.config import settings
from app.models.service import Service
from app.schemas.analysis import Analysis, LlmAnalysis, llm_json_schema
from app.services.audio_processing.analysis import prompts
from app.services.audio_processing.analysis.llm_base import LlmClient, LlmUsage
from app.services.audio_processing.analysis.stats import compute_stats
from app.services.audio_processing.analysis.tokens import estimate_tokens

logger = logging.getLogger(__name__)


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def format_transcript_lines(transcript: dict, speaker_names: dict[str, str] | None = None) -> list[tuple[int, str]]:
    names = {s["id"]: (s.get("name") or s["id"]) for s in transcript.get("speakers", [])}
    names.update(speaker_names or {})
    lines = []
    for seg in transcript.get("segments", []):
        if not (seg.get("text") or "").strip():
            continue
        tag = " [overlap]" if seg.get("overlap") else ""
        lines.append((seg["id"], f"[{_hms(seg['start'])}] #{seg['id']} {names.get(seg['speaker'], seg['speaker'])} ({seg.get('language', '')}){tag}: {seg['text'].strip()}"))
    return lines


def resolve_report_lang(transcript: dict, requested: str) -> str:
    if requested and requested != "auto":
        return requested
    speakers = transcript.get("speakers", [])
    if speakers:
        best = max(speakers, key=lambda s: s.get("total_speech_sec", 0))
        if best.get("language") in ("uz", "ru", "en"):
            return best["language"]
    return "ru"


def chunk_lines(lines: list[tuple[int, str]], chunk_tokens: int) -> list[list[tuple[int, str]]]:
    chunks, cur, cur_tokens = [], [], 0
    for item in lines:
        n = estimate_tokens(item[1]) + 1
        if cur and cur_tokens + n > chunk_tokens:
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append(item)
        cur_tokens += n
    if cur:
        chunks.append(cur)
    return chunks


async def _ask(llm: LlmClient, system: str, user: str, max_tokens: int, lang: str, usage: LlmUsage) -> LlmAnalysis:
    """Один вызов + один повтор с сообщением об ошибке валидации."""
    schema = llm_json_schema()
    resp = await llm.complete_json(system, user, schema, max_tokens)
    usage.add(resp.usage)
    try:
        return LlmAnalysis.model_validate(resp.data)
    except ValidationError as e:
        logger.warning("LLM JSON failed validation, retrying once: %s", str(e)[:300])
        repair = prompts.t(prompts.REPAIR, lang).format(error=str(e)[:1500])
        resp = await llm.complete_json(system, user + "\n\n" + repair, schema, max_tokens)
        usage.add(resp.usage)
        return LlmAnalysis.model_validate(resp.data)


async def analyze_transcript(
    llm: LlmClient,
    service: Service,
    transcript: dict,
    report_lang: str,
    speaker_names: dict[str, str] | None = None,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
) -> dict | None:
    lang = resolve_report_lang(transcript, report_lang)
    stats = compute_stats(transcript)
    if speaker_names:
        for s in stats["by_speaker"]:
            s["name"] = speaker_names.get(s["id"], s.get("name"))
    lines = format_transcript_lines(transcript, speaker_names)
    warnings: list[str] = []

    if not lines:
        return {
            **LlmAnalysis(title=transcript.get("title") or "", summary="").model_dump(),
            "report_lang": lang,
            "stats": stats,
            "meta": {"llm": service.model_name, "prompt_version": prompts.PROMPT_VERSION,
                     "input_tokens": 0, "output_tokens": 0, "chunked": False, "chunks": 0,
                     "warnings": ["no_speech"]},
        }

    max_input = service.max_input_tokens or settings.analysis.max_input_tokens
    chunk_tokens = service.chunk_tokens or settings.analysis.chunk_tokens
    max_output = service.max_output_tokens or settings.analysis.max_output_tokens
    stats_json = json.dumps(stats, ensure_ascii=False)
    system = prompts.system_prompt(lang, stats_json)
    usage = LlmUsage()
    total_tokens = sum(estimate_tokens(line[1]) for line in lines)

    try:
        if total_tokens <= max_input:
            if on_progress:
                await on_progress(0, 1)
            user = prompts.t(prompts.USER_FULL, lang).format(transcript="\n".join(line[1] for line in lines))
            result = await _ask(llm, system, user, max_output, lang, usage)
            chunked, n_chunks = False, 1
        else:
            chunks = chunk_lines(lines, chunk_tokens)
            n_chunks = len(chunks)
            partials: list[LlmAnalysis] = []
            for i, chunk in enumerate(chunks, start=1):
                if on_progress:
                    await on_progress(i - 1, n_chunks + 1)
                user = prompts.t(prompts.USER_CHUNK, lang).format(
                    i=i, n=n_chunks, transcript="\n".join(line[1] for line in chunk)
                )
                partials.append(await _ask(llm, system, user, max_output, lang, usage))
            if on_progress:
                await on_progress(n_chunks, n_chunks + 1)
            parts = "\n\n".join(
                f"--- part {i} ---\n{json.dumps(p.model_dump(), ensure_ascii=False)}"
                for i, p in enumerate(partials, start=1)
            )
            # Итоговый прогон тоже должен влезть в контекст: если промежуточные
            # анализы слишком велики — обрезаем цитаты.
            if estimate_tokens(parts) > max_input:
                for p in partials:
                    for s in p.speakers:
                        s.key_quotes = s.key_quotes[:1]
                parts = "\n\n".join(
                    f"--- part {i} ---\n{json.dumps(p.model_dump(), ensure_ascii=False)}"
                    for i, p in enumerate(partials, start=1)
                )
            user = prompts.t(prompts.USER_REDUCE, lang).format(n=n_chunks, parts=parts)
            result = await _ask(llm, system, user, max_output, lang, usage)
            chunked = True
    except Exception as e:
        logger.exception("analysis failed: %s", e)
        return None

    valid_ids = {s["id"] for s in transcript.get("segments", [])}
    for coll in (result.topics, result.decisions, result.action_items):
        for item in coll:
            item.segment_ids = [i for i in item.segment_ids if i in valid_ids]

    analysis = Analysis(
        **result.model_dump(),
        report_lang=lang,
        stats=stats,
        meta={
            "llm": service.model_name,
            "prompt_version": prompts.PROMPT_VERSION,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "chunked": chunked,
            "chunks": n_chunks,
            "warnings": warnings,
        },
    )
    return analysis.model_dump(mode="json")
