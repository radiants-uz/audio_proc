"""Оркестрация пайплайна (§2): ingest → diarization → segments → LID → ASR →
assemble; отдельно analyze и report. Также CLI для отладки:
    python -m app.services.audio_processing.run --audio x.wav [--stop-after diarize]
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from app.core.config import settings
from app.models.service import Service
from app.schemas.progress import Stage
from app.schemas.task_context import TaskParams
from app.services.audio_processing import assemble
from app.services.audio_processing.analysis.analyze import analyze_transcript
from app.services.audio_processing.analysis.llm_openai import OpenAICompatibleClient
from app.services.audio_processing.asr import build_asr_client
from app.services.audio_processing.enhance import enhance_for_asr
from app.services.audio_processing.ingest import IngestError, load_wav, to_wav16k
from app.services.audio_processing.langid import (
    identify_speakers,
    label_chunks,
    recheck_segments,
    speaker_languages,
)
from app.services.audio_processing.models import ModelRegistry
from app.services.audio_processing.report.build import DOCX_CT, PDF_CT, build_docx, docx_to_pdf
from app.services.audio_processing.router import has_words, plan_chunks, transcribe_segments
from app.services.audio_processing.segments import SegmentConfig, build_segments
from app.services.audio_processing.speakers import merge_speakers
from app.services.progress import Progress

logger = logging.getLogger(__name__)


class _NullProgress:
    async def set_stage(self, *a, **k): ...
    async def set_total(self, *a, **k): ...
    async def set_current(self, *a, **k): ...
    async def increment_current(self, *a, **k): ...
    async def add_warning(self, *a, **k): ...


class PipelineRunner:
    def __init__(self, models: ModelRegistry, progress: Progress | None, work_dir: Path, llm_service: Service | None = None):
        self.models = models
        self.progress = progress or _NullProgress()
        self.work_dir = work_dir
        self.llm_service = llm_service

    # ------------------------------------------------------------------
    async def transcribe(self, input_path: Path, filename: str, job_id: str, params: TaskParams) -> dict:
        timings: dict[str, float] = {}
        warnings: list[str] = []
        t_total = time.perf_counter()

        # [1] ingest
        t0 = time.perf_counter()
        wav_path = self.work_dir / "audio16k.wav"
        duration = await asyncio.to_thread(to_wav16k, input_path, wav_path)
        if duration < settings.audio.min_duration_sec:
            raise IngestError(f"Audio is too short: {duration:.1f}s")
        if duration > settings.audio.max_duration_sec:
            raise IngestError(f"Audio is too long: {duration / 60:.0f} min, limit {settings.audio.max_duration_sec / 60:.0f} min")
        audio = await asyncio.to_thread(load_wav, wav_path)
        timings["ingest"] = time.perf_counter() - t0
        logger.info("ingest: %.1fs of audio in %.1fs", duration, timings["ingest"])

        # [1b] улучшение звука для ASR и LID (диаризация — по исходному сигналу)
        t0 = time.perf_counter()
        enh_path, enh_filters, stats = await asyncio.to_thread(
            enhance_for_asr, wav_path, self.work_dir / "audio16k_enh.wav", self.models.denoiser
        )
        asr_audio = audio if enh_path == wav_path else await asyncio.to_thread(load_wav, enh_path)
        timings["enhance"] = time.perf_counter() - t0

        # [2] diarization
        await self.progress.set_stage(Stage.DIARIZATION)
        t0 = time.perf_counter()
        min_spk = params.expected_speakers or settings.diarization.min_speakers
        max_spk = params.expected_speakers or settings.diarization.max_speakers
        diar = await asyncio.to_thread(self.models.diarizer.diarize, wav_path, min_spk, max_spk)
        timings["diarization"] = time.perf_counter() - t0
        n_found = len({t.speaker for t in diar.turns})
        logger.info("diarization: %d turns, %d speakers in %.1fs", len(diar.turns), n_found, timings["diarization"])
        if params.expected_speakers and n_found < params.expected_speakers:
            warnings.append("speaker_count_low")

        # [3] segments
        seg_cfg = SegmentConfig(**settings.segments.model_dump())
        segments = build_segments(diar.exclusive_turns, diar.turns, seg_cfg, audio)
        if not segments:
            warnings.append("no_speech")

        # [3b] слияние ложных спикеров (обрывки шума, эхо) по голосовым эмбеддингам
        merge_report = None
        if segments and settings.speakers.enabled:
            merge_report = await asyncio.to_thread(
                merge_speakers, self.models.embedder.embed, audio, segments, params.expected_speakers
            )
            if merge_report.merged:
                warnings.append("speakers_merged")

        # [4] куски для ASR + язык
        chunk_lid = settings.langid.chunk_lid
        chunks = plan_chunks(segments, asr_audio, duration, self.models.vad) if segments else []
        await self.progress.set_stage(Stage.LANGID, total=max(1, len(chunks)))
        t0 = time.perf_counter()
        lid_audio = audio if settings.langid.on_raw_audio else asr_audio
        loop = asyncio.get_running_loop()

        def lid_progress(done: int, total: int) -> None:
            # label_chunks работает в потоке — прогресс возвращаем в цикл событий
            asyncio.run_coroutine_threadsafe(self.progress.set_current(done), loop)

        if chunk_lid:
            # Язык — на каждый кусок (спикер может переключать языки, §5.4).
            if chunks:
                await asyncio.to_thread(label_chunks, self.models.langid, lid_audio, chunks, warnings, lid_progress)
            # предварительно, без warnings: окончательно — после ASR (повторы меняют язык кусков)
            languages = speaker_languages(chunks)
        else:
            # legacy: язык на спикера + необязательная перепроверка сегментов
            # (lid_recheck из запроса переопределяет LANGID_PER_SEGMENT_RECHECK)
            recheck = params.lid_recheck if params.lid_recheck is not None else settings.langid.per_segment_recheck
            languages = await asyncio.to_thread(identify_speakers, self.models.langid, lid_audio, segments, warnings) if segments else {}
            for seg in segments:
                seg.language = languages[seg.speaker].language
                seg.lang_confidence = languages[seg.speaker].confidence
            if recheck and segments:
                changed = await asyncio.to_thread(recheck_segments, self.models.langid, lid_audio, segments)
                logger.info("LID recheck changed %d segments", changed)
            for ch in chunks:
                seg = next(x for x in segments if x.id == ch.seg_id)
                ch.language, ch.lang_confidence, ch.lang_source = seg.language, seg.lang_confidence, "speaker"
        timings["langid"] = time.perf_counter() - t0

        # [5] ASR
        await self.progress.set_stage(Stage.ASR, total=max(1, len(chunks)))
        t0 = time.perf_counter()
        asr = build_asr_client()
        asr_stats: dict = {}
        try:
            async def on_batch(done: int, total: int):
                await self.progress.set_current(done)

            if segments:
                served = [x for x in settings.langid.allowed if x in settings.asr.languages]
                asr_stats = await transcribe_segments(
                    asr, asr_audio, duration, segments, chunks=chunks, on_batch_done=on_batch,
                    raw_audio=audio, vad=self.models.vad, textcheck=self.models.textcheck,
                    languages_served=served,
                )
                if asr_stats.get("unsupported"):
                    warnings.append("asr_lang_unsupported")
                if asr_stats.get("rejected_chunks"):
                    warnings.append("asr_chunks_rejected")
                # Повторы и перенаправления могли сменить язык кусков — языки спикеров
                # (и warning lang_switch_detected) считаются по окончательным кускам.
                post = speaker_languages(chunks, warnings)
                if chunk_lid:
                    languages = post
                else:
                    for spk, res in post.items():
                        if spk in languages and res.language != languages[spk].language:
                            languages[spk] = res
                        elif spk in languages:
                            languages[spk].languages = res.languages
        finally:
            await asr.aclose()
        timings["asr"] = time.perf_counter() - t0
        # ASR отдал пустой текст на большинстве реплик: почти всегда это тихая,
        # дальняя или зашумлённая запись, а не отсутствие речи. Пусть ARK покажет это.
        if segments:
            empty = sum(1 for s in segments if not has_words(s.text))
            if empty / len(segments) >= settings.asr.empty_warning_ratio:
                warnings.append("asr_mostly_empty")
                logger.warning("ASR returned empty text for %d/%d segments", empty, len(segments))

        # [6] assemble
        await self.progress.set_stage(Stage.ASSEMBLE)
        mapping = assemble.rename_speakers(segments)
        languages = {mapping[old]: res for old, res in languages.items() if old in mapping}
        timings["total"] = time.perf_counter() - t_total
        for w in warnings:
            await self.progress.add_warning(w)

        return assemble.assemble_transcript(
            job_id=job_id,
            filename=filename,
            duration=duration,
            segments=segments,
            languages=languages,
            models={
                "diarization": diar.model_name,
                "speaker_embedder": self.models.embedder.model_name,
                "enhance": {"filters": enh_filters, "mean_db": stats.mean_db, "noise_floor_db": stats.noise_floor_db},
                "speakers_merge": (
                    {"before": merge_report.before, "after": merge_report.after, "merged": merge_report.merged}
                    if merge_report else None
                ),
                "langid": self.models.langid.model_name,
                "langid_mode": "chunk" if chunk_lid else "speaker",
                "asr_stats": asr_stats,
                "asr": {"backend": asr.name, "uz": "nvidia/stt_uz_fastconformer_hybrid_large_pc",
                        "ru": "gigaam-v3-e2e-rnnt", "en": "nvidia/parakeet-tdt-0.6b-v2"},
            },
            timings=timings,
            config={
                "min_speakers": min_spk, "max_speakers": max_spk,
                "lid_min_confidence": settings.langid.min_confidence,
                "chunk_min_confidence": settings.langid.chunk_min_confidence,
                "chunk_min_sec": settings.langid.chunk_min_sec,
                "text_check": settings.asr.text_check,
                "max_segment_sec": seg_cfg.max_segment_sec,
                "device": settings.audio.device,
            },
            warnings=warnings,
            title=params.title,
            speaker_names=params.speaker_names,
        )

    # ------------------------------------------------------------------
    async def analyze(self, transcript: dict, params: TaskParams) -> dict | None:
        if self.llm_service is None:
            raise RuntimeError("LLM service is not configured for this task")
        llm = OpenAICompatibleClient(self.llm_service)

        async def on_progress(done: int, total: int):
            await self.progress.set_total(total)
            await self.progress.set_current(done)

        return await analyze_transcript(
            llm, self.llm_service, transcript,
            report_lang=params.report_lang or settings.analysis.default_report_lang,
            speaker_names=params.speaker_names, on_progress=on_progress,
        )

    # ------------------------------------------------------------------
    async def report(self, transcript: dict, analysis: dict, params: TaskParams) -> tuple[bytes, str, str, list[str]]:
        docx_bytes = await asyncio.to_thread(build_docx, transcript, analysis, params.title, params.speaker_names)
        if params.report_format == "docx":
            return docx_bytes, "docx", DOCX_CT, []
        pdf = await asyncio.to_thread(docx_to_pdf, docx_bytes)
        if pdf is None:
            return docx_bytes, "docx", DOCX_CT, ["pdf_failed"]
        return pdf, "pdf", PDF_CT, []

    @staticmethod
    def apply_speaker_names(transcript: dict, speaker_names: dict[str, str]) -> dict:
        return assemble.apply_speaker_names(transcript, speaker_names)


# ---------------------------------------------------------------------- CLI
async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--stop-after", choices=["diarize", "segments", "langid", "transcribe"], default="transcribe")
    ap.add_argument("--expected-speakers", type=int)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    models = ModelRegistry()
    models.load_all()
    work = Path(settings.audio.tmp_dir) / "cli"
    work.mkdir(parents=True, exist_ok=True)

    if args.stop_after in ("diarize", "segments", "langid"):
        wav = work / "audio16k.wav"
        dur = to_wav16k(Path(args.audio), wav)
        res = models.diarizer.diarize(wav, args.expected_speakers, args.expected_speakers)
        if args.stop_after == "diarize":
            for t in res.turns:
                print(f"{t.start:8.2f} {t.end:8.2f} {t.speaker}")
            return
        audio = load_wav(wav)
        segs = build_segments(res.exclusive_turns, res.turns, SegmentConfig(**settings.segments.model_dump()), audio)
        if args.stop_after == "langid":
            warnings: list[str] = []
            langs = identify_speakers(models.langid, audio, segs, warnings)
            for spk, r in langs.items():
                print(f"{spk}: {r.language} ({r.confidence:.2f})")
            print("warnings:", warnings)
            return
        for s in segs:
            print(f"{s.id:4d} {s.start:8.2f} {s.end:8.2f} {s.speaker} overlap={s.overlap} {s.overlap_with}")
        print(f"{len(segs)} segments, duration {dur:.1f}s")
        return

    runner = PipelineRunner(models=models, progress=None, work_dir=work)
    transcript = await runner.transcribe(Path(args.audio), Path(args.audio).name, "cli", TaskParams(expected_speakers=args.expected_speakers))
    print(json.dumps(transcript, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
