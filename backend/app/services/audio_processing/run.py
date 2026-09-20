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
from app.services.audio_processing.ingest import IngestError, load_wav, to_wav16k
from app.services.audio_processing.langid import identify_speakers, recheck_segments
from app.services.audio_processing.models import ModelRegistry
from app.services.audio_processing.report.build import DOCX_CT, PDF_CT, build_docx, docx_to_pdf
from app.services.audio_processing.router import transcribe_segments
from app.services.audio_processing.segments import SegmentConfig, build_segments
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

        # [4] LID
        await self.progress.set_stage(Stage.LANGID, total=max(1, len({s.speaker for s in segments})))
        t0 = time.perf_counter()
        languages = await asyncio.to_thread(identify_speakers, self.models.langid, audio, segments, warnings) if segments else {}
        for seg in segments:
            seg.language = languages[seg.speaker].language
        recheck = params.lid_recheck if params.lid_recheck is not None else settings.langid.per_segment_recheck
        if recheck and segments:
            changed = await asyncio.to_thread(recheck_segments, self.models.langid, audio, segments)
            logger.info("LID recheck changed %d segments", changed)
        timings["langid"] = time.perf_counter() - t0

        # [5] ASR
        await self.progress.set_stage(Stage.ASR, total=max(1, len(segments)))
        t0 = time.perf_counter()
        asr = build_asr_client()
        try:
            async def on_batch(done: int, total: int):
                await self.progress.set_total(total)
                await self.progress.set_current(done)

            if segments:
                await transcribe_segments(asr, audio, duration, segments, on_batch_done=on_batch)
        finally:
            await asr.aclose()
        timings["asr"] = time.perf_counter() - t0
        # ASR отдал пустой текст на большинстве реплик: почти всегда это тихая,
        # дальняя или зашумлённая запись, а не отсутствие речи. Пусть ARK покажет это.
        if segments:
            empty = sum(1 for s in segments if not (s.text or "").strip())
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
                "langid": self.models.langid.model_name,
                "asr": {"backend": asr.name, "uz": "nvidia/stt_uz_fastconformer_hybrid_large_pc",
                        "ru": "gigaam-v3-e2e-rnnt", "en": "nvidia/parakeet-tdt-0.6b-v2"},
            },
            timings=timings,
            config={
                "min_speakers": min_spk, "max_speakers": max_spk,
                "lid_min_confidence": settings.langid.min_confidence,
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
