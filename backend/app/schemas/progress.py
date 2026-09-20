from dataclasses import dataclass

PROGRESS_PREFIX = "task_progress:"
WORKER_PREFIX = "worker:"


@dataclass
class Status:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Stage:
    INGEST = "ingest"
    DIARIZATION = "diarization"
    LANGID = "langid"
    ASR = "asr"
    ASSEMBLE = "assemble"
    ANALYSIS = "analysis"
    REPORT = "report"
