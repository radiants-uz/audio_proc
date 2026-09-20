"""Схема analysis.json (§7.1 плана). Текстовые поля — на report_lang.
Модель LlmAnalysis — это то, что просим у LLM (structured output),
Analysis — итог с посчитанной кодом статистикой и meta."""


from pydantic import BaseModel, ConfigDict, Field


class Topic(BaseModel):
    name: str
    summary: str
    segment_ids: list[int] = Field(default_factory=list)


class Decision(BaseModel):
    text: str
    speaker: str | None = None
    segment_ids: list[int] = Field(default_factory=list)


class ActionItem(BaseModel):
    text: str
    assignee: str | None = None
    due: str | None = None
    segment_ids: list[int] = Field(default_factory=list)


class KeyQuote(BaseModel):
    segment_id: int
    original: str
    translated: str | None = None


class SpeakerAnalysis(BaseModel):
    id: str
    role_hint: str | None = None
    position: str = ""
    key_quotes: list[KeyQuote] = Field(default_factory=list)


class LlmAnalysis(BaseModel):
    """Ровно то, что генерирует LLM. extra=forbid — чтобы схема для
    response_format была строгой."""

    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    speakers: list[SpeakerAnalysis] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class SpeakerStat(BaseModel):
    id: str
    name: str | None = None
    language: str
    speech_sec: float
    share: float
    segments: int


class LanguageStat(BaseModel):
    language: str
    speech_sec: float
    share: float


class Stats(BaseModel):
    total_speech_sec: float
    speakers_count: int
    segments_count: int
    overlap_segments: int
    avg_segment_sec: float
    by_speaker: list[SpeakerStat]
    by_language: list[LanguageStat]


class AnalysisMeta(BaseModel):
    llm: str
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0
    chunked: bool = False
    chunks: int = 1
    warnings: list[str] = Field(default_factory=list)


class Analysis(LlmAnalysis):
    model_config = ConfigDict(extra="ignore")

    report_lang: str
    stats: Stats
    meta: AnalysisMeta


def llm_json_schema() -> dict:
    """JSON-схема для response_format={"type": "json_schema", ...}."""
    return LlmAnalysis.model_json_schema()
