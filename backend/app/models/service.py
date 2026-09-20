from sqlalchemy import JSON, Boolean, Column, Float, Integer, String

from app.models.base import Base


class Service(Base):
    """LLM-сервис для анализа (GLM-5.3 заказчика через OpenAI-совместимый
    эндпоинт). Заводится в /admin, как «Services» в doc-proc. ASR и
    диаризация — не сервисы, они конфигурируются env."""

    __tablename__ = "services"

    name = Column(String(32), primary_key=True)
    provider = Column(String(32), nullable=False, default="openai_compatible")
    base_url = Column(String(512), nullable=False)
    api_key = Column(String(512), nullable=True)
    model_name = Column(String(128), nullable=False)

    # GLM-5.3: reasoning не отключается, low — минимум. Пусто — не передавать.
    reasoning_effort = Column(String(16), nullable=True, default="low")
    # Строгий response_format=json_schema. На GLM-5.3 через SGLang constrained
    # decoding портит ответ (пустые списки, пропущенные поля) — выключено;
    # по умолчанию схема идёт в промпт + json_object.
    use_json_schema = Column(Boolean, nullable=False, default=False)
    temperature = Column(Float, nullable=True, default=0.2)

    # Семафор параллельных запросов к эндпоинту (воркеры ходят одновременно).
    limit_requests = Column(Integer, nullable=False, default=2)
    # Переопределяют ANALYSIS_MAX_INPUT_TOKENS / ANALYSIS_CHUNK_TOKENS, если заданы.
    max_input_tokens = Column(Integer, nullable=True)
    chunk_tokens = Column(Integer, nullable=True)
    max_output_tokens = Column(Integer, nullable=True)

    # ["uz","ru","en"] — на каких языках модель пригодна для отчёта.
    languages = Column(JSON, nullable=False, default=list)
    # ["analyze","report"]
    operations = Column(JSON, nullable=False, default=list)
    is_active = Column(Boolean, nullable=False, default=True)

    input_token_cost = Column(Float, nullable=True)
    output_token_cost = Column(Float, nullable=True)

    def __str__(self):
        return self.name
