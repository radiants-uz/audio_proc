from sqladmin import ModelView

from app.models.service import Service


class ServiceAdmin(ModelView, model=Service):
    name = "LLM service"
    name_plural = "LLM services"
    icon = "fa-solid fa-brain"
    # name — первичный ключ; без этого sqladmin прячет его из формы создания,
    # и INSERT падает на NOT NULL.
    form_include_pk = True

    column_list = [
        Service.name, Service.provider, Service.model_name, Service.base_url,
        Service.reasoning_effort, Service.limit_requests, Service.is_active,
    ]
    column_details_exclude_list = [Service.api_key]
    form_columns = [
        Service.name, Service.provider, Service.base_url, Service.api_key, Service.model_name,
        Service.reasoning_effort, Service.use_json_schema, Service.temperature,
        Service.limit_requests, Service.max_input_tokens, Service.chunk_tokens, Service.max_output_tokens,
        Service.languages, Service.operations, Service.is_active,
        Service.input_token_cost, Service.output_token_cost,
    ]
    form_args = {
        "name": {"description": "Short id used by ARK as `model`, e.g. glm-5.3 (letters, digits, - _ .)"},
        "base_url": {"description": "OpenAI-compatible base, e.g. https://llm.int.glob.uz/openai/v1"},
        "languages": {"description": 'JSON list, e.g. ["uz","ru","en"]; empty = any'},
        "operations": {"description": 'JSON list, e.g. ["analyze","report"]; empty = any'},
        "reasoning_effort": {"description": "GLM-5.3: low | high | max (low recommended)"},
        "use_json_schema": {"description": "Strict response_format=json_schema. Keep OFF for GLM-5.3/SGLang (breaks the answer); schema goes into the prompt + json_object instead"},
    }
