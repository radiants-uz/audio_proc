"""Оценка числа токенов без токенизатора GLM: консервативно, ~3 символа на
токен для кириллицы/узбекской латиницы, чтобы не упереться в контекст."""

CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1
