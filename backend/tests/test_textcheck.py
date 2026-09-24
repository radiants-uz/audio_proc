import time

import pytest

from app.services.audio_processing.textcheck import TextPlausibility, has_words, normalize


@pytest.fixture(scope="module")
def tc():
    t = TextPlausibility()
    t.load()
    return t


RU_GARBAGE = "Zagruziti zapes Strechi Lelizvanka Ivitish topolochitit."
RU_OK = "Загрузите запись встречи или звонка и выберите, что получить."
UZ_OK = "Uchrashuv yoki qoʻngʻiroq yozuvni yuklang va nima kerakligini tanlang."


def test_normalize():
    assert normalize("Ёж, ёлка! 12 oʻzbek’cha") == "еж елка oʻzbekʻcha"
    assert has_words("a") and not has_words("...")


def test_russian_through_uzbek_model_is_flagged_with_ru_hint(tc):
    r = tc.check(RU_GARBAGE, "uz", 4.8)
    assert r.wrong and "lm_margin" in r.reasons and r.suggest == "ru"


def test_correct_transcripts_pass(tc):
    assert not tc.check(RU_OK, "ru", 5.3).wrong
    assert not tc.check(UZ_OK, "uz", 5.1).wrong


def test_empty_is_flagged(tc):
    r = tc.check("", "ru", 5.0)
    assert r.wrong and r.reasons == ["empty"]


def test_short_answers_skip_lm_rules(tc):
    # 1 слово: только алфавит и плотность; при нормальной плотности проходит
    assert not tc.check("Кстати.", "ru", 0.8).wrong
    assert not tc.check("Ладно.", "ru", 0.6).wrong


def test_low_density_flags_short_hallucination(tc):
    r = tc.check("Toshkent.", "uz", 5.0)  # 9 символов за 5 с — 1.8 симв/с
    assert r.wrong and "low_density" in r.reasons


def test_foreign_script_flags_latin_under_russian(tc):
    r = tc.check("I agree with the plan. We neet to review the timeline.", "ru", 4.0)
    assert r.wrong and "foreign_script" in r.reasons


def test_pick_prefers_the_language_whose_text_fits(tc):
    assert tc.pick({"uz": RU_GARBAGE, "ru": RU_OK}) == "ru"
    assert tc.pick({"uz": UZ_OK, "ru": ""}) == "uz"
    assert tc.pick({"uz": "", "ru": "."}) is None


def test_speed(tc):
    t = time.perf_counter()
    for _ in range(50):
        tc.check(RU_OK, "ru", 5.0)
    assert (time.perf_counter() - t) / 50 < 0.005


def test_digits_or_symbols_only_never_crash_and_pass(tc):
    for text in ("2024.", "100%", "+7 900 123 45 67", "№5", "👍", "12:30"):
        for lang in ("uz", "ru", "en"):
            assert not tc.check(text, lang, 1.5).wrong
    assert tc.pick({"uz": "2024.", "ru": "100%"}) is None


def test_density_rule_only_on_longer_chunks(tc):
    assert not tc.check("Да.", "ru", 1.2).wrong  # односложный ответ в куске 1.2 с — нормально
    assert not tc.check("Ha.", "uz", 1.0).wrong
    assert "low_density" in tc.check("Да.", "ru", 3.0).reasons


def test_pick_keeps_original_unless_candidate_clearly_better(tc):
    # транслитерация проигрывает верному русскому на 3+ бита → замена
    assert tc.pick({"uz": RU_GARBAGE, "ru": RU_OK}, original="uz") == "ru"
    # «Да.» против «Yes.»: модели по-разному щедры, но запаса нет → исходник остаётся
    assert tc.pick({"ru": "Да.", "en": "Yes."}, original="ru") == "ru"
    # без original — просто лучший по поправленному счёту
    assert tc.pick({"uz": RU_GARBAGE, "ru": RU_OK}) == "ru"
