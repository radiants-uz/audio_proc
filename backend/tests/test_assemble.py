from app.services.audio_processing.assemble import assemble_transcript, rename_speakers
from app.services.audio_processing.langid import LangResult
from app.services.audio_processing.types import Segment


def test_rename_speakers_by_first_appearance():
    segs = [
        Segment(0, "SPK7", 5.0, 8.0, overlap=True, overlap_with=["SPK2"]),
        Segment(1, "SPK2", 0.0, 4.0),
        Segment(2, "SPK7", 9.0, 10.0),
    ]
    mapping = rename_speakers(segs)
    assert mapping == {"SPK2": "SPEAKER_00", "SPK7": "SPEAKER_01"}
    assert segs[0].speaker == "SPEAKER_01" and segs[0].overlap_with == ["SPEAKER_00"]


def test_assemble_transcript_shape():
    segs = [Segment(5, "SPEAKER_01", 5.0, 8.0, text="hi", language="en"),
            Segment(9, "SPEAKER_00", 0.0, 4.0, text="privet", language="ru"),
            Segment(1, "SPEAKER_00", 10.0, 11.0, text="", language="ru")]
    langs = {"SPEAKER_00": LangResult("ru", 0.95), "SPEAKER_01": LangResult("en", 0.8)}
    t = assemble_transcript("job", "a.wav", 12.0, segs, langs, {"asr": "mock"}, {"total": 1.234}, {}, ["x"],
                            title="T", speaker_names={"SPEAKER_00": "Ali"})
    assert [s["id"] for s in t["segments"]] == [0, 1, 2]
    assert [s["start"] for s in t["segments"]] == [0.0, 5.0, 10.0]
    assert t["segments"][2]["text"] == ""  # пустые реплики остаются
    assert t["speakers"][0]["id"] == "SPEAKER_00" and t["speakers"][0]["total_speech_sec"] == 5.0
    assert t["speakers"][0]["name"] == "Ali" and t["speakers"][0]["segments_count"] == 2
    assert t["meta"]["timings_sec"]["total"] == 1.23 and t["meta"]["warnings"] == ["x"]
    assert t["title"] == "T"


def test_assemble_emits_chunks_and_speaker_languages():
    from app.services.audio_processing.types import Chunk

    seg = Segment(0, "SPEAKER_00", 0.0, 10.0, text="а b", language="ru", lang_confidence=0.9)
    seg.chunks = [Chunk(0, 0, 0.0, 6.0, "SPEAKER_00", language="ru", lang_confidence=0.99, lang_source="own", text="а"),
                  Chunk(0, 1, 6.0, 10.0, "SPEAKER_00", language="uz", lang_confidence=1.0, lang_source="asr_text", text="b", warning="text_check:lm_margin")]
    langs = {"SPEAKER_00": LangResult("ru", 0.6, languages=[{"language": "ru", "speech_sec": 6.0, "share": 0.6}, {"language": "uz", "speech_sec": 4.0, "share": 0.4}])}
    t = assemble_transcript("job", "a.wav", 12.0, [seg], langs, {}, {}, {}, [])
    s0 = t["segments"][0]
    assert s0["lang_confidence"] == 0.9 and [c["language"] for c in s0["chunks"]] == ["ru", "uz"]
    assert s0["chunks"][1]["warning"] == "text_check:lm_margin" and "warning" not in s0["chunks"][0]
    assert t["speakers"][0]["languages"][0]["language"] == "ru" and t["meta"]["pipeline_version"] == "1.1.0"


def test_rename_speakers_renames_chunks_too():
    from app.services.audio_processing.types import Chunk

    seg = Segment(0, "SPK3", 0.0, 2.0)
    seg.chunks = [Chunk(0, 0, 0.0, 2.0, "SPK3")]
    rename_speakers([seg])
    assert seg.chunks[0].speaker == "SPEAKER_00"
