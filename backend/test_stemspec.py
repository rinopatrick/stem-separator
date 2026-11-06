"""T1 red: 9-name stem registry contract."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_registry_has_nine_stems():
    import stemspec

    assert len(stemspec.STEMS) == 9


def test_registry_names():
    import stemspec

    names = [s.name for s in stemspec.STEMS]
    assert names == [
        "vocal",
        "drums",
        "bass",
        "piano",
        "guitar",
        "strings",
        "synth",
        "cue",
        "click",
    ]


def test_registry_fields():
    import stemspec

    by_name = {s.name: s for s in stemspec.STEMS}
    assert by_name["vocal"].stage == 1
    assert by_name["vocal"].source == "mix"
    assert by_name["drums"].kind == "direct"
    assert by_name["piano"].stage == 2
    assert by_name["piano"].source == "other"
    assert by_name["synth"].kind == "leftover"
    assert by_name["cue"].kind == "extracted"
    assert by_name["cue"].source == "vocal"
    assert by_name["click"].kind == "synthesized"
    assert by_name["click"].source == "timestamps"


def test_registry_optionals():
    import stemspec

    by_name = {s.name: s for s in stemspec.STEMS}
    assert by_name["cue"].optional is True
    assert by_name["click"].optional is True
    assert by_name["vocal"].optional is False


def test_stem_names_helper():
    import stemspec

    assert stemspec.STEM_NAMES == [
        "vocal",
        "drums",
        "bass",
        "piano",
        "guitar",
        "strings",
        "synth",
        "cue",
        "click",
    ]
