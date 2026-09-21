"""Tests für jev_mcp.config.

Keine echten Schlüssel, keine Netzaufrufe. Umgebungsvariablen werden
ausschliesslich über monkeypatch gesetzt, der Browser-Test über eine
eingespeiste Prüffunktion.
"""

import dataclasses
import json
import os
import threading
import time
from pathlib import Path

import pytest

from jev_mcp import config

# Erfundene Platzhalter. Sehen wie Schlüssel aus, sind aber keine.
FAKE_TYPESAFE = "ts-fake-0000-typesafe-placeholder"
FAKE_TEXT = "tm-fake-0000-textmodel-placeholder"
FAKE_DEEPSEEK = "dsk-fake-0000-deepseek-placeholder"
FAKE_OPENROUTER = "orr-fake-0000-openrouter-placeholder"
FAKE_MOONSHOT = "msk-fake-0000-moonshot-placeholder"
FAKE_KIMI = "kmi-fake-0000-kimi-placeholder"

# Erfundene Anmeldedaten für die Basis-URL. Lang genug, damit ein Teilstueck
# im Abzug sicher auffallen wuerde.
URL_USER = "urluser-fake-0000-benutzer"
URL_PASSWORD = "urlpass-fake-0000-passwort"
URL_QUERY_SECRET = "urlquery-fake-0000-abfrage"
URL_FRAGMENT = "urlfragment-fake-0000-teil"

ALL_KEY_VARS = (
    "TYPESAFE_API_KEY",
    "TYPESAFE_MODEL",
    "TEXT_MODEL_API_KEY",
    "TEXT_MODEL_BASE_URL",
    "TEXT_MODEL",
    "TEXT_MODEL_REASONING",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Jeder Test startet ohne geerbte Schlüssel aus der echten Shell."""
    for name in ALL_KEY_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def missing_file(tmp_path):
    """Ein Pfad, an dem sicher keine Konfigurationsdatei liegt."""
    return tmp_path / "gibt-es-nicht" / "env"


def browser(daemon=True, chrome=True):
    """Eine eingespeiste Browser-Prüfung ohne Netz und ohne Daemon."""

    def probe():
        return config.BrowserStatus(
            daemon_running=daemon,
            browser_connected=chrome,
            detail="Eingespeiste Prüfung für den Test.",
        )

    return probe


def run(config_path, probe=None):
    return config.diagnose(config_path=config_path, probe_browser=probe or browser())


# --- Auflösung der Schlüssel ------------------------------------------------


def test_no_keys_at_all(missing_file):
    result = run(missing_file)
    assert result.typesafe.present is False
    assert result.typesafe.source is None
    assert result.text_model.present is False
    assert result.text_model.source is None
    assert result.ready is False
    assert len(result.blocked_operations) == 2


def test_only_typesafe(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    result = run(missing_file)
    assert result.typesafe.present is True
    assert result.typesafe.source == config.SOURCE_ENVIRONMENT
    assert result.text_model.present is False
    assert result.ready is True
    assert len(result.blocked_operations) == 1
    assert "Tippen" in result.blocked_operations[0]


def test_typesafe_plus_deepseek(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(missing_file)
    assert result.text_model.present is True
    assert result.text_model.variable == "DEEPSEEK_API_KEY"
    assert result.text_model.provider == "deepseek"
    assert result.text_model.base_url == "https://api.deepseek.com/v1"
    assert result.text_model.model == "deepseek-chat"
    assert result.blocked_operations == ()
    assert result.ready is True


def test_typesafe_plus_openrouter(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER)
    result = run(missing_file)
    assert result.text_model.present is True
    assert result.text_model.variable == "OPENROUTER_API_KEY"
    assert result.text_model.provider == "openrouter"
    assert result.text_model.base_url == "https://openrouter.ai/api/v1"
    assert result.text_model.model == "inception/mercury-2.5"


def test_explicit_text_model_key_beats_deepseek(monkeypatch, missing_file):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER)
    result = run(missing_file)
    assert result.text_model.variable == "TEXT_MODEL_API_KEY"


def test_explicit_base_url_and_model_are_kept(monkeypatch, missing_file):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("TEXT_MODEL", "inception/mercury-2.5")
    result = run(missing_file)
    assert result.text_model.base_url == "https://openrouter.ai/api/v1"
    assert result.text_model.model == "inception/mercury-2.5"
    assert result.text_model.provider == "openrouter"


def test_text_model_key_without_base_url_uses_upstream_default(monkeypatch, missing_file):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    result = run(missing_file)
    assert result.text_model.base_url == "https://api.deepseek.com/v1"
    assert result.text_model.model == "deepseek-chat"


def test_typesafe_plus_kimi(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    result = run(missing_file)
    assert result.text_model.present is True
    assert result.text_model.variable == "MOONSHOT_API_KEY"
    assert result.text_model.provider == "kimi"
    assert result.text_model.base_url == "https://api.moonshot.ai/v1"
    assert result.text_model.model == "kimi-k3"


def test_kimi_beats_deepseek_and_openrouter(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_OPENROUTER)
    result = run(missing_file)
    assert result.text_model.variable == "MOONSHOT_API_KEY"
    assert result.text_model.provider == "kimi"


def test_kimi_api_key_is_used_when_moonshot_is_missing(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("KIMI_API_KEY", FAKE_KIMI)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(missing_file)
    assert result.text_model.variable == "KIMI_API_KEY"
    assert result.text_model.base_url == "https://api.moonshot.ai/v1"
    assert result.text_model.model == "kimi-k3"


def test_moonshot_beats_kimi_variable(monkeypatch, missing_file):
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("KIMI_API_KEY", FAKE_KIMI)
    result = run(missing_file)
    assert result.text_model.variable == "MOONSHOT_API_KEY"


def test_explicit_text_model_key_beats_moonshot(monkeypatch, missing_file):
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    result = run(missing_file)
    assert result.text_model.variable == "TEXT_MODEL_API_KEY"


def test_text_model_overrides_the_kimi_model_without_changing_the_provider(monkeypatch, missing_file):
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("TEXT_MODEL", "kimi-k2.7-code-highspeed")
    result = run(missing_file)
    assert result.text_model.model == "kimi-k2.7-code-highspeed"
    assert result.text_model.base_url == "https://api.moonshot.ai/v1"
    assert result.text_model.provider == "kimi"


def test_kimi_from_the_file_beats_deepseek_from_the_file(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        f"DEEPSEEK_API_KEY={FAKE_DEEPSEEK}\nMOONSHOT_API_KEY={FAKE_MOONSHOT}\n",
        encoding="utf-8",
    )
    result = run(path)
    assert result.text_model.variable == "MOONSHOT_API_KEY"
    assert result.text_model.provider == "kimi"


def test_file_kimi_beats_environment_deepseek(monkeypatch, tmp_path):
    """Befund 4: der Anbieter geht vor der Quelle.

    Wer den Kimi-Schlüssel eigens für dieses Projekt in die Datei legt, will
    Kimi, auch wenn in der Shell noch ein alter DeepSeek-Schlüssel steht.
    """
    path = tmp_path / "env"
    path.write_text(f"MOONSHOT_API_KEY={FAKE_MOONSHOT}\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(path)
    assert result.text_model.variable == "MOONSHOT_API_KEY"
    assert result.text_model.provider == "kimi"
    assert result.text_model.source == config.file_source(path)


def test_environment_deepseek_beats_file_deepseek(monkeypatch, tmp_path):
    """Innerhalb einer Anbieterstufe bleibt die Umgebung vor der Datei."""
    path = tmp_path / "env"
    path.write_text("DEEPSEEK_API_KEY=dsk-fake-aus-der-datei\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(path)
    assert result.text_model.variable == "DEEPSEEK_API_KEY"
    assert result.text_model.source == config.SOURCE_ENVIRONMENT


def test_file_text_model_key_beats_environment_moonshot(monkeypatch, tmp_path):
    """TEXT_MODEL_API_KEY ist die erste Stufe, auch aus der Datei."""
    path = tmp_path / "env"
    path.write_text(f"TEXT_MODEL_API_KEY={FAKE_TEXT}\n", encoding="utf-8")
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    result = run(path)
    assert result.text_model.variable == "TEXT_MODEL_API_KEY"
    assert result.text_model.source == config.file_source(path)


def test_file_openrouter_loses_against_environment_deepseek(monkeypatch, tmp_path):
    """DeepSeek steht in der Rangfolge über OpenRouter, egal aus welcher Quelle."""
    path = tmp_path / "env"
    path.write_text(f"OPENROUTER_API_KEY={FAKE_OPENROUTER}\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(path)
    assert result.text_model.variable == "DEEPSEEK_API_KEY"
    assert result.text_model.source == config.SOURCE_ENVIRONMENT


def test_provider_and_model_appear_in_the_human_readable_output(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    result = run(missing_file)
    assert "kimi" in result.text_model.detail
    assert "kimi-k3" in result.text_model.detail
    assert "kimi" in result.summary and "kimi-k3" in result.summary


# --- Datei als Quelle -------------------------------------------------------


def test_file_is_used_when_environment_is_empty(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        f"# Kommentar\n\nTYPESAFE_API_KEY=\"{FAKE_TYPESAFE}\"\nDEEPSEEK_API_KEY='{FAKE_DEEPSEEK}'\n",
        encoding="utf-8",
    )
    result = run(path)
    assert result.typesafe.present is True
    assert result.typesafe.source == config.file_source(path)
    assert result.text_model.present is True
    assert result.text_model.variable == "DEEPSEEK_API_KEY"
    assert result.text_model.source == config.file_source(path)


def test_environment_beats_file(monkeypatch, tmp_path):
    path = tmp_path / "env"
    path.write_text(f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\n", encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-fake-aus-der-umgebung")
    result = run(path)
    assert result.typesafe.source == config.SOURCE_ENVIRONMENT


def test_environment_text_key_beats_file_deepseek_key(monkeypatch, tmp_path):
    path = tmp_path / "env"
    path.write_text(f"DEEPSEEK_API_KEY={FAKE_DEEPSEEK}\n", encoding="utf-8")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    result = run(path)
    assert result.text_model.variable == "TEXT_MODEL_API_KEY"
    assert result.text_model.source == config.SOURCE_ENVIRONMENT


def test_quotes_and_whitespace_are_stripped(tmp_path):
    path = tmp_path / "env"
    path.write_text(f'  TYPESAFE_API_KEY =  "{FAKE_TYPESAFE}"  \n', encoding="utf-8")
    values, notes = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE
    assert notes == ()


def test_garbage_lines_do_not_crash(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        "das ist kein paar\n"
        "\x00\xff binär müll\n"
        "=ohne name\n"
        f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\n"
        "noch eine kaputte zeile\n",
        encoding="utf-8",
    )
    values, notes = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE
    assert len(notes) == 1
    assert "4" in notes[0]
    result = run(path)
    assert result.typesafe.present is True
    assert any("4" in note for note in result.notes)


def test_unreadable_file_does_not_crash(tmp_path):
    path = tmp_path / "env"
    path.mkdir()  # ein Verzeichnis statt einer Datei
    values, notes = config.read_config_file(path)
    assert values == {}
    assert len(notes) == 1
    result = run(path)
    assert result.typesafe.present is False
    assert result.notes


def test_missing_file_produces_no_note(missing_file):
    values, notes = config.read_config_file(missing_file)
    assert values == {}
    assert notes == ()


# --- Keine Geheimnisse in der Ausgabe ---------------------------------------


def test_no_key_value_appears_in_the_diagnosis(monkeypatch, tmp_path):
    path = tmp_path / "env"
    path.write_text(f"OPENROUTER_API_KEY={FAKE_OPENROUTER}\n", encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    result = run(path)
    haystack = "\n".join(
        [repr(result), str(result), json.dumps(dataclasses.asdict(result), ensure_ascii=False)]
    )
    for secret in (FAKE_TYPESAFE, FAKE_MOONSHOT, FAKE_OPENROUTER):
        assert secret not in haystack
        # Auch kein Teilstück, das laenger als vier Zeichen ist.
        assert secret[-8:] not in haystack
        assert secret[:8] not in haystack


def dump(result):
    """Alles, was ein MCP-Werkzeug oder ein Log von der Diagnose zu sehen bekommt."""
    return "\n".join([repr(result), str(result), json.dumps(dataclasses.asdict(result), ensure_ascii=False)])


def test_no_injected_secret_appears_anywhere_in_the_dump(monkeypatch, tmp_path):
    """Befund 1: der ganze asdict-Abzug wird nach eingeschleusten Geheimnissen durchsucht.

    Geprüft werden Schlüssel aus Umgebung und Datei, Anmeldedaten in der
    Basis-URL und ein Schlüssel als Abfrageparameter.
    """
    path = tmp_path / "env"
    path.write_text(f"OPENROUTER_API_KEY={FAKE_OPENROUTER}\n", encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    monkeypatch.setenv(
        "TEXT_MODEL_BASE_URL",
        f"https://{URL_USER}:{URL_PASSWORD}@proxy.example.com:8443/v1?api-key={URL_QUERY_SECRET}#{URL_FRAGMENT}",
    )
    monkeypatch.setenv("TYPESAFE_MODEL", "jev-latest")
    result = run(path)
    haystack = dump(result)
    for secret in (
        FAKE_TYPESAFE,
        FAKE_TEXT,
        FAKE_OPENROUTER,
        FAKE_MOONSHOT,
        URL_USER,
        URL_PASSWORD,
        URL_QUERY_SECRET,
        URL_FRAGMENT,
    ):
        assert secret not in haystack, f"Geheimnis {secret[:3]}... steht im Abzug"
        assert secret[:8] not in haystack
        assert secret[-8:] not in haystack


def test_credentials_in_the_base_url_are_stripped(monkeypatch, missing_file):
    """Befund 1: nur Schema, Host, Port und Pfad bleiben stehen."""
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    monkeypatch.setenv(
        "TEXT_MODEL_BASE_URL",
        f"https://{URL_USER}:{URL_PASSWORD}@proxy.example.com:8443/v1?api-key={URL_QUERY_SECRET}",
    )
    result = run(missing_file)
    assert result.text_model.base_url == "https://proxy.example.com:8443/v1"


def test_an_unusable_base_url_falls_back_to_the_provider_default(monkeypatch, missing_file):
    """Befund 1: was nicht sauber zerlegbar ist, wird gar nicht erst angezeigt."""
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", f"benutzer:{URL_PASSWORD}@example.com/v1")
    result = run(missing_file)
    assert result.text_model.base_url == "https://api.moonshot.ai/v1"
    assert URL_PASSWORD not in dump(result)
    assert any("unbrauchbar" in note for note in result.notes)


def test_apply_environment_still_passes_the_raw_base_url_on(monkeypatch, missing_file):
    """Der Rohwert darf nach os.environ, nur nicht in eine Rückgabe."""
    raw = f"https://{URL_USER}:{URL_PASSWORD}@proxy.example.com:8443/v1?api-key={URL_QUERY_SECRET}"
    environ = {}
    result = config.apply_environment(
        env={"TEXT_MODEL_API_KEY": FAKE_TEXT, "TEXT_MODEL_BASE_URL": raw},
        config_path=missing_file,
        environ=environ,
    )
    assert result.ok is True
    assert environ["TEXT_MODEL_BASE_URL"] == raw
    assert URL_PASSWORD not in repr(result)
    assert URL_QUERY_SECRET not in repr(result)


# --- diagnose() wirft nie ---------------------------------------------------


class HostileMapping:
    """Eine Umgebung, die bei jedem Zugriff kaputtgeht."""

    def get(self, *args, **kwargs):
        raise RuntimeError("kaputte Umgebung")

    def __getitem__(self, name):
        raise RuntimeError("kaputte Umgebung")


class HostilePath:
    """Ein Pfadobjekt, dessen __fspath__ selbst fliegt."""

    def __fspath__(self):
        raise RuntimeError("kaputter Pfad")

    def __repr__(self):
        return "<kaputter Pfad>"


def test_diagnose_survives_a_hostile_environment(missing_file):
    result = config.diagnose(env=HostileMapping(), config_path=missing_file, probe_browser=browser())
    assert isinstance(result, config.Diagnosis)
    assert result.ready is False
    assert result.notes


def test_diagnose_survives_a_failing_browser_probe(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)

    def explode():
        raise OSError("Daemon-Sockel kaputt")

    result = config.diagnose(config_path=missing_file, probe_browser=explode)
    assert result.browser.daemon_running is False
    assert result.ready is False
    assert result.notes


def test_diagnose_survives_a_broken_config_path(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    result = config.diagnose(config_path=object(), probe_browser=browser())
    assert isinstance(result, config.Diagnosis)
    assert result.notes


def test_diagnose_without_arguments_does_not_raise(monkeypatch, missing_file):
    """Auch der Vorgabeweg mit der Vorgabedatei fliegt nicht.

    Die Browser-Prüfung wird eingespeist. Der echte Weg würde den Sockel des
    Entwicklerrechners anfassen und die Antwort hinge davon ab, ob dort gerade
    ein Daemon läuft.
    """
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", missing_file)
    result = config.diagnose(probe_browser=browser())
    assert isinstance(result, config.Diagnosis)
    assert result.config_file == config.display_path(missing_file)


def test_diagnose_survives_a_path_object_that_explodes(monkeypatch):
    """Befund 11: auch display_path und die Existenzprüfung liegen im Schutz."""
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    result = config.diagnose(config_path=HostilePath(), probe_browser=browser())
    assert isinstance(result, config.Diagnosis)
    assert result.config_file_present is False


# --- Die Sätze --------------------------------------------------------------


def all_sentences(result):
    return [
        *result.blocked_operations,
        result.browser.detail,
        result.typesafe.detail,
        result.text_model.detail,
        result.summary,
        *result.notes,
    ]


@pytest.mark.parametrize(
    ("typesafe", "text", "daemon", "chrome"),
    [
        (False, False, False, False),
        (True, False, True, True),
        (True, True, True, False),
        (True, True, False, False),
        (False, True, True, True),
    ],
)
def test_sentences_are_whole_german_sentences(monkeypatch, missing_file, typesafe, text, daemon, chrome):
    if typesafe:
        monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    if text:
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(missing_file, probe=browser(daemon=daemon, chrome=chrome))
    for sentence in all_sentences(result):
        assert sentence, "leerer Satz"
        assert sentence[0].isupper()
        assert sentence.rstrip().endswith(".")
        assert len(sentence.split()) >= 5
        assert "ß" not in sentence
        assert "–" not in sentence and "—" not in sentence
        assert " - " not in sentence


def test_missing_text_model_sentence_names_what_still_works(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    result = run(missing_file)
    sentence = result.blocked_operations[0]
    assert "Tippen" in sentence
    assert "Textmodell" in sentence
    assert "Klicken" in sentence and "Scrollen" in sentence


def test_missing_typesafe_blocks_everything(missing_file):
    result = run(missing_file)
    assert any("TypeSafe" in s for s in result.blocked_operations)
    assert result.ready is False


def test_daemon_down_blocks_the_run(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(missing_file, probe=browser(daemon=False, chrome=False))
    assert result.ready is False
    assert any("Daemon" in s for s in result.blocked_operations)


def test_daemon_up_without_chrome_blocks_the_run(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(missing_file, probe=browser(daemon=True, chrome=False))
    assert result.ready is False
    assert any("Chrome" in s for s in result.blocked_operations)


# --- Anwenden auf os.environ ------------------------------------------------


def test_apply_environment_sets_the_resolved_values(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\nOPENROUTER_API_KEY={FAKE_OPENROUTER}\n",
        encoding="utf-8",
    )
    environ = {}
    result = config.apply_environment(env={}, config_path=path, environ=environ)
    assert environ["TYPESAFE_API_KEY"] == FAKE_TYPESAFE
    assert environ["TEXT_MODEL_API_KEY"] == FAKE_OPENROUTER
    assert environ["TEXT_MODEL_BASE_URL"] == "https://openrouter.ai/api/v1"
    assert environ["TEXT_MODEL"] == "inception/mercury-2.5"
    assert result.ok is True
    assert set(result.applied) == set(environ)


def test_apply_environment_uses_the_kimi_defaults(missing_file):
    environ = {}
    config.apply_environment(
        env={"MOONSHOT_API_KEY": FAKE_MOONSHOT}, config_path=missing_file, environ=environ
    )
    assert environ["TEXT_MODEL_API_KEY"] == FAKE_MOONSHOT
    assert environ["TEXT_MODEL_BASE_URL"] == "https://api.moonshot.ai/v1"
    assert environ["TEXT_MODEL"] == "kimi-k3"


def test_apply_environment_keeps_an_explicit_text_model_key(tmp_path, missing_file):
    environ = {}
    env = {"TEXT_MODEL_API_KEY": FAKE_TEXT, "DEEPSEEK_API_KEY": FAKE_DEEPSEEK}
    config.apply_environment(env=env, config_path=missing_file, environ=environ)
    assert environ["TEXT_MODEL_API_KEY"] == FAKE_TEXT


def test_apply_environment_without_keys_sets_nothing(missing_file):
    environ = {}
    result = config.apply_environment(env={}, config_path=missing_file, environ=environ)
    assert result.ok is True
    assert result.applied == ()
    assert environ == {}


def test_apply_environment_passes_through_optional_variables(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\nTYPESAFE_MODEL=jev-latest\nTEXT_MODEL_REASONING=none\n",
        encoding="utf-8",
    )
    environ = {}
    config.apply_environment(env={}, config_path=path, environ=environ)
    assert environ["TYPESAFE_MODEL"] == "jev-latest"
    assert environ["TEXT_MODEL_REASONING"] == "none"


def test_apply_environment_never_raises():
    environ = {}
    result = config.apply_environment(env=HostileMapping(), config_path=object(), environ=environ)
    assert result.ok is False
    assert result.applied == ()
    assert result.notes
    assert environ == {}


# --- Befund 2: ganz oder gar nicht ------------------------------------------


class PickyEnviron(dict):
    """Eine Zielumgebung, die genau eine Variable nicht annimmt.

    Bildet den Fall nach, dass `os.environ.__setitem__` einen Wert ablehnt,
    zum Beispiel wegen eines Nullbytes.
    """

    def __init__(self, rejects, **start):
        super().__init__(**start)
        self.rejects = rejects

    def __setitem__(self, name, value):
        if name == self.rejects:
            raise ValueError("diese Variable nimmt die Umgebung nicht an")
        super().__setitem__(name, value)


def test_apply_environment_sets_everything_or_nothing(missing_file):
    """Befund 2: kein halb gesetzter Zustand, kein stilles Verschlucken."""
    environ = PickyEnviron("TEXT_MODEL_BASE_URL")
    result = config.apply_environment(
        env={"TYPESAFE_API_KEY": FAKE_TYPESAFE, "MOONSHOT_API_KEY": FAKE_MOONSHOT},
        config_path=missing_file,
        environ=environ,
    )
    assert result.ok is False
    assert dict(environ) == {}
    assert result.notes
    assert any(note.endswith(".") for note in result.notes)


def test_apply_environment_restores_the_previous_values_on_failure(missing_file):
    """Befund 2: ein Fehler mitten drin lässt nichts Fremdes zurück."""
    environ = PickyEnviron(
        "TEXT_MODEL_BASE_URL",
        TEXT_MODEL_API_KEY="dsk-fake-vorher-vorhanden",
        TYPESAFE_API_KEY="ts-fake-vorher-vorhanden",
    )
    result = config.apply_environment(
        env={"TYPESAFE_API_KEY": FAKE_TYPESAFE, "MOONSHOT_API_KEY": FAKE_MOONSHOT},
        config_path=missing_file,
        environ=environ,
    )
    assert result.ok is False
    assert environ["TEXT_MODEL_API_KEY"] == "dsk-fake-vorher-vorhanden"
    assert environ["TYPESAFE_API_KEY"] == "ts-fake-vorher-vorhanden"


def test_apply_environment_tells_nothing_to_do_apart_from_failure(missing_file):
    """Befund 2: der Rückgabewert unterscheidet die beiden Fälle."""
    nothing = config.apply_environment(env={}, config_path=missing_file, environ={})
    broken = config.apply_environment(
        env={"TYPESAFE_API_KEY": FAKE_TYPESAFE},
        config_path=missing_file,
        environ=PickyEnviron("TYPESAFE_API_KEY"),
    )
    assert nothing.ok is True and nothing.applied == () and nothing.notes == ()
    assert broken.ok is False and broken.applied == () and broken.notes


def test_apply_environment_refuses_a_value_with_a_null_byte(missing_file):
    """Befund 2: erst prüfen, dann setzen."""
    environ = {}
    result = config.apply_environment(
        env={"TYPESAFE_API_KEY": "ts-fake-mit\x00nullbyte"},
        config_path=missing_file,
        environ=environ,
    )
    assert result.ok is False
    assert environ == {}
    assert any("Nullbyte" in note for note in result.notes)
    assert "nullbyte" not in "\n".join(result.notes)


def test_apply_environment_never_puts_a_value_into_a_note(missing_file):
    result = config.apply_environment(
        env={"TYPESAFE_API_KEY": FAKE_TYPESAFE, "MOONSHOT_API_KEY": FAKE_MOONSHOT},
        config_path=missing_file,
        environ=PickyEnviron("TEXT_MODEL_BASE_URL"),
    )
    haystack = repr(result)
    assert FAKE_TYPESAFE not in haystack
    assert FAKE_MOONSHOT not in haystack


# --- Befund 3: Schlüssel und Basis-URL müssen zusammenpassen -----------------


def test_a_stale_base_url_does_not_hijack_the_kimi_key(monkeypatch, tmp_path):
    """Befund 3: Kimi-Schlüssel aus der Datei, DeepSeek-URL aus der alten Shell."""
    path = tmp_path / "env"
    path.write_text(f"MOONSHOT_API_KEY={FAKE_MOONSHOT}\n", encoding="utf-8")
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1")
    result = run(path)
    assert result.text_model.provider == "kimi"
    assert result.text_model.base_url == "https://api.moonshot.ai/v1"
    assert result.text_model.model == "kimi-k3"
    hint = [note for note in result.notes if "deepseek" in note and "MOONSHOT_API_KEY" in note]
    assert hint, result.notes
    assert hint[0].endswith(".")


def test_the_mismatched_base_url_does_not_reach_the_environment(monkeypatch, tmp_path):
    """Befund 3: auch apply_environment darf die falsche Basis-URL nicht setzen."""
    path = tmp_path / "env"
    path.write_text(f"MOONSHOT_API_KEY={FAKE_MOONSHOT}\n", encoding="utf-8")
    environ = {}
    result = config.apply_environment(
        env={"TEXT_MODEL_BASE_URL": "https://api.deepseek.com/v1"},
        config_path=path,
        environ=environ,
    )
    assert result.ok is True
    assert environ["TEXT_MODEL_BASE_URL"] == "https://api.moonshot.ai/v1"
    assert environ["TEXT_MODEL_API_KEY"] == FAKE_MOONSHOT


def test_an_unknown_host_is_kept_but_gets_a_hint(monkeypatch, missing_file):
    """Ein eigener Zwischenserver bleibt erlaubt, bekommt aber einen Hinweis."""
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://proxy.example.com/v1")
    result = run(missing_file)
    assert result.text_model.base_url == "https://proxy.example.com/v1"
    assert result.text_model.provider == "kimi"
    assert any("proxy.example.com" in note for note in result.notes)


def test_an_own_base_url_with_the_neutral_key_stays_untouched(monkeypatch, missing_file):
    """TEXT_MODEL_API_KEY gehört keinem Anbieter, dort wird nichts verworfen."""
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1")
    result = run(missing_file)
    assert result.text_model.base_url == "https://api.deepseek.com/v1"
    assert result.notes == ()


def test_a_model_name_from_another_provider_only_gets_a_hint(monkeypatch, missing_file):
    """Befund 3: beim Modellnamen genügt ein Hinweis, Namen sind frei."""
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("TEXT_MODEL", "deepseek-chat")
    result = run(missing_file)
    assert result.text_model.model == "deepseek-chat"
    assert result.text_model.provider == "kimi"
    hint = [note for note in result.notes if "deepseek-chat" in note]
    assert hint, result.notes


def test_a_fitting_model_name_produces_no_hint(monkeypatch, missing_file):
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("TEXT_MODEL", "kimi-k2.7-code-highspeed")
    result = run(missing_file)
    assert result.notes == ()


# --- Befund 7: Schlüssel aus reinen Leerzeichen ------------------------------


def test_apply_environment_removes_a_whitespace_only_key(missing_file):
    """Befund 7: was die Diagnose als fehlend wertet, muss aus der Umgebung raus."""
    environ = {"TYPESAFE_API_KEY": "   ", "TEXT_MODEL_API_KEY": "  \t "}
    source = dict(environ)
    result = config.apply_environment(env=source, config_path=missing_file, environ=environ)
    assert result.ok is True
    assert "TYPESAFE_API_KEY" not in environ
    assert "TEXT_MODEL_API_KEY" not in environ
    assert set(result.removed) >= {"TYPESAFE_API_KEY", "TEXT_MODEL_API_KEY"}


def test_diagnosis_and_environment_agree_about_a_whitespace_only_key(missing_file):
    environ = {"TYPESAFE_API_KEY": " "}
    result = config.diagnose(env=dict(environ), config_path=missing_file, probe_browser=browser())
    config.apply_environment(env=dict(environ), config_path=missing_file, environ=environ)
    assert result.typesafe.present is False
    assert "TYPESAFE_API_KEY" not in environ


def test_apply_environment_removes_a_whitespace_only_passthrough(missing_file):
    environ = {"TYPESAFE_MODEL": "  "}
    config.apply_environment(env=dict(environ), config_path=missing_file, environ=environ)
    assert "TYPESAFE_MODEL" not in environ


def test_apply_environment_leaves_foreign_variables_alone(missing_file):
    environ = {"PATH": "/usr/bin", "HOME": "/Users/test"}
    config.apply_environment(env={}, config_path=missing_file, environ=environ)
    assert environ == {"PATH": "/usr/bin", "HOME": "/Users/test"}


# --- Befund 5: nur gewöhnliche Dateien, nur bis zur Obergrenze --------------


def test_a_fifo_is_skipped_instead_of_blocking(tmp_path):
    """Befund 5: Path.read_text() würde im open() einer FIFO stehen bleiben."""
    path = tmp_path / "env"
    os.mkfifo(path)
    finished = threading.Event()
    box = {}

    def work():
        box["result"] = config.read_config_file(path)
        finished.set()

    threading.Thread(target=work, daemon=True).start()
    assert finished.wait(5), "read_config_file() haengt an der FIFO"
    values, notes = box["result"]
    assert values == {}
    assert len(notes) == 1
    assert "gewöhnliche Datei" in notes[0]


def test_diagnose_does_not_hang_on_a_fifo(monkeypatch, tmp_path):
    path = tmp_path / "env"
    os.mkfifo(path)
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    finished = threading.Event()
    box = {}

    def work():
        box["result"] = run(path)
        finished.set()

    threading.Thread(target=work, daemon=True).start()
    assert finished.wait(10), "diagnose() haengt an der FIFO"
    assert box["result"].typesafe.present is True
    assert box["result"].notes


def test_an_oversized_file_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "env"
    monkeypatch.setattr(config, "MAX_CONFIG_FILE_BYTES", 64)
    path.write_text("# " + "x" * 200 + "\n", encoding="utf-8")
    values, notes = config.read_config_file(path)
    assert values == {}
    assert len(notes) == 1
    assert "Bytes" in notes[0]


def test_a_file_at_the_limit_is_still_read(tmp_path, monkeypatch):
    path = tmp_path / "env"
    content = f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\n"
    monkeypatch.setattr(config, "MAX_CONFIG_FILE_BYTES", len(content.encode("utf-8")))
    path.write_text(content, encoding="utf-8")
    values, notes = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE
    assert notes == ()


# --- Befund 6: export NAME=WERT ---------------------------------------------


def test_export_prefix_is_understood(tmp_path):
    """Befund 6: die verbreitetste Schreibweise darf nicht verloren gehen."""
    path = tmp_path / "env"
    path.write_text(
        f"export TYPESAFE_API_KEY={FAKE_TYPESAFE}\nexport MOONSHOT_API_KEY={FAKE_MOONSHOT}\n",
        encoding="utf-8",
    )
    values, notes = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE
    assert values["MOONSHOT_API_KEY"] == FAKE_MOONSHOT
    assert notes == ()
    result = run(path)
    assert result.typesafe.present is True
    assert result.text_model.variable == "MOONSHOT_API_KEY"


def test_an_invalid_name_counts_as_skipped(tmp_path):
    """Befund 6: was kein gültiger Variablenname ist, taucht im Hinweis auf."""
    path = tmp_path / "env"
    path.write_text(
        f"2FALSCH=wert\nmit leerzeichen=wert\nTYPESAFE_API_KEY={FAKE_TYPESAFE}\n",
        encoding="utf-8",
    )
    values, notes = config.read_config_file(path)
    assert set(values) == {"TYPESAFE_API_KEY"}
    assert len(notes) == 1
    assert "2" in notes[0]


# --- Befund 9: Kommentar am Zeilenende --------------------------------------


def test_a_trailing_comment_is_removed(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        f"TYPESAFE_API_KEY={FAKE_TYPESAFE} # der Schluessel fuer TypeSafe\n",
        encoding="utf-8",
    )
    values, notes = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE
    assert notes == ()


def test_a_comment_after_a_quoted_value_is_removed(tmp_path):
    path = tmp_path / "env"
    path.write_text(f'TYPESAFE_API_KEY="{FAKE_TYPESAFE}"  # Kommentar\n', encoding="utf-8")
    values, _ = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE


def test_a_hash_inside_a_quoted_value_survives(tmp_path):
    path = tmp_path / "env"
    path.write_text('TYPESAFE_API_KEY="ts-fake-mit # darin"\n', encoding="utf-8")
    values, _ = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == "ts-fake-mit # darin"


def test_a_hash_without_a_space_stays_part_of_the_value(tmp_path):
    path = tmp_path / "env"
    path.write_text("TYPESAFE_API_KEY=ts-fake#teil\n", encoding="utf-8")
    values, _ = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == "ts-fake#teil"


# --- Befund 10: Datei in falscher Kodierung ---------------------------------


def test_a_utf16_file_is_reported_instead_of_being_silently_empty(tmp_path):
    path = tmp_path / "env"
    path.write_bytes(f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\n".encode("utf-16"))
    values, notes = config.read_config_file(path)
    assert values == {}
    assert len(notes) == 1
    result = run(path)
    assert result.typesafe.present is False
    assert result.notes


# --- Befund 12: Einzahl und Mehrzahl ----------------------------------------


def test_a_single_skipped_line_uses_the_singular(tmp_path):
    path = tmp_path / "env"
    path.write_text(f"kaputt\nTYPESAFE_API_KEY={FAKE_TYPESAFE}\n", encoding="utf-8")
    _, notes = config.read_config_file(path)
    assert "wurde 1 Zeile übersprungen" in notes[0]
    assert "entspricht" in notes[0]


def test_two_skipped_lines_use_the_plural(tmp_path):
    path = tmp_path / "env"
    path.write_text("kaputt\nnoch kaputt\n", encoding="utf-8")
    _, notes = config.read_config_file(path)
    assert "wurden 2 Zeilen übersprungen" in notes[0]
    assert "entsprechen" in notes[0]


# --- Befund 13: display_path schneidet auf Pfadebene ------------------------


def test_display_path_only_shortens_at_a_path_boundary(monkeypatch, tmp_path):
    home = tmp_path / "nutzer"
    home.mkdir()
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: home))
    assert config.display_path(home / "geheim") == str(Path("~") / "geheim")
    fremd = tmp_path / "nutzerXYZ" / "geheim"
    assert config.display_path(fremd) == str(fremd)
    assert config.display_path(home) == "~"


def test_display_path_survives_a_broken_object():
    assert isinstance(config.display_path(HostilePath()), str)


# --- Befund 8: Gesamtbudget für die Browser-Prüfung -------------------------


def test_a_slow_browser_probe_is_capped_and_counts_as_unknown(monkeypatch, missing_file):
    """Befund 8: das Budget gilt für den ganzen Vorgang, nicht je Socket."""
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setattr(config, "BROWSER_PROBE_BUDGET_SECONDS", 0.05)

    def tropft():
        time.sleep(30)
        raise AssertionError("haette nie zurueckkommen duerfen")

    started = time.monotonic()
    result = config.diagnose(config_path=missing_file, probe_browser=tropft)
    assert time.monotonic() - started < 5
    assert result.browser.known is False
    assert result.blocked_operations == ()
    assert result.ready is True
    assert result.notes


def test_an_unknown_browser_state_does_not_block_the_run(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)

    def unbekannt():
        return config.BrowserStatus(
            daemon_running=False,
            browser_connected=False,
            detail="Der Zustand ist in diesem Test unbekannt.",
            known=False,
        )

    result = config.diagnose(config_path=missing_file, probe_browser=unbekannt)
    assert result.ready is True
    assert not any("Daemon" in s for s in result.blocked_operations)
    assert any("unbekannt" in note for note in result.notes)


def test_a_busy_daemon_is_not_reported_as_disconnected(monkeypatch):
    """Befund 8: ein Nein erst am Zeitlimit ist keine Auskunft."""
    import browser_harness.admin as admin

    monkeypatch.setattr(admin, "daemon_alive", lambda *a, **k: True)

    def langsam(*args, **kwargs):
        time.sleep(config._DAEMON_SLOW_ANSWER_SECONDS + 0.05)
        return False

    monkeypatch.setattr(admin, "daemon_browser_ready", langsam)
    status = config.probe_browser()
    assert status.known is False
    assert status.daemon_running is True


def test_a_quick_no_still_means_no_chrome(monkeypatch):
    import browser_harness.admin as admin

    monkeypatch.setattr(admin, "daemon_alive", lambda *a, **k: True)
    monkeypatch.setattr(admin, "daemon_browser_ready", lambda *a, **k: False)
    status = config.probe_browser()
    assert status.known is True
    assert status.browser_connected is False


def test_the_browser_probe_docstring_names_the_budget():
    """Befund 8: der Docstring darf nichts versprechen, was nicht stimmt."""
    text = config.probe_browser.__doc__ or ""
    assert "je Socket-Aufruf" in text
    assert "BROWSER_PROBE_BUDGET_SECONDS" in text


# --- browser-harness gehört in die Abhängigkeiten ---------------------------


def test_browser_harness_is_a_declared_dependency():
    root = Path(config.__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "browser-harness" in block


# --- Auch die neuen Hinweise sind ganze deutsche Sätze ----------------------


def scenarios(tmp_path, missing_file):
    """Lagen, die einen Hinweis erzeugen, je als Paar aus Umgebung und Pfad."""
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    kaputt = tmp_path / "kaputt"
    kaputt.write_text("keine zuweisung\nauch nicht\n", encoding="utf-8")
    eine = tmp_path / "eine"
    eine.write_text("keine zuweisung\n", encoding="utf-8")
    moonshot = tmp_path / "moonshot"
    moonshot.write_text(f"MOONSHOT_API_KEY={FAKE_MOONSHOT}\n", encoding="utf-8")
    return [
        ({}, fifo),
        ({}, kaputt),
        ({}, eine),
        ({"TEXT_MODEL_BASE_URL": "https://api.deepseek.com/v1"}, moonshot),
        ({"TEXT_MODEL_BASE_URL": "https://proxy.example.com/v1"}, moonshot),
        ({"TEXT_MODEL_BASE_URL": "kaputt:sehr"}, moonshot),
        ({"TEXT_MODEL": "deepseek-chat"}, moonshot),
        ({}, object()),
    ]


def test_every_note_is_a_whole_german_sentence(tmp_path, missing_file):
    seen = 0
    for env, path in scenarios(tmp_path, missing_file):
        result = config.diagnose(env=env, config_path=path, probe_browser=browser())
        for sentence in all_sentences(result):
            seen += 1
            assert sentence[0].isupper(), sentence
            assert sentence.rstrip().endswith("."), sentence
            assert len(sentence.split()) >= 5, sentence
            assert "ß" not in sentence, sentence
            assert "–" not in sentence and "—" not in sentence, sentence
            assert " - " not in sentence, sentence
        assert result.notes, f"Lage ohne Hinweis: {env} {path}"
    assert seen > 20


def test_the_notes_of_a_failed_application_are_whole_sentences(missing_file):
    result = config.apply_environment(
        env={"TYPESAFE_API_KEY": FAKE_TYPESAFE, "MOONSHOT_API_KEY": FAKE_MOONSHOT},
        config_path=missing_file,
        environ=PickyEnviron("TEXT_MODEL_BASE_URL"),
    )
    assert result.notes
    for sentence in result.notes:
        assert sentence[0].isupper()
        assert sentence.rstrip().endswith(".")
        assert len(sentence.split()) >= 5
        assert "ß" not in sentence
        assert "–" not in sentence and "—" not in sentence
        assert " - " not in sentence
