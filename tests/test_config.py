"""Tests for jev_mcp.config.

No real keys, no network calls. Environment variables are set exclusively
via monkeypatch, and the browser test uses an injected probe function.
"""

import dataclasses
import json
import os
import re
import threading
import time
from pathlib import Path

import pytest

from jev_mcp import config

# Made-up placeholders. They look like keys, but they are not.
FAKE_TYPESAFE = "ts-fake-0000-typesafe-placeholder"
FAKE_TEXT = "tm-fake-0000-textmodel-placeholder"
FAKE_DEEPSEEK = "dsk-fake-0000-deepseek-placeholder"
FAKE_OPENROUTER = "orr-fake-0000-openrouter-placeholder"
FAKE_MOONSHOT = "msk-fake-0000-moonshot-placeholder"
FAKE_KIMI = "kmi-fake-0000-kimi-placeholder"

# Made-up credentials for the base URL. Long enough that any piece of them
# would certainly stand out in the dump.
URL_USER = "urluser-fake-0000-username"
URL_PASSWORD = "urlpass-fake-0000-password"
URL_QUERY_SECRET = "urlquery-fake-0000-query"
URL_FRAGMENT = "urlfragment-fake-0000-part"

# German letters must not appear in a user-facing sentence.
GERMAN_LETTERS = re.compile("[äöüÄÖÜß]")
# German without umlauts: two different German function words in one sentence.
# "die" is left out because it is also an English word, and a single hit is not
# enough because a host name such as das.de or mit.edu may appear in a sentence.
GERMAN_STOPWORDS = re.compile(r"\b(der|das|und|nicht|wurde|ist|ein|eine|mit|oder)\b", re.IGNORECASE)


def looks_german(sentence: str) -> bool:
    """True if the sentence contains two different German function words."""
    return len({word.lower() for word in GERMAN_STOPWORDS.findall(sentence)}) >= 2


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
    """Every test starts without keys inherited from the real shell."""
    for name in ALL_KEY_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def missing_file(tmp_path):
    """A path where there is certainly no configuration file."""
    return tmp_path / "does-not-exist" / "env"


def browser(daemon=True, chrome=True):
    """An injected browser probe without network and without a daemon."""

    def probe():
        return config.BrowserStatus(
            daemon_running=daemon,
            browser_connected=chrome,
            detail="Injected probe for the test.",
        )

    return probe


def run(config_path, probe=None):
    return config.diagnose(config_path=config_path, probe_browser=probe or browser())


# --- Resolving the keys -----------------------------------------------------


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
    assert "Typing" in result.blocked_operations[0]


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
    """Finding 4: the provider comes before the source.

    Whoever puts the Kimi key into the file specifically for this project wants
    Kimi, even if an old DeepSeek key is still set in the shell.
    """
    path = tmp_path / "env"
    path.write_text(f"MOONSHOT_API_KEY={FAKE_MOONSHOT}\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(path)
    assert result.text_model.variable == "MOONSHOT_API_KEY"
    assert result.text_model.provider == "kimi"
    assert result.text_model.source == config.file_source(path)


def test_environment_deepseek_beats_file_deepseek(monkeypatch, tmp_path):
    """Within a provider tier, the environment stays ahead of the file."""
    path = tmp_path / "env"
    path.write_text("DEEPSEEK_API_KEY=dsk-fake-from-the-file\n", encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(path)
    assert result.text_model.variable == "DEEPSEEK_API_KEY"
    assert result.text_model.source == config.SOURCE_ENVIRONMENT


def test_file_text_model_key_beats_environment_moonshot(monkeypatch, tmp_path):
    """TEXT_MODEL_API_KEY is the first tier, also when it comes from the file."""
    path = tmp_path / "env"
    path.write_text(f"TEXT_MODEL_API_KEY={FAKE_TEXT}\n", encoding="utf-8")
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    result = run(path)
    assert result.text_model.variable == "TEXT_MODEL_API_KEY"
    assert result.text_model.source == config.file_source(path)


def test_file_openrouter_loses_against_environment_deepseek(monkeypatch, tmp_path):
    """DeepSeek ranks above OpenRouter, regardless of the source."""
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


# --- File as a source -------------------------------------------------------


def test_file_is_used_when_environment_is_empty(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        f"# comment\n\nTYPESAFE_API_KEY=\"{FAKE_TYPESAFE}\"\nDEEPSEEK_API_KEY='{FAKE_DEEPSEEK}'\n",
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
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-fake-from-the-environment")
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
        "this is not a pair\n"
        "\x00\xff binary junk\n"
        "=without name\n"
        f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\n"
        "another broken line\n",
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
    path.mkdir()  # a directory instead of a file
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


# --- No secrets in the output ----------------------------------------------


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
        # Not even a piece longer than four characters.
        assert secret[-8:] not in haystack
        assert secret[:8] not in haystack


def dump(result):
    """Everything an MCP tool or a log gets to see of the diagnosis."""
    return "\n".join([repr(result), str(result), json.dumps(dataclasses.asdict(result), ensure_ascii=False)])


def test_no_injected_secret_appears_anywhere_in_the_dump(monkeypatch, tmp_path):
    """Finding 1: the whole asdict dump is searched for injected secrets.

    Checked are keys from the environment and the file, credentials in the
    base URL and a key as a query parameter.
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
        assert secret not in haystack, f"Secret {secret[:3]}... appears in the dump"
        assert secret[:8] not in haystack
        assert secret[-8:] not in haystack


def test_credentials_in_the_base_url_are_stripped(monkeypatch, missing_file):
    """Finding 1: only scheme, host, port and path remain."""
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    monkeypatch.setenv(
        "TEXT_MODEL_BASE_URL",
        f"https://{URL_USER}:{URL_PASSWORD}@proxy.example.com:8443/v1?api-key={URL_QUERY_SECRET}",
    )
    result = run(missing_file)
    assert result.text_model.base_url == "https://proxy.example.com:8443/v1"


def test_an_unusable_base_url_falls_back_to_the_provider_default(monkeypatch, missing_file):
    """Finding 1: what cannot be parsed cleanly is not shown at all."""
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", f"user:{URL_PASSWORD}@example.com/v1")
    result = run(missing_file)
    assert result.text_model.base_url == "https://api.moonshot.ai/v1"
    assert URL_PASSWORD not in dump(result)
    assert any("unusable" in note for note in result.notes)


def test_apply_environment_still_passes_the_raw_base_url_on(monkeypatch, missing_file):
    """The raw value may go to os.environ, just not into a return value."""
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


# --- diagnose() never raises ------------------------------------------------


class HostileMapping:
    """An environment that breaks on every access."""

    def get(self, *args, **kwargs):
        raise RuntimeError("broken environment")

    def __getitem__(self, name):
        raise RuntimeError("broken environment")


class HostilePath:
    """A path object whose own __fspath__ raises."""

    def __fspath__(self):
        raise RuntimeError("broken path")

    def __repr__(self):
        return "<broken path>"


def test_diagnose_survives_a_hostile_environment(missing_file):
    result = config.diagnose(env=HostileMapping(), config_path=missing_file, probe_browser=browser())
    assert isinstance(result, config.Diagnosis)
    assert result.ready is False
    assert result.notes


def test_diagnose_survives_a_failing_browser_probe(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)

    def explode():
        raise OSError("daemon socket broken")

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
    """The default path with the default file does not raise either.

    The browser probe is injected. The real path would touch the socket on the
    developer's machine, and the answer would depend on whether a daemon
    happens to be running there.
    """
    monkeypatch.setattr(config, "DEFAULT_CONFIG_PATH", missing_file)
    result = config.diagnose(probe_browser=browser())
    assert isinstance(result, config.Diagnosis)
    assert result.config_file == config.display_path(missing_file)


def test_diagnose_survives_a_path_object_that_explodes(monkeypatch):
    """Finding 11: display_path and the existence check are covered by the guard too."""
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    result = config.diagnose(config_path=HostilePath(), probe_browser=browser())
    assert isinstance(result, config.Diagnosis)
    assert result.config_file_present is False


# --- The sentences ----------------------------------------------------------


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
def test_sentences_are_whole_english_sentences(monkeypatch, missing_file, typesafe, text, daemon, chrome):
    if typesafe:
        monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    if text:
        monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(missing_file, probe=browser(daemon=daemon, chrome=chrome))
    for sentence in all_sentences(result):
        assert sentence, "empty sentence"
        assert sentence[0].isupper()
        assert sentence.rstrip().endswith(".")
        assert len(sentence.split()) >= 5
        assert not GERMAN_LETTERS.search(sentence)
        assert not looks_german(sentence)
        assert "–" not in sentence and "—" not in sentence
        assert " - " not in sentence


def test_missing_text_model_sentence_names_what_still_works(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    result = run(missing_file)
    sentence = result.blocked_operations[0]
    assert "Typing" in sentence
    assert "text model" in sentence
    assert "Clicking" in sentence and "scrolling" in sentence


def test_missing_typesafe_blocks_everything(missing_file):
    result = run(missing_file)
    assert any("TypeSafe" in s for s in result.blocked_operations)
    assert result.ready is False


def test_daemon_down_blocks_the_run(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(missing_file, probe=browser(daemon=False, chrome=False))
    assert result.ready is False
    assert any("daemon" in s for s in result.blocked_operations)


def test_daemon_up_without_chrome_blocks_the_run(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("DEEPSEEK_API_KEY", FAKE_DEEPSEEK)
    result = run(missing_file, probe=browser(daemon=True, chrome=False))
    assert result.ready is False
    assert any("Chrome" in s for s in result.blocked_operations)


# --- Applying to os.environ -------------------------------------------------


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


# --- Finding 2: all or nothing ---------------------------------------------


class PickyEnviron(dict):
    """A target environment that refuses exactly one variable.

    Reproduces the case where `os.environ.__setitem__` rejects a value, for
    example because of a null byte.
    """

    def __init__(self, rejects, **start):
        super().__init__(**start)
        self.rejects = rejects

    def __setitem__(self, name, value):
        if name == self.rejects:
            raise ValueError("the environment does not accept this variable")
        super().__setitem__(name, value)


def test_apply_environment_sets_everything_or_nothing(missing_file):
    """Finding 2: no half-set state, no silent swallowing."""
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
    """Finding 2: a failure halfway through leaves nothing foreign behind."""
    environ = PickyEnviron(
        "TEXT_MODEL_BASE_URL",
        TEXT_MODEL_API_KEY="dsk-fake-present-before",
        TYPESAFE_API_KEY="ts-fake-present-before",
    )
    result = config.apply_environment(
        env={"TYPESAFE_API_KEY": FAKE_TYPESAFE, "MOONSHOT_API_KEY": FAKE_MOONSHOT},
        config_path=missing_file,
        environ=environ,
    )
    assert result.ok is False
    assert environ["TEXT_MODEL_API_KEY"] == "dsk-fake-present-before"
    assert environ["TYPESAFE_API_KEY"] == "ts-fake-present-before"


def test_apply_environment_tells_nothing_to_do_apart_from_failure(missing_file):
    """Finding 2: the return value tells the two cases apart."""
    nothing = config.apply_environment(env={}, config_path=missing_file, environ={})
    broken = config.apply_environment(
        env={"TYPESAFE_API_KEY": FAKE_TYPESAFE},
        config_path=missing_file,
        environ=PickyEnviron("TYPESAFE_API_KEY"),
    )
    assert nothing.ok is True and nothing.applied == () and nothing.notes == ()
    assert broken.ok is False and broken.applied == () and broken.notes


def test_apply_environment_refuses_a_value_with_a_null_byte(missing_file):
    """Finding 2: check first, then set."""
    environ = {}
    result = config.apply_environment(
        env={"TYPESAFE_API_KEY": "ts-fake-with\x00nullbyte"},
        config_path=missing_file,
        environ=environ,
    )
    assert result.ok is False
    assert environ == {}
    assert any("null byte" in note for note in result.notes)
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


# --- Finding 3: key and base URL must match ---------------------------------


def test_a_stale_base_url_does_not_hijack_the_kimi_key(monkeypatch, tmp_path):
    """Finding 3: Kimi key from the file, DeepSeek URL from the old shell."""
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
    """Finding 3: apply_environment must not set the wrong base URL either."""
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
    """A custom proxy server stays allowed, but gets a note."""
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://proxy.example.com/v1")
    result = run(missing_file)
    assert result.text_model.base_url == "https://proxy.example.com/v1"
    assert result.text_model.provider == "kimi"
    assert any("proxy.example.com" in note for note in result.notes)


def test_an_own_base_url_with_the_neutral_key_stays_untouched(monkeypatch, missing_file):
    """TEXT_MODEL_API_KEY belongs to no provider, so nothing is discarded there."""
    monkeypatch.setenv("TEXT_MODEL_API_KEY", FAKE_TEXT)
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1")
    result = run(missing_file)
    assert result.text_model.base_url == "https://api.deepseek.com/v1"
    assert result.notes == ()


def test_a_model_name_from_another_provider_only_gets_a_hint(monkeypatch, missing_file):
    """Finding 3: for the model name a note is enough, names are free."""
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


# --- Finding 7: keys made of whitespace only -------------------------------


def test_apply_environment_removes_a_whitespace_only_key(missing_file):
    """Finding 7: what the diagnosis treats as missing must leave the environment."""
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


# --- Finding 5: only regular files, only up to the limit ---------------------


def test_a_fifo_is_skipped_instead_of_blocking(tmp_path):
    """Finding 5: Path.read_text() would block in the open() of a FIFO."""
    path = tmp_path / "env"
    os.mkfifo(path)
    finished = threading.Event()
    box = {}

    def work():
        box["result"] = config.read_config_file(path)
        finished.set()

    threading.Thread(target=work, daemon=True).start()
    assert finished.wait(5), "read_config_file() hangs on the FIFO"
    values, notes = box["result"]
    assert values == {}
    assert len(notes) == 1
    assert "regular file" in notes[0]


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
    assert finished.wait(10), "diagnose() hangs on the FIFO"
    assert box["result"].typesafe.present is True
    assert box["result"].notes


def test_an_oversized_file_is_skipped(tmp_path, monkeypatch):
    path = tmp_path / "env"
    monkeypatch.setattr(config, "MAX_CONFIG_FILE_BYTES", 64)
    path.write_text("# " + "x" * 200 + "\n", encoding="utf-8")
    values, notes = config.read_config_file(path)
    assert values == {}
    assert len(notes) == 1
    assert "bytes" in notes[0]


def test_a_file_at_the_limit_is_still_read(tmp_path, monkeypatch):
    path = tmp_path / "env"
    content = f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\n"
    monkeypatch.setattr(config, "MAX_CONFIG_FILE_BYTES", len(content.encode("utf-8")))
    path.write_text(content, encoding="utf-8")
    values, notes = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE
    assert notes == ()


# --- Finding 6: export NAME=VALUE -------------------------------------------


def test_export_prefix_is_understood(tmp_path):
    """Finding 6: the most common notation must not get lost."""
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
    """Finding 6: whatever is not a valid variable name shows up in the note."""
    path = tmp_path / "env"
    path.write_text(
        f"2WRONG=value\nwith spaces=value\nTYPESAFE_API_KEY={FAKE_TYPESAFE}\n",
        encoding="utf-8",
    )
    values, notes = config.read_config_file(path)
    assert set(values) == {"TYPESAFE_API_KEY"}
    assert len(notes) == 1
    assert "2" in notes[0]


# --- Finding 9: comment at the end of a line -------------------------------


def test_a_trailing_comment_is_removed(tmp_path):
    path = tmp_path / "env"
    path.write_text(
        f"TYPESAFE_API_KEY={FAKE_TYPESAFE} # the key for TypeSafe\n",
        encoding="utf-8",
    )
    values, notes = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE
    assert notes == ()


def test_a_comment_after_a_quoted_value_is_removed(tmp_path):
    path = tmp_path / "env"
    path.write_text(f'TYPESAFE_API_KEY="{FAKE_TYPESAFE}"  # comment\n', encoding="utf-8")
    values, _ = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == FAKE_TYPESAFE


def test_a_hash_inside_a_quoted_value_survives(tmp_path):
    path = tmp_path / "env"
    path.write_text('TYPESAFE_API_KEY="ts-fake-with # inside"\n', encoding="utf-8")
    values, _ = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == "ts-fake-with # inside"


def test_a_hash_without_a_space_stays_part_of_the_value(tmp_path):
    path = tmp_path / "env"
    path.write_text("TYPESAFE_API_KEY=ts-fake#part\n", encoding="utf-8")
    values, _ = config.read_config_file(path)
    assert values["TYPESAFE_API_KEY"] == "ts-fake#part"


# --- Finding 10: file in the wrong encoding ---------------------------------


def test_a_utf16_file_is_reported_instead_of_being_silently_empty(tmp_path):
    path = tmp_path / "env"
    path.write_bytes(f"TYPESAFE_API_KEY={FAKE_TYPESAFE}\n".encode("utf-16"))
    values, notes = config.read_config_file(path)
    assert values == {}
    assert len(notes) == 1
    result = run(path)
    assert result.typesafe.present is False
    assert result.notes


# --- Finding 12: singular and plural ----------------------------------------


def test_a_single_skipped_line_uses_the_singular(tmp_path):
    path = tmp_path / "env"
    path.write_text(f"broken\nTYPESAFE_API_KEY={FAKE_TYPESAFE}\n", encoding="utf-8")
    _, notes = config.read_config_file(path)
    assert "1 line was skipped" in notes[0]
    assert "does not match" in notes[0]


def test_two_skipped_lines_use_the_plural(tmp_path):
    path = tmp_path / "env"
    path.write_text("broken\nstill broken\n", encoding="utf-8")
    _, notes = config.read_config_file(path)
    assert "2 lines were skipped" in notes[0]
    assert "they do not match" in notes[0]


# --- Finding 13: display_path cuts at the path level -----------------------


def test_display_path_only_shortens_at_a_path_boundary(monkeypatch, tmp_path):
    home = tmp_path / "user"
    home.mkdir()
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: home))
    assert config.display_path(home / "secret") == str(Path("~") / "secret")
    foreign = tmp_path / "userXYZ" / "secret"
    assert config.display_path(foreign) == str(foreign)
    assert config.display_path(home) == "~"


def test_display_path_survives_a_broken_object():
    assert isinstance(config.display_path(HostilePath()), str)


# --- Finding 8: overall budget for the browser probe ------------------------


def test_a_slow_browser_probe_is_capped_and_counts_as_unknown(monkeypatch, missing_file):
    """Finding 8: the budget applies to the whole operation, not per socket."""
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)
    monkeypatch.setenv("MOONSHOT_API_KEY", FAKE_MOONSHOT)
    monkeypatch.setattr(config, "BROWSER_PROBE_BUDGET_SECONDS", 0.05)

    def trickles():
        time.sleep(30)
        raise AssertionError("should never have returned")

    started = time.monotonic()
    result = config.diagnose(config_path=missing_file, probe_browser=trickles)
    assert time.monotonic() - started < 5
    assert result.browser.known is False
    assert result.blocked_operations == ()
    assert result.ready is True
    assert result.notes


def test_an_unknown_browser_state_does_not_block_the_run(monkeypatch, missing_file):
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_TYPESAFE)

    def unknown():
        return config.BrowserStatus(
            daemon_running=False,
            browser_connected=False,
            detail="The state is unknown in this test.",
            known=False,
        )

    result = config.diagnose(config_path=missing_file, probe_browser=unknown)
    assert result.ready is True
    assert not any("daemon" in s for s in result.blocked_operations)
    assert any("unknown" in note for note in result.notes)


def test_a_busy_daemon_is_not_reported_as_disconnected(monkeypatch):
    """Finding 8: a no that arrives only at the time limit is not an answer."""
    import browser_harness.admin as admin

    monkeypatch.setattr(admin, "daemon_alive", lambda *a, **k: True)

    def slow(*args, **kwargs):
        time.sleep(config._DAEMON_SLOW_ANSWER_SECONDS + 0.05)
        return False

    monkeypatch.setattr(admin, "daemon_browser_ready", slow)
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
    """Finding 8: the docstring must not promise anything that is not true."""
    text = config.probe_browser.__doc__ or ""
    assert "per socket call" in text
    assert "BROWSER_PROBE_BUDGET_SECONDS" in text


# --- browser-harness belongs in the dependencies ----------------------------


def test_browser_harness_is_a_declared_dependency():
    root = Path(config.__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "browser-harness" in block


# --- The new notes are whole English sentences too -------------------------


def scenarios(tmp_path, missing_file):
    """Scenarios that produce a note, each as a pair of environment and path."""
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    broken = tmp_path / "broken"
    broken.write_text("no assignment\nnor this\n", encoding="utf-8")
    single = tmp_path / "single"
    single.write_text("no assignment\n", encoding="utf-8")
    moonshot = tmp_path / "moonshot"
    moonshot.write_text(f"MOONSHOT_API_KEY={FAKE_MOONSHOT}\n", encoding="utf-8")
    return [
        ({}, fifo),
        ({}, broken),
        ({}, single),
        ({"TEXT_MODEL_BASE_URL": "https://api.deepseek.com/v1"}, moonshot),
        ({"TEXT_MODEL_BASE_URL": "https://proxy.example.com/v1"}, moonshot),
        ({"TEXT_MODEL_BASE_URL": "broken:very"}, moonshot),
        ({"TEXT_MODEL": "deepseek-chat"}, moonshot),
        ({}, object()),
    ]


def test_every_note_is_a_whole_english_sentence(tmp_path, missing_file):
    seen = 0
    for env, path in scenarios(tmp_path, missing_file):
        result = config.diagnose(env=env, config_path=path, probe_browser=browser())
        for sentence in all_sentences(result):
            seen += 1
            assert sentence[0].isupper(), sentence
            assert sentence.rstrip().endswith("."), sentence
            assert len(sentence.split()) >= 5, sentence
            assert not GERMAN_LETTERS.search(sentence), sentence
            assert not looks_german(sentence), sentence
            assert "–" not in sentence and "—" not in sentence, sentence
            assert " - " not in sentence, sentence
        assert result.notes, f"Scenario without a note: {env} {path}"
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
        assert not GERMAN_LETTERS.search(sentence)
        assert not looks_german(sentence)
        assert "–" not in sentence and "—" not in sentence
        assert " - " not in sentence
