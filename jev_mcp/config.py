"""Resolve, apply and diagnose the prerequisites for a jev-ultrafast run.

Finding from `jev_ultrafast/model.py`, read on 2026-09-20:
The library reads its variables only when called, not at import time. At
module level there is only the httpx client, which touches no variable. Only
in `choose()` is `os.environ["TYPESAFE_API_KEY"]` read, along with
`os.environ.get("TYPESAFE_MODEL", "jev-latest")`. Only in `field_text()` are
`TEXT_MODEL_API_KEY`, `TEXT_MODEL_BASE_URL` (default `https://api.deepseek.com/v1`),
`TEXT_MODEL` (default `deepseek-chat`) and `TEXT_MODEL_REASONING` read.
It follows that setting `os.environ` before the call is enough. Whether
jev_ultrafast was imported before or after does not matter. `apply_environment()`
nevertheless always sets the base URL and the model name as well, because the
library defaults point to DeepSeek. A key from another provider would
otherwise silently run against the wrong endpoint.

Resolution order for the text model key: the provider comes before the
source. `TEXT_MODEL_API_KEY` counts first, then Kimi, then DeepSeek, and
OpenRouter last, and within each tier the environment comes first and the
configuration file second. Whoever puts a key into the file specifically for
this project means that provider, even if an old key from another provider is
still set in the shell. For `TYPESAFE_API_KEY` there is only one variable, so
there it is simply environment before file.

Secrets leave this module only through `apply_environment()`, which writes
them to `os.environ`. No public data structure and no message of this module
contains a key value or any piece of one. This also applies to the base URL:
only a sanitized version without credentials, without query and without
fragment goes out, because both are common hiding places for secrets.
"""

from __future__ import annotations

import os
import re
import stat
import threading
import time
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "jev-mcp" / "env"

SOURCE_ENVIRONMENT = "environment"

TYPESAFE_VARIABLE = "TYPESAFE_API_KEY"

# Larger files are not read. A configuration file of environment variables is
# never that large, anything above this is a mistake.
MAX_CONFIG_FILE_BYTES = 256 * 1024

# Hard overall budget for the browser probe. browser_harness sets its time
# limit per socket call, not for the whole operation, so a separate upper bound
# is needed here.
BROWSER_PROBE_BUDGET_SECONDS = 3.0

# browser_harness allows one second per socket call. If the daemon answers no
# only shortly before that, it was probably its time limit and not a real
# answer, so the threshold sits just below it.
_DAEMON_SLOW_ANSWER_SECONDS = 0.9

# Provider defaults. The base URL and the model name can still be overridden via
# TEXT_MODEL_BASE_URL and TEXT_MODEL without switching the provider.
KIMI = ("https://api.moonshot.ai/v1", "kimi-k3")
DEEPSEEK = ("https://api.deepseek.com/v1", "deepseek-chat")
OPENROUTER = ("https://openrouter.ai/api/v1", "inception/mercury-2.5")

# The library's own default, see the module docstring. Applies when someone
# sets TEXT_MODEL_API_KEY without also setting a base URL and model.
UPSTREAM_DEFAULT = DEEPSEEK


@dataclass(frozen=True)
class _Tier:
    """One tier of the key search, one tier per provider."""

    variables: tuple[str, ...]
    defaults: tuple[str, str]
    provider: str | None


# The first match wins. The tiers are processed from top to bottom, and within
# a tier, for each variable, the environment comes first and then the file.
TEXT_MODEL_TIERS: tuple[_Tier, ...] = (
    _Tier(("TEXT_MODEL_API_KEY",), UPSTREAM_DEFAULT, None),
    _Tier(("MOONSHOT_API_KEY", "KIMI_API_KEY"), KIMI, "kimi"),
    _Tier(("DEEPSEEK_API_KEY",), DEEPSEEK, "deepseek"),
    _Tier(("OPENROUTER_API_KEY",), OPENROUTER, "openrouter"),
)

TEXT_MODEL_VARIABLES: tuple[str, ...] = tuple(name for tier in TEXT_MODEL_TIERS for name in tier.variables)

# Provider name by host of the base URL. This keeps the display correct even
# when someone sets TEXT_MODEL_API_KEY together with their own base URL.
PROVIDERS_BY_HOST = {
    "api.moonshot.ai": "kimi",
    "api.deepseek.com": "deepseek",
    "openrouter.ai": "openrouter",
}

# How a model name can be recognized as belonging to its provider. Used only
# for a note, never for a decision, because model names can be chosen freely.
# OpenRouter is deliberately missing, since its names carry the name of the
# other provider.
PROVIDER_MODEL_MARKERS = {
    "kimi": ("kimi", "moonshot"),
    "deepseek": ("deepseek",),
}

# Non-secret extra variables that are also taken over from the file.
PASSTHROUGH_VARIABLES = ("TYPESAFE_MODEL", "TEXT_MODEL_REASONING")

_SETTINGS_VARIABLES = ("TEXT_MODEL_BASE_URL", "TEXT_MODEL", *PASSTHROUGH_VARIABLES)
_ALL_VARIABLES = (
    TYPESAFE_VARIABLE,
    *TEXT_MODEL_VARIABLES,
    *_SETTINGS_VARIABLES,
)

# Variables that apply_environment() sets or, when there is no value, removes
# from the target environment.
_MANAGED_VARIABLES = (TYPESAFE_VARIABLE, "TEXT_MODEL_API_KEY", *PASSTHROUGH_VARIABLES)

_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_EXPORT_PREFIX = re.compile(r"\Aexport\s+")


@dataclass(frozen=True)
class KeyStatus:
    """Whether a key is present and where it comes from. Never the key itself."""

    present: bool
    source: str | None
    variable: str | None
    detail: str


@dataclass(frozen=True)
class TextModelAccess:
    """The text model access without the key.

    `base_url` is the sanitized version: scheme, host, port and path, nothing
    else. Credentials, query and fragment are removed beforehand.
    """

    present: bool
    source: str | None
    variable: str | None
    provider: str | None
    model: str | None
    base_url: str | None
    detail: str


@dataclass(frozen=True)
class BrowserStatus:
    """State of the browser-harness daemon.

    `known` is false when the probe ran into its time budget. In that case
    `daemon_running` and `browser_connected` say nothing, and the state must
    not block a run.
    """

    daemon_running: bool
    browser_connected: bool
    detail: str
    known: bool = True


@dataclass(frozen=True)
class Diagnosis:
    """Everything that is known about the prerequisites before a run."""

    ready: bool
    typesafe: KeyStatus
    text_model: TextModelAccess
    browser: BrowserStatus
    blocked_operations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    config_file: str = ""
    config_file_present: bool = False
    summary: str = ""


@dataclass(frozen=True)
class EnvironmentApplication:
    """Result of `apply_environment()`. Contains only names, never values.

    `ok` distinguishes a successful run, which may also have had nothing to
    do, from a failed one. If `ok` is false, the environment was not changed
    and `notes` says why.
    """

    ok: bool
    applied: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def display_path(path: object) -> str:
    """Path for humans, with ~ instead of the home directory.

    The comparison happens at the path level, so that a directory which merely
    happens to start with the home directory's name does not get a wrong ~.
    """
    try:
        candidate = Path(path)  # type: ignore[arg-type]
        home = Path.home()
        try:
            relative = candidate.relative_to(home)
        except ValueError:
            return str(candidate)
        return str(Path("~") / relative)
    except Exception:
        # Even a path object whose own __fspath__ raises must not crash a
        # diagnosis.
        try:
            return str(path)
        except Exception:
            return "<unusable path>"


def file_source(path: object) -> str:
    """Source label for values that come from the configuration file."""
    return f"file {display_path(path)}"


def sanitized_url(value: object) -> str | None:
    """The URL without credentials, query and fragment, otherwise None.

    Only this version may go into a return value or a message. The raw value
    can contain a user name, a password or a key in the query.
    """
    if not isinstance(value, str):
        return None
    try:
        parts = urlsplit(value.strip())
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return None
    if not parts.netloc or not host:
        # Without a network location it is not a usable base URL, and whatever
        # is then in the path could contain unchecked credentials.
        return None
    if ":" in host:
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    cleaned = urlunsplit((parts.scheme, host, parts.path, "", ""))
    return cleaned or None


def read_config_file(path: object) -> tuple[dict[str, str], tuple[str, ...]]:
    """Reads one `KEY=VALUE` per line. Never raises, reports problems as notes.

    Comment lines starting with `#` and blank lines are skipped, a leading
    `export ` is removed, quotes around values are removed and a comment at
    the end of a line is cut off. Lines with a name that does not match the
    pattern `[A-Za-z_][A-Za-z0-9_]*` count as skipped, so that a file in the
    wrong encoding gets noticed. Only a regular file up to
    `MAX_CONFIG_FILE_BYTES` is read, so that a FIFO cannot bring the diagnosis
    to a halt. A missing file is not a problem and produces no note, but an
    unreadable or partly broken file does.
    """
    try:
        candidate = Path(path)  # type: ignore[arg-type]
    except Exception:
        return {}, ("The path to the configuration file is unusable, so no file is read.",)
    label = display_path(candidate)
    try:
        info = candidate.stat()
    except FileNotFoundError:
        return {}, ()
    except OSError as exc:
        return {}, (
            f"The configuration file {label} could not be checked ({type(exc).__name__}), so it is not read.",
        )
    if not stat.S_ISREG(info.st_mode):
        return {}, (f"The path {label} is not a regular file, so it is not read.",)
    if info.st_size > MAX_CONFIG_FILE_BYTES:
        return {}, (
            f"The configuration file {label} is {info.st_size} bytes, which is larger than the "
            f"allowed {MAX_CONFIG_FILE_BYTES} bytes, so it is not read.",
        )
    try:
        raw = candidate.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        return {}, (f"The configuration file {label} exists but could not be read ({type(exc).__name__}).",)
    values: dict[str, str] = {}
    skipped = 0
    for raw_line in raw.splitlines():
        line = _EXPORT_PREFIX.sub("", raw_line.strip()).strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            skipped += 1
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if not _NAME_PATTERN.match(name):
            skipped += 1
            continue
        values[name] = _value_of(value)
    notes: tuple[str, ...] = ()
    if skipped == 1:
        notes = (
            f"In the configuration file {label}, 1 line was skipped because it does not match "
            "the pattern NAME=VALUE.",
        )
    elif skipped:
        notes = (
            f"In the configuration file {label}, {skipped} lines were skipped because they do "
            "not match the pattern NAME=VALUE.",
        )
    return values, notes


def _value_of(raw: str) -> str:
    """Unpack the value of a line, dropping quotes and comment."""
    value = raw.strip()
    quote = value[:1]
    if quote in ('"', "'"):
        closing = value.find(quote, 1)
        if closing != -1:
            return value[1:closing]
        return _without_comment(value[1:])
    return _without_comment(value)


def _without_comment(value: str) -> str:
    """Cut off everything from a comment marker that follows whitespace."""
    if value.startswith("#"):
        return ""
    cut = len(value)
    for marker in (" #", "\t#"):
        found = value.find(marker)
        if found != -1:
            cut = min(cut, found)
    return value[:cut].strip()


def _clean(value: object) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        if value:
            return value
    return None


def _snapshot(env: Mapping[str, str] | None) -> dict[str, str]:
    """Pick up only the variables that concern us."""
    source = os.environ if env is None else env
    values: dict[str, str] = {}
    for name in _ALL_VARIABLES:
        value = _clean(source.get(name))
        if value is not None:
            values[name] = value
    return values


@dataclass(frozen=True)
class _Layers:
    """The two places to look, in order, environment first."""

    env_values: dict[str, str] = field(default_factory=dict)
    file_values: dict[str, str] = field(default_factory=dict)
    file_label: str = ""

    def ordered(self) -> tuple[tuple[str, dict[str, str]], ...]:
        return ((SOURCE_ENVIRONMENT, self.env_values), (self.file_label, self.file_values))

    def first(self, name: str) -> tuple[str | None, str | None]:
        for label, values in self.ordered():
            value = _clean(values.get(name))
            if value is not None:
                return value, label
        return None, None


def _layers(env: Mapping[str, str] | None, config_path: object) -> tuple[_Layers, tuple[str, ...]]:
    file_values, notes = read_config_file(config_path)
    return (
        _Layers(
            env_values=_snapshot(env),
            file_values={k: v for k, v in ((k, _clean(v)) for k, v in file_values.items()) if v},
            file_label=file_source(config_path),
        ),
        notes,
    )


def _host_provider(base_url: str) -> str | None:
    """Provider name by host, None if the host is unknown."""
    try:
        host = (urlsplit(base_url).hostname or "").lower()
    except ValueError:
        return None
    return PROVIDERS_BY_HOST.get(host)


def _provider_name(base_url: str) -> str:
    try:
        host = (urlsplit(base_url).hostname or "").lower()
    except ValueError:
        host = ""
    return PROVIDERS_BY_HOST.get(host, host or "unknown")


@dataclass(frozen=True)
class _TextModelPlan:
    """The resolved text model access, internally with key and raw values."""

    secret: str | None = None
    variable: str | None = None
    source: str | None = None
    provider: str | None = None
    raw_base_url: str | None = None
    base_url: str | None = None
    model: str | None = None
    notes: tuple[str, ...] = ()


def _find_text_secret(layers: _Layers) -> tuple[str | None, str | None, str | None, _Tier]:
    """Find the text model key. Provider before source, see the docstring."""
    ordered = layers.ordered()
    for tier in TEXT_MODEL_TIERS:
        for variable in tier.variables:
            for label, values in ordered:
                value = _clean(values.get(variable))
                if value is not None:
                    return value, variable, label, tier
    return None, None, None, TEXT_MODEL_TIERS[0]


def _plan_text_model(layers: _Layers) -> _TextModelPlan:
    secret, variable, label, tier = _find_text_secret(layers)
    if variable is None:
        return _TextModelPlan()

    notes: list[str] = []
    default_url, default_model = tier.defaults
    raw_base_url = default_url
    given, given_label = layers.first("TEXT_MODEL_BASE_URL")
    if given is not None:
        shown = sanitized_url(given)
        if shown is None:
            notes.append(
                f"The base URL from the source {given_label} is unusable, so the provider's "
                f"default {sanitized_url(default_url)} applies."
            )
        else:
            found = _host_provider(shown)
            if tier.provider is not None and found is not None and found != tier.provider:
                notes.append(
                    f"The base URL {shown} from the source {given_label} belongs to the provider "
                    f"{found}, but the key comes from the variable {variable} of the provider "
                    f"{tier.provider}, so the default {sanitized_url(default_url)} applies."
                )
            elif tier.provider is not None and found is None:
                notes.append(
                    f"The base URL {shown} from the source {given_label} belongs to no known "
                    f"provider, but it is still used with the key from the variable {variable}."
                )
                raw_base_url = given
            else:
                raw_base_url = given

    base_url = sanitized_url(raw_base_url)
    provider = tier.provider or (_provider_name(base_url) if base_url else "unknown")
    model = layers.first("TEXT_MODEL")[0] or default_model
    mismatch = _model_mismatch(provider, model)
    if mismatch is not None:
        notes.append(mismatch)
    return _TextModelPlan(
        secret=secret,
        variable=variable,
        source=label,
        provider=provider,
        raw_base_url=raw_base_url,
        base_url=base_url,
        model=model,
        notes=tuple(notes),
    )


def _model_mismatch(provider: str | None, model: str) -> str | None:
    """A note when the model name obviously belongs to another provider."""
    if provider is None or "/" in model:
        return None
    lowered = model.lower()
    own = PROVIDER_MODEL_MARKERS.get(provider, ())
    if any(lowered.startswith(marker) for marker in own):
        return None
    for other, markers in PROVIDER_MODEL_MARKERS.items():
        if other == provider:
            continue
        if any(lowered.startswith(marker) for marker in markers):
            return (
                f"The model name {model} looks like it belongs to the provider {other}, but the "
                f"requests go to the provider {provider}. Please check this combination."
            )
    return None


def _text_model_access(plan: _TextModelPlan) -> TextModelAccess:
    if plan.variable is None:
        return TextModelAccess(
            present=False,
            source=None,
            variable=None,
            provider=None,
            model=None,
            base_url=None,
            detail=("No text model key was found, neither in the environment nor in the configuration file."),
        )
    return TextModelAccess(
        present=True,
        source=plan.source,
        variable=plan.variable,
        provider=plan.provider,
        model=plan.model,
        base_url=plan.base_url,
        detail=(
            f"The text model access comes from the source {plan.source} via the variable "
            f"{plan.variable}, provider {plan.provider}, model {plan.model}."
        ),
    )


def resolve_text_model(
    env: Mapping[str, str] | None = None,
    config_path: object | None = None,
) -> TextModelAccess:
    """Resolve the text model access without handing out the key."""
    layers, _ = _layers(env, DEFAULT_CONFIG_PATH if config_path is None else config_path)
    return _text_model_access(_plan_text_model(layers))


def _typesafe_status(layers: _Layers) -> KeyStatus:
    value, label = layers.first(TYPESAFE_VARIABLE)
    if value is None:
        return KeyStatus(
            present=False,
            source=None,
            variable=None,
            detail=(
                "The TypeSafe key is missing, neither the environment nor the configuration file "
                f"contains {TYPESAFE_VARIABLE}."
            ),
        )
    return KeyStatus(
        present=True,
        source=label,
        variable=TYPESAFE_VARIABLE,
        detail=f"The TypeSafe key was found, source {label}, variable {TYPESAFE_VARIABLE}.",
    )


def resolve_typesafe(
    env: Mapping[str, str] | None = None,
    config_path: object | None = None,
) -> KeyStatus:
    """Resolve the TypeSafe key without handing out the key."""
    layers, _ = _layers(env, DEFAULT_CONFIG_PATH if config_path is None else config_path)
    return _typesafe_status(layers)


def probe_browser() -> BrowserStatus:
    """Probes browser-harness without starting anything.

    Deliberately not `browser_harness.admin.ensure_daemon`: that function
    starts a daemon, starts Chrome if necessary and waits up to sixty seconds
    while doing so. For a diagnosis without side effects that is the wrong
    path. `daemon_alive()` sends a ping, `daemon_browser_ready()` asks the
    running daemon about its browser connection. Both set their time limit of
    one second per socket call, not for the whole operation, so
    `_browser_status()` additionally caps the probe with
    `BROWSER_PROBE_BUDGET_SECONDS`.

    If the no to the question about the browser connection arrives only
    shortly before this time limit, the daemon was probably busy and it is not
    a real answer. This case counts as unknown and does not block a run.
    """
    from browser_harness.admin import daemon_alive, daemon_browser_ready

    if not daemon_alive():
        return BrowserStatus(
            daemon_running=False,
            browser_connected=False,
            detail="The browser-harness daemon is not running and therefore does not answer any ping.",
        )
    started = time.monotonic()
    ready = daemon_browser_ready()
    elapsed = time.monotonic() - started
    if not ready:
        if elapsed >= _DAEMON_SLOW_ANSWER_SECONDS:
            return BrowserStatus(
                daemon_running=True,
                browser_connected=False,
                detail=(
                    "The browser-harness daemon is running but did not answer within its time "
                    "limit, so its browser state is unknown."
                ),
                known=False,
            )
        return BrowserStatus(
            daemon_running=True,
            browser_connected=False,
            detail="The browser-harness daemon is running, but no Chrome is connected to it.",
        )
    return BrowserStatus(
        daemon_running=True,
        browser_connected=True,
        detail="The browser-harness daemon is running and a Chrome instance is connected to it.",
    )


_UNKNOWN_BROWSER_DETAIL = "The state of browser-harness could not be determined."


def _browser_status(probe: Callable[[], BrowserStatus] | None) -> tuple[BrowserStatus, tuple[str, ...]]:
    """Run the browser probe under a hard overall budget.

    The probe runs in its own thread. If the budget runs out, the state is
    unknown. Unknown is not the same as not connected and therefore does not
    block a run.
    """
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            box["status"] = (probe or probe_browser)()
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=worker, name="jev-mcp-browser-probe", daemon=True)
    thread.start()
    thread.join(BROWSER_PROBE_BUDGET_SECONDS)
    if thread.is_alive():
        return (
            BrowserStatus(
                daemon_running=False,
                browser_connected=False,
                detail=("The browser-harness probe took too long, so its state is unknown."),
                known=False,
            ),
            (
                "The browser-harness probe was aborted after "
                f"{BROWSER_PROBE_BUDGET_SECONDS} seconds, so the browser state counts as unknown "
                "and does not block a run.",
            ),
        )
    error = box.get("error")
    if error is not None:
        return (
            BrowserStatus(
                daemon_running=False,
                browser_connected=False,
                detail=_UNKNOWN_BROWSER_DETAIL,
            ),
            (
                "The browser-harness probe failed with an error "
                f"({type(error).__name__}), so the browser counts as unavailable.",
            ),
        )
    status = box.get("status")
    if not isinstance(status, BrowserStatus):
        return (
            BrowserStatus(
                daemon_running=False,
                browser_connected=False,
                detail=_UNKNOWN_BROWSER_DETAIL,
            ),
            ("The browser-harness probe returned an unexpected answer.",),
        )
    if not status.known:
        return status, ("The state of browser-harness is unknown, but the agent attempts the run anyway.",)
    return status, ()


def _blocked_operations(
    typesafe: KeyStatus, text_model: TextModelAccess, browser: BrowserStatus
) -> tuple[str, ...]:
    blocked: list[str] = []
    if not typesafe.present:
        blocked.append(
            "No browser run is possible because the TypeSafe key is missing. Without it the "
            "agent cannot choose a single action, not even clicking or scrolling. Put "
            f"{TYPESAFE_VARIABLE} into the environment or into the file {display_path(DEFAULT_CONFIG_PATH)}."
        )
    if not text_model.present:
        blocked.append(
            "Typing into fields is not possible because no text model key was found. "
            "Clicking, scrolling, navigating and selecting from dropdowns still work. "
            "Forms and search fields stay empty until a text model key is available."
        )
    if not browser.known:
        return tuple(blocked)
    if not browser.daemon_running:
        blocked.append(
            "No browser run is possible because the browser-harness daemon is not running. Start "
            "it, and the agent can then open and operate the page."
        )
    elif not browser.browser_connected:
        blocked.append(
            "No browser run is possible because the daemon is running, but no Chrome is connected "
            "to it. Connect Chrome, and clicking, typing and navigating will work."
        )
    return tuple(blocked)


def _summary(ready: bool, text_model: TextModelAccess, browser: BrowserStatus) -> str:
    if ready and not browser.known:
        return (
            "A run is possible, but the state of the browser could not be determined, so the "
            "agent will try anyway."
        )
    if ready and text_model.present:
        return (
            "All prerequisites are met, the agent can click, type, scroll and navigate. "
            f"Typing uses the provider {text_model.provider} and the model "
            f"{text_model.model}."
        )
    if ready:
        return "A run is possible, but limited, because no text model access was found."
    return "A run is not possible at the moment, the reasons are in the list of blocked operations."


def diagnose(
    env: Mapping[str, str] | None = None,
    config_path: object | None = None,
    probe_browser: Callable[[], BrowserStatus] | None = None,
) -> Diagnosis:
    """Determine the state of all prerequisites. Never raises, under any circumstances.

    Every error ends up as a whole sentence in `notes` instead of propagating.
    The parameter name deliberately shadows the function `probe_browser`,
    because it names exactly that function's role.
    """
    path = DEFAULT_CONFIG_PATH if config_path is None else config_path
    notes: list[str] = []
    config_file = ""
    config_file_present = False
    try:
        config_file = display_path(path)
        try:
            config_file_present = bool(Path(path).is_file())  # type: ignore[arg-type]
        except (TypeError, ValueError, OSError):
            config_file_present = False
        layers, file_notes = _layers(env, path)
        notes.extend(file_notes)
        typesafe = _typesafe_status(layers)
        plan = _plan_text_model(layers)
        notes.extend(plan.notes)
        text_model = _text_model_access(plan)
    except Exception as exc:
        notes.append(
            "The environment and the configuration file could not be evaluated "
            f"({type(exc).__name__}), so it is assumed that no key is present."
        )
        typesafe = KeyStatus(
            present=False,
            source=None,
            variable=None,
            detail="The TypeSafe key could not be determined.",
        )
        text_model = TextModelAccess(
            present=False,
            source=None,
            variable=None,
            provider=None,
            model=None,
            base_url=None,
            detail="The text model access could not be determined.",
        )
    browser, browser_notes = _browser_status(probe_browser)
    notes.extend(browser_notes)
    # An unknown browser state is not a no and therefore does not block.
    browser_ok = not browser.known or (browser.daemon_running and browser.browser_connected)
    ready = bool(typesafe.present and browser_ok)
    return Diagnosis(
        ready=ready,
        typesafe=typesafe,
        text_model=text_model,
        browser=browser,
        blocked_operations=_blocked_operations(typesafe, text_model, browser),
        notes=tuple(notes),
        config_file=config_file,
        config_file_present=config_file_present,
        summary=_summary(ready, text_model, browser),
    )


def _plan_environment(
    env: Mapping[str, str] | None, path: object
) -> tuple[dict[str, str], list[str], tuple[str, ...]]:
    """Prepare all values before even one of them is set."""
    layers, _ = _layers(env, path)
    values: dict[str, str] = {}
    remove: list[str] = []

    typesafe = layers.first(TYPESAFE_VARIABLE)[0]
    if typesafe:
        values[TYPESAFE_VARIABLE] = typesafe

    plan = _plan_text_model(layers)
    if plan.secret and plan.raw_base_url and plan.model:
        values["TEXT_MODEL_API_KEY"] = plan.secret
        values["TEXT_MODEL_BASE_URL"] = plan.raw_base_url
        values["TEXT_MODEL"] = plan.model

    for name in PASSTHROUGH_VARIABLES:
        value = layers.first(name)[0]
        if value:
            values[name] = value

    for name in _MANAGED_VARIABLES:
        if name not in values:
            remove.append(name)
    return values, remove, plan.notes


def _unusable(name: str, value: object) -> str | None:
    """Checks a value before it is set. Never names the value itself."""
    if not isinstance(value, str):
        return f"The value for the variable {name} is not a string and is therefore not set."
    if "\x00" in value:
        return (
            f"The value for the variable {name} contains a null byte and therefore cannot be "
            "written to the environment."
        )
    if "=" in name or not _NAME_PATTERN.match(name):
        return f"The name {name} is not allowed as an environment variable and is not set."
    return None


def apply_environment(
    env: Mapping[str, str] | None = None,
    config_path: object | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> EnvironmentApplication:
    """Writes the resolved values to `os.environ`. Never raises.

    Must run before `jev_ultrafast` is used, because the library reads its
    variables from the environment when called, see the module docstring.

    Everything is first prepared and checked and then set in one go. If
    anything fails along the way, the previous state is restored and `ok` is
    false. A half-set environment, in which for example one provider's key
    meets another provider's base URL, therefore cannot occur. Variables for
    which there is no usable value are removed from the target environment, so
    that the diagnosis and reality say the same thing. Returns only names,
    never values.
    """
    target = os.environ if environ is None else environ
    path = DEFAULT_CONFIG_PATH if config_path is None else config_path
    try:
        values, remove, notes = _plan_environment(env, path)
    except Exception as exc:
        return EnvironmentApplication(
            ok=False,
            notes=(f"The environment could not be prepared ({type(exc).__name__}), so nothing was set.",),
        )

    problems = [problem for name, value in values.items() if (problem := _unusable(name, value))]
    if problems:
        return EnvironmentApplication(ok=False, notes=(*notes, *problems))

    touched = [*values, *remove]
    try:
        before = {name: target[name] for name in touched if name in target}
    except Exception as exc:
        return EnvironmentApplication(
            ok=False,
            notes=(
                *notes,
                "The previous state of the environment could not be saved "
                f"({type(exc).__name__}), so nothing was set.",
            ),
        )

    applied: list[str] = []
    removed: list[str] = []
    try:
        for name, value in values.items():
            target[name] = value
            applied.append(name)
        for name in remove:
            if name in target:
                del target[name]
                removed.append(name)
    except Exception as exc:
        rollback = _rollback(target, touched, before)
        return EnvironmentApplication(
            ok=False,
            notes=(
                *notes,
                "Setting the environment variables failed "
                f"({type(exc).__name__}), so not a single variable was changed.",
                *rollback,
            ),
        )
    return EnvironmentApplication(ok=True, applied=tuple(applied), removed=tuple(removed), notes=notes)


def _rollback(
    target: MutableMapping[str, str], touched: list[str], before: dict[str, str]
) -> tuple[str, ...]:
    """Write the saved state back, as far as possible."""
    failed: list[str] = []
    for name in touched:
        try:
            if name in before:
                target[name] = before[name]
            elif name in target:
                del target[name]
        except Exception:
            failed.append(name)
    if failed:
        return (f"The previous state could not be restored for these variables: {', '.join(failed)}.",)
    return ()
