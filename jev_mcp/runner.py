"""One run of the browser agent: bounded, supervised and explained in words.

This layer sits between the MCP tool layer and `jev_ultrafast`. It does four
things the library does not do:

1. It applies the environment before anything runs. `jev_ultrafast/model.py`
   reads its variables from `os.environ` only at call time, and its built-in
   default points to DeepSeek. Without `apply_environment()` a foreign key
   would silently run against the wrong endpoint. If applying reports
   `ok=False`, no run starts, and the notes are in the response.
2. It attaches the domain lock from `guards.py` to both intended moments.
3. It bounds the run twice: in actions and in wall-clock time. The library
   only knows the action budget, and a single step can hang.
4. It translates the terse English exceptions of the library into complete
   sentences that say what happened and what can be done.

The target address before the click: what the library provides
---------------------------------------------------------------
Checked on 2026-09-20 in `jev_ultrafast/snapshot.js` and `browser.py`.

The element table the decision model sees (`page["actions"]`, built in
`snapshot.js` line 61) carries **no** target address. It contains `node`,
`role`, `label`, `kind` and `value`, nothing more.

The target address exists nonetheless, in a different place: `cache.guard()`
in `snapshot.js` lines 47 to 54 creates a tuple of fourteen fields for every
element, and position 12 holds `e.getAttribute('href')`. These tuples come
along as `page["guards"]`, where `browser.py` line 97 uses them for the
staleness check. We read them there as well, see `planned_target_url()`.

The protection therefore applies before the click, but not without gaps. To be
honest:

* Only `a[href]` provides an address. A `<button>`, a form submission or a
  click that only a script turns into a navigation contributes nothing. There
  the domain lock only applies after the load, and in the logged-in profile the
  loaded page already is the damage.
* `href="javascript:..."` does not say where it leads. We do not check it as a
  navigation, otherwise every ordinary button link would stop the run, and we
  record it as a note instead.
* Position 12 is a slot in an unnamed tuple of the library. If the order there
  changes, we read the wrong field. That is why we check two things: the length
  of the tuple, which catches insertions and removals, and the shape of the
  value read, which catches a swap. Right next to the address sits the text
  surrounding the element, and that would otherwise turn into a plausible
  address on the start domain. Both end in a note, never in silently waving it
  through. A contract test also holds length and position against the
  installed `snapshot.js`.

How addresses are read
----------------------
There is exactly one way of reading addresses in the whole project, and it
lives in `guards.py`. This module therefore never resolves on its own but calls
`guards.resolve_url()`. The reason is explained there in detail:
`urllib.parse.urljoin` reads the backslash as an ordinary character, the
browser turns it into a slash, and whoever checks the first reading and
executes the second checks the wrong address.

What the result contains, and what that means for secrets
---------------------------------------------------------
`RunResult` can be turned into JSON via `dataclasses.asdict()` without special
handling. It is built from a fixed list of fields, never from state passed
through from the library. Keys from the environment or from the configuration
are therefore not in it.

Two fields do carry content that the task produced, however, and that is
intentional:

* `RunResult.goals` carries the task text verbatim, exactly as it came in.
* `StepRecord.text` carries every value the agent typed into a field.

`snapshot.js` line 9 excludes fields of type `password`, `file` and `hidden`
from observation, so nothing is typed there. A one-time password, a customer
number or an ID number in an ordinary text field is not covered by that and
ends up in the result.

None of this is masked, deliberately. The typed text is the most important
detail for retracing a run, and the text model does not invent credentials: it
can only type what follows from the task. Whoever does not want a confidential
value in the result does not put it into the task.
"""

import contextlib
import math
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from itertools import count
from urllib.parse import urlsplit

from .config import DEFAULT_CONFIG_PATH, EnvironmentApplication, apply_environment
from .guards import DomainDecision, Moment, Policy, RunGuard, resolve_url, start_run

__all__ = [
    "DEFAULT_MAX_ACTIONS",
    "DEFAULT_READ_TIME_BUDGET_S",
    "DEFAULT_TEXT_LIMIT",
    "DEFAULT_TIME_BUDGET_S",
    "DomainStop",
    "LIBRARY_MAX_ACTIONS",
    "LIBRARY_MAX_MODEL_CALLS",
    "LIBRARY_TEXT_LIMIT",
    "MAX_ELEMENT_OPTIONS",
    "MAX_ELEMENT_TEXT",
    "MAX_READ_ELEMENTS",
    "MAX_TEXT_LIMIT",
    "MAX_TIME_BUDGET_S",
    "MIN_TEXT_LIMIT",
    "PlannedStep",
    "ReadElement",
    "ReadResult",
    "RunResult",
    "RunStatus",
    "StepRecord",
    "own_window",
    "install_own_window",
    "condense",
    "stdout_to_stderr",
    "planned_target_url",
    "read_page",
    "run_task",
    "safe_text",
    "translate_error",
    "wait_until_idle",
]

LIBRARY_MAX_ACTIONS = 60
"""The library's upper limit, `jev_ultrafast.questions.MAX_STEPS`.

The value is fixed here so that this module stays loadable without the browser
harness. A contract test holds it against the installed library.
"""

LIBRARY_MAX_MODEL_CALLS = LIBRARY_MAX_ACTIONS * 2
"""The library's model-call budget, `MAX_STEPS * 2` in `agent.py`."""

DEFAULT_MAX_ACTIONS = 25
"""Default for a run. Deliberately below the upper limit, so that a call does not block for long."""

DEFAULT_TIME_BUDGET_S = 120.0
"""Default for the wall-clock time of a run, in seconds."""

MAX_TIME_BUDGET_S = 900.0
"""Upper limit for the wall-clock time. No caller sensibly waits longer."""

_GUARD_ENTRY_LENGTH = 14
"""Length of the tuple from `cache.guard()` in `snapshot.js`."""

_GUARD_HREF_INDEX = 12
"""Position of the target address in that tuple."""

_MAX_HREF_LENGTH = 2048
"""Longest href that is still plausible. Anything longer is not a link target."""

_NAVIGABLE_SCHEMES = frozenset({"http", "https"})

_SCHEME_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789+.-"

_MAX_STALE_PER_STEP = 5
_MAX_TRANSITIONS_PER_RUN = 5
"""How often a run observes again when it stands on a transitional state."""

_MAX_ERROR_TEXT = 240
_SETTLE_TIME_S = 0.5
"""How long the caller still waits for the thread once the caller itself is done."""

LIBRARY_TEXT_LIMIT = 6000
"""The maximum number of characters of visible text `jev_ultrafast/snapshot.js` observes.

The limit sits there in `words.join('\n').slice(0,6000)` and applies before
`text_limit` has anything to shorten. If the observed text reaches exactly this
length, the page is probably longer, and the result says so. A contract test
holds the number against the installed file.
"""

_RUN_LOCK = threading.Lock()
"""Only one run at a time, see `run_task`."""

_RUN_NUMBER = count(1)

_AFTERRUN_CAP_S = 120.0
"""The longest time the release of the run lock waits for the worker thread.

The lock is only released once the thread is really done, otherwise two runs
work in the same browser after a timeout. The cap makes sure that a thread that
hangs for good does not keep the lock forever.
"""

_STDOUT_LOCK = threading.Lock()
_STDOUT_DEPTH = 0
_STDOUT_ORIGINAL: object = None


@contextlib.contextmanager
def stdout_to_stderr() -> Iterator[None]:
    """Redirects standard output to stderr for as long as anyone holds this latch.

    With stdio, standard output is the protocol channel, and every stray byte in
    it destroys the connection. The latch counts how many currently hold it and
    only restores the output once the last one lets go. Counting is necessary
    because two holders overlap: the tool call and the worker thread that
    outlives the time budget. A plain `redirect_stdout` would be restored in
    the wrong order and would leave `sys.stdout` pointing at stderr in the end.

    The latch is not complete protection. It touches `sys.stdout`, not file
    descriptor 1: an `os.write(1, ...)` or a subprocess bypasses it. That this
    does not end up on the wire either is solely due to the SDK, which points
    descriptor 1 at stderr while it is running.
    """
    global _STDOUT_DEPTH, _STDOUT_ORIGINAL
    target = sys.stderr
    if target is None:  # Only in environments without stderr.
        yield
        return
    with _STDOUT_LOCK:
        if _STDOUT_DEPTH == 0:
            _STDOUT_ORIGINAL = sys.stdout
            sys.stdout = target
        _STDOUT_DEPTH += 1
    try:
        yield
    finally:
        with _STDOUT_LOCK:
            _STDOUT_DEPTH -= 1
            if _STDOUT_DEPTH <= 0:
                _STDOUT_DEPTH = 0
                sys.stdout = _STDOUT_ORIGINAL  # type: ignore[assignment]
                _STDOUT_ORIGINAL = None


def wait_until_idle(timeout: float = 30.0) -> bool:
    """Waits until no run and no read is in progress any more.

    Meant for callers that need to know whether the browser is free again, for
    example a test suite between two cases. Returns `True` if the lock was free
    within the deadline.
    """
    if not _RUN_LOCK.acquire(timeout=timeout):
        return False
    _RUN_LOCK.release()
    return True


class RunStatus(StrEnum):
    """How a run ended."""

    DONE = "done"
    """The agent declared the goal reached."""

    BLOCKED = "blocked"
    """The agent could not get any further by itself."""

    PLANNED = "planned"
    """Dry run: something was planned, but nothing was executed."""

    STOPPED_DOMAIN = "stopped_domain"
    """The domain lock stopped the run."""

    STOPPED_BUDGET = "stopped_budget"
    """An action or model-call budget was used up."""

    STOPPED_TIME = "stopped_time"
    """The time budget was used up."""

    FAILED = "failed"
    """The run failed because of an error."""

    NOT_STARTED = "not_started"
    """No browser was opened, the prerequisites were not met."""


@dataclass(frozen=True)
class StepRecord:
    """An executed step, in readable form."""

    step: int
    action: str
    kind: str
    url: str
    page_changed: bool | None = None
    text: str | None = None
    operation: str | None = None
    target: str | None = None
    confidence: float | None = None
    elapsed_ms: int = 0


@dataclass(frozen=True)
class DomainStop:
    """The domain decision that stopped a run.

    A serializable copy of `guards.DomainDecision`. The addresses in it are the
    guard's already defused short forms.
    """

    verdict: str
    reason: str
    moment: str
    start_domain: str | None = None
    target_domain: str | None = None
    target_url: str = ""
    policy_note: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlannedStep:
    """What the agent would do next. Result of a dry run."""

    choice: str
    action: str
    kind: str
    operation: str | None = None
    target: str | None = None
    confidence: float | None = None
    target_url: str | None = None
    will_type: bool = False


@dataclass(frozen=True)
class RunResult:
    """The result of a run, complete and without secrets."""

    status: RunStatus
    ok: bool
    summary: str
    start_url: str
    url: str
    title: str
    goals: tuple[str, ...] = ()
    steps: tuple[StepRecord, ...] = ()
    duration_ms: int = 0
    actions_used: int = 0
    model_calls: int = 0
    max_actions: int = DEFAULT_MAX_ACTIONS
    time_budget_s: float = DEFAULT_TIME_BUDGET_S
    budget_exhausted: bool = False
    budget_kind: str | None = None
    domain_stop: DomainStop | None = None
    planned: PlannedStep | None = None
    notes: tuple[str, ...] = ()
    error: str | None = None


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

_KEY_LOCATION = f"~{str(DEFAULT_CONFIG_PATH).replace(str(DEFAULT_CONFIG_PATH.home()), '', 1)}"

_TRANSLATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^stopped at the \d+-action demo budget"),
        "The library used up its own action budget and ended the run. "
        "Narrow the goal or split it across several calls.",
    ),
    (
        re.compile(r"^reached the demo's model-call budget"),
        "The library's budget for model calls is used up, so the run ends here. "
        "This happens when the agent discards many steps because the page keeps changing. "
        "Try again with a narrower goal or on a calmer page.",
    ),
    (
        re.compile(r"^this run has stopped\b"),
        "This run had already ended, so no further step can be carried out in it. "
        "Start a new run if there is still something to do.",
    ),
    (
        re.compile(r"^type_text needs text_model_api_key\b"),
        "The next step would have been typing, and the key of the text model is missing for that "
        "(TEXT_MODEL_API_KEY). Clicking, selecting and navigating still work, only forms and "
        f"search fields do not. Store the key in {_KEY_LOCATION} or in the environment, "
        "then the agent can type as well.",
    ),
    (
        re.compile(r"^text helper returned no valid field value\b"),
        "The text model did not return a usable field value, so nothing was typed. "
        "State more precisely in the goal what belongs in the field, and try again.",
    ),
    (
        re.compile(r"^invalid typesafe response\b"),
        "The decision model returned a response that could not be evaluated, "
        "so nothing was executed. Trying again usually helps here.",
    ),
    (
        re.compile(r"^model connection failed\b"),
        "The decision or text model could not be reached, so nothing was executed. "
        "Check the internet connection and the stored keys, then start the run again.",
    ),
    (
        re.compile(r"^model provider returned http \d+"),
        "The model provider answered the call with an error, so nothing was executed. "
        "This is usually an expired key, an empty balance or rate limiting. "
        "Check the account with the provider and then try again.",
    ),
    (
        re.compile(r"^model unavailable$"),
        "The model provider was still unavailable after several attempts, so nothing was executed. "
        "Wait a moment and then start the run again.",
    ),
    (
        re.compile(r"^observe and choose before acting$"),
        "The run tried to act without having a valid decision first. "
        "This is a fault in the run control of this server, not on the page. "
        "Start the run again and report the case if it happens again.",
    ),
    (
        re.compile(r"^supply a task$"),
        "No goal was passed to the agent. Say in words what should happen on the page.",
    ),
    (
        re.compile(r"^invalid observed node$"),
        "The agent wanted to operate an element that could no longer be identified unambiguously, "
        "so nothing was executed. Start the run on the page again.",
    ),
    (
        re.compile(r"^target changed or is covered\b"),
        "The element moved or lies beneath another one, so the click was not carried out. "
        "Close overlays such as cookie banners or chat windows and start the run again.",
    ),
    (
        re.compile(r"^dropdown execution was (?:interrupted|not confirmed)\b"),
        "A dropdown list could not be operated safely, so the state of the field is unclear. "
        "Check in the browser what is selected there now before you repeat the run.",
    ),
    (
        re.compile(r"\brequired daemon \S+ is (?:not running|unhealthy)\b|^daemon-starting\b"),
        "The browser-harness daemon is not running or is not healthy, so no Chrome could be "
        "reached. Start it with `browser-harness daemon start` and try again.",
    ),
    (
        re.compile(r"\bdaemon \S+ didn't come up\b"),
        "The browser-harness daemon did not come up, so no Chrome could be reached. "
        "Look at its log, restart it and then try again.",
    ),
    (
        re.compile(r"^permission-blocked\b|^remote debugging is turned off\b"),
        "Chrome does not allow remote control, so the run could not begin. "
        "Allow it in Chrome under chrome://inspect and confirm the prompt, then start the "
        "run again.",
    ),
    (
        re.compile(r"^devtoolsactiveport not found\b"),
        "Chrome is running without remote control switched on, so the run could not begin. "
        "Switch it on under chrome://inspect and then start the run again.",
    ),
    (
        re.compile(r"^bu_cdp_url=\S* unreachable\b|^cdp ws handshake failed\b"),
        "The connection to Chrome did not respond, so the run was aborted. "
        "Check that Chrome is running and connected to the harness, then start the run again.",
    ),
    (
        re.compile(r"^javascript evaluation failed\b"),
        "The page rejected the observation, usually because it is reloading at that moment. "
        "Wait a moment and then start the run again.",
    ),
    (
        re.compile(r"\btimed out waiting for\b"),
        "A call exceeded its time limit, so nothing further was executed. "
        "Check the network and the browser, then start the run again.",
    ),
)
"""The library's wordings, each one anchored at the start of its sentence.

Anchored, not searched for as a loose substring. Previously the word `chrome`
anywhere in the text decided, and the message "Could not click element 'Go to
Chrome extension'" became "Chrome could not be reached". Likewise,
"Element not found: a[href='/connection-settings']" became a connection
failure. If no pattern matches, the sentence says so honestly instead of
claiming a wrong cause.

A contract test holds every pattern against the installed `jev_ultrafast` and
`browser_harness`. If an update renames a message there, it gets noticed.
"""

_TYPE_TRANSLATIONS: tuple[tuple[type[BaseException], str], ...] = (
    (
        TimeoutError,
        "A call exceeded its time limit, so nothing further was executed. "
        "Check the network and the browser, then start the run again.",
    ),
    (
        ConnectionError,
        "A connection dropped, so the run was aborted. "
        "Check the network and the browser, then start the run again.",
    ),
)
"""Cases that can be recognized more reliably by the type of the exception than by its text."""


def condense(text: str) -> str:
    """A single-line, shortened version of a foreign error text."""
    cleaned = " ".join(str(text).split())
    return cleaned[: _MAX_ERROR_TEXT - 1] + "…" if len(cleaned) > _MAX_ERROR_TEXT else cleaned


def translate_error(error: BaseException) -> str:
    """Translates an exception of the library into a complete sentence.

    The sentence says what happened and what the user can do. If nothing
    matches, the original text comes along, shortened and on one line, so that
    nothing that helps with searching disappears.
    """
    text = " ".join(str(error).split()).lower()
    for pattern, sentence in _TRANSLATIONS:
        if pattern.search(text):
            return sentence
    for exc_type, sentence in _TYPE_TRANSLATIONS:
        if isinstance(error, exc_type):
            return sentence
    return (
        "The run failed at a point that this server cannot classify "
        f"({type(error).__name__}: {condense(str(error) or 'no message')}). "
        "Start the run again and report the case if it happens again."
    )


def _exhausted_budget(state: "_ProgressState", task: "_Task") -> str | None:
    """Says from this module's **own** counters whether a budget was used up.

    Deliberately not from the library's error text. The status of a run must
    not depend on how a foreign message happens to be worded: a harmless rename
    from "demo budget" to "step budget" previously flipped the status from
    `stopped_budget` to `failed` without anything about the run having changed.
    What is counted is what this module keeps track of anyway: the executed
    steps and the decisions of the model.
    """
    if len(state.steps) >= min(task.max_actions, LIBRARY_MAX_ACTIONS):
        return "actions"
    if state.model_calls >= LIBRARY_MAX_MODEL_CALLS:
        return "model_calls"
    return None


def _is_stale_page(error: BaseException) -> bool:
    """True for `jev_ultrafast.browser.StalePage`, without importing the library.

    It is recognized by the class name in the inheritance chain. That keeps this
    module free of an import that pulls in the browser harness, and it lets
    tests use their own double with the same name.
    """
    return any(klass.__name__ == "StalePage" for klass in type(error).__mro__)


# ---------------------------------------------------------------------------
# The target address before the click
# ---------------------------------------------------------------------------


def _scheme_of(address: str) -> str:
    """The scheme of an address, in lower case, or an empty text."""
    head, separator, _ = address.partition(":")
    if not separator or not head:
        return ""
    lowered = head.lower()
    if not lowered[0].isalpha() or any(char not in _SCHEME_CHARS for char in lowered):
        return ""
    return lowered


_NO_TARGET_NOTE = (
    "For at least one click there was no target URL in advance. Only links with an href provide "
    "one, buttons and forms do not. For such clicks the domain lock only applies after the load, and "
    "in the logged-in profile the loaded page already is the damage."
)

_TRANSITION_NOTE = (
    "In between, the browser stood on a transitional state that is not a target of the task, such as "
    "about:blank or a page of the browser itself. That does not stop the run, but no action is taken "
    "there: instead the run observed again and waited for the next proper URL."
)

_STALE_NOTE = (
    "The page changed during a step, so the step was discarded and the page observed again. This is "
    "the intended retry case and not an error."
)

_IMPLAUSIBLE_TARGET_NOTE = (
    "The slot of the target URL held something that cannot be a URL, such as a piece of "
    "page text. The order of the fields in the library has probably changed. The target URL was "
    "therefore not checked, and for this step the domain lock only applies after the load."
)

_UNRESOLVABLE_NOTE = (
    "A target URL could not be resolved to a complete URL, so it was not checked before the "
    "click. For this step the domain lock only applies after the load."
)


def _plausible_href(value: str) -> bool:
    """Says whether a value read can be a target address at all.

    The position of the address in the library's tuple is only secured by the
    length, and a length does not catch a swap. Right next to the address sits
    the text surrounding the element. If the two are swapped, "Sign in now and
    confirm your account" turns into a seemingly harmless address on the start
    domain that would pass without objection.

    The distinction is therefore made by shape: an address carries no
    whitespace, and it is not arbitrarily long. Running text carries both.
    """
    if not value or len(value) > _MAX_HREF_LENGTH:
        return False
    return not any(char.isspace() for char in value)


def planned_target_url(page: Mapping, choice: str) -> tuple[str | None, str | None]:
    """Reads the target address a planned click would head for.

    Returns `(address, note)`. The address is absolute and carries a scheme the
    browser really navigates with, otherwise it is `None`. The note is a
    sentence for the result as soon as there is something to say, otherwise
    `None`.

    `None` without a note means: there is no navigation to check here, for
    example with a text field, a jump within the page or `mailto:`. `None` with
    a note means: there would have been something to check, but the library does
    not provide it. See the module docstring for where the data comes from.
    """
    action = next((a for a in (page.get("actions") or []) if a.get("id") == choice), None)
    if action is None or action.get("kind") != "click":
        return None, None

    entry = (page.get("guards") or {}).get(str(action.get("node")))
    if entry is None:
        return None, _NO_TARGET_NOTE
    if not isinstance(entry, list | tuple) or len(entry) != _GUARD_ENTRY_LENGTH:
        return None, (
            "The library's element table has an unexpected shape, so the target URL could not be "
            "read before the click. For this step the domain lock only applies after the load."
        )

    href = entry[_GUARD_HREF_INDEX]
    if not isinstance(href, str) or not href.strip():
        return None, _NO_TARGET_NOTE

    target = href.strip()
    if not _plausible_href(target):
        return None, _IMPLAUSIBLE_TARGET_NOTE
    if target.startswith("#"):
        return None, None

    url_scheme = _scheme_of(target)
    if url_scheme == "javascript":
        return None, (
            "At least one click led to a script target (javascript:) that does not reveal in advance "
            "where it goes. For such clicks the domain lock only applies after the load."
        )
    if url_scheme and url_scheme not in _NAVIGABLE_SCHEMES:
        return None, None

    if url_scheme:
        resolved: str | None = target
    else:
        resolved = resolve_url(str(page.get("url") or ""), target)
    if resolved is None or urlsplit(resolved).scheme.lower() not in _NAVIGABLE_SCHEMES:
        return None, _UNRESOLVABLE_NOTE
    return resolved, None


# ---------------------------------------------------------------------------
# Progress shared between the thread and the caller
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ProgressState:
    """A snapshot of the progress, safe to read."""

    url: str
    title: str
    steps: tuple[StepRecord, ...]
    model_calls: int
    notes: tuple[str, ...]
    duration_ms: int


def _step_from(entry: Mapping) -> StepRecord:
    """Builds a step from an entry of the history, field by field."""
    return StepRecord(
        step=int(entry.get("step") or 0),
        action=str(entry.get("action") or ""),
        kind=str(entry.get("kind") or ""),
        url=str(entry.get("url") or ""),
        page_changed=entry.get("page_changed"),
        text=entry.get("text"),
        operation=entry.get("operation"),
        target=entry.get("target"),
        confidence=entry.get("confidence"),
        elapsed_ms=int(entry.get("elapsed_ms") or 0),
    )


class _Progress:
    """What has happened so far, under a lock, because two threads look at it."""

    def __init__(self, url: str, notes: Sequence[str]) -> None:
        self._lock = threading.Lock()
        self._url = url
        self._title = ""
        self._steps: tuple[StepRecord, ...] = ()
        self._model_calls = 0
        self._notes: list[str] = list(notes)
        self._started = time.monotonic()

    def adopt(self, snapshot: Mapping) -> None:
        """Reads address, title, history and model calls from a snapshot."""
        page = snapshot.get("page") or {}
        steps = tuple(_step_from(entry) for entry in (snapshot.get("history") or []))
        calls = len(snapshot.get("decisions") or [])
        with self._lock:
            self._url = str(page.get("url") or self._url)
            self._title = str(page.get("title") or self._title)
            self._steps = steps
            self._model_calls = max(self._model_calls, calls)

    def add_note(self, note: str | None) -> None:
        """Records a note, each wording only once."""
        if not note:
            return
        with self._lock:
            if note not in self._notes:
                self._notes.append(note)

    def read(self) -> _ProgressState:
        with self._lock:
            return _ProgressState(
                url=self._url,
                title=self._title,
                steps=self._steps,
                model_calls=self._model_calls,
                notes=tuple(self._notes),
                duration_ms=round((time.monotonic() - self._started) * 1000),
            )


class _Session:
    """Holds the agent and closes it exactly once, no matter from which thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agent: object | None = None
        self._closed = False
        self._close_ok = True
        self._had_agent = False

    @property
    def had_agent(self) -> bool:
        """True as soon as an agent was ever handed over, that is, as soon as a tab was ever open.

        If a run is aborted while `Agent.__init__` is still stuck in the
        library's `ensure_daemon()`, there never was a tab. The result must then
        not claim to have closed one.
        """
        with self._lock:
            return self._had_agent

    def adopt(self, agent: object) -> None:
        with self._lock:
            self._had_agent = True
            if not self._closed:
                self._agent = agent
                return
        _close_agent(agent)

    def close(self) -> bool:
        """Closes the agent and says whether that worked."""
        with self._lock:
            if self._closed:
                return self._close_ok
            self._closed = True
            agent = self._agent
            self._agent = None
        closed_ok = _close_agent(agent)
        with self._lock:
            self._close_ok = closed_ok
        return closed_ok


def _close_agent(agent: object | None) -> bool:
    """Closes an agent, swallows every error while doing so and reports the outcome.

    Nothing may fail any more during cleanup, an error here would overwrite the
    actual result of the run. It is not kept quiet, though: if closing fails,
    the browser tab stays open, and the caller learns about it through the
    return value and, in the end, through a note in the result.
    """
    if agent is None:
        return True
    try:
        agent.close()  # type: ignore[attr-defined]
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Task:
    """The checked settings of a run."""

    start_url: str
    goals: tuple[str, ...]
    max_actions: int
    time_budget_s: float
    dry_run: bool
    agent_factory: Callable[..., object]
    notes: tuple[str, ...] = field(default=())


def _goals(goals: Sequence[str] | str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Turns the goal input into a list of non-empty sentences plus notes.

    An empty goal used to disappear silently, and `["Find the page", ""]`
    became a run with one goal without that being stated anywhere. It is still
    discarded, but now it is said.
    """
    if goals is None:
        return (), ()
    raw = (goals,) if isinstance(goals, str) else tuple(goals)
    kept = tuple(text.strip() for text in raw if isinstance(text, str) and text.strip())
    dropped = len(raw) - len(kept)
    if not dropped:
        return kept, ()
    return kept, (
        f"Of the given goals, {dropped} were empty or not a sentence in words. They were "
        f"dropped, and the run uses the remaining {len(kept)}.",
    )


def _as_number(value: object) -> float | None:
    """Reads a finite number, or `None`.

    `None` also stands for `nan` and for the infinities. They are numbers in the
    sense of `float()`, but not budgets: `nan <= 0` is false, so `nan` would get
    through every lower bound, `Event.wait(nan)` returns immediately, and `nan`
    in the result becomes `NaN` in the JSON, which is not valid JSON. A strict
    caller rejects such a response. `json.loads` produces exactly these values,
    `{"max_actions": Infinity}` is enough for that.

    `True` and `False` are numbers in Python as well, and `float(True)` is a
    plain one. `max_actions: true` therefore silently produced a run with a
    single action, and `time_budget_s: true` a budget of one second that then
    ended as a timeout, without it being stated anywhere why. A boolean is
    therefore not a number here.
    """
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _budget(max_actions: object) -> tuple[int, tuple[str, ...]]:
    """Checks the action budget and caps it at the library's upper limit."""
    number = _as_number(max_actions)
    if number is None:
        return DEFAULT_MAX_ACTIONS, (
            f"The action budget was not a finite number, so the default of "
            f"{DEFAULT_MAX_ACTIONS} actions applies.",
        )
    value = int(number)
    if value > LIBRARY_MAX_ACTIONS:
        return LIBRARY_MAX_ACTIONS, (
            f"The action budget of {value} is above the library's upper limit and was capped at "
            f"{LIBRARY_MAX_ACTIONS} actions.",
        )
    if value < 1:
        return 1, ("The action budget was below one action and was raised to one action.",)
    return value, ()


def _time_budget(time_budget_s: object) -> tuple[float, tuple[str, ...]]:
    """Checks the time budget and keeps it between one second and the upper limit."""
    value = _as_number(time_budget_s)
    if value is None:
        return DEFAULT_TIME_BUDGET_S, (
            f"The time budget was not a finite number, so the default of "
            f"{int(DEFAULT_TIME_BUDGET_S)} seconds applies.",
        )
    if value <= 0:
        return DEFAULT_TIME_BUDGET_S, (
            "The time budget was not positive, so the default of "
            f"{int(DEFAULT_TIME_BUDGET_S)} seconds applies.",
        )
    if value > MAX_TIME_BUDGET_S:
        return MAX_TIME_BUDGET_S, (
            f"The time budget of {value:g} seconds is above the upper limit and was capped at "
            f"{int(MAX_TIME_BUDGET_S)} seconds.",
        )
    return value, ()


def _default_agent(url: str, goals: list[str]) -> object:
    """Builds the real agent. The import stays here, not at module level.

    `jev_ultrafast` pulls in the browser harness on import. The runner should be
    loadable and checkable even when there is no browser around, so the import
    only happens once a run really begins.
    """
    import jev_ultrafast.browser as library_browser
    from jev_ultrafast import Agent

    install_own_window(library_browser)
    return Agent(url, goals)


def own_window(cdp: Callable[..., object]) -> Callable[..., object]:
    """Creates background tabs as a separate window in the background.

    The library opens its tab with `background=True` so that it does not take
    the visible tab away from the user. Chrome 153, however, does not answer a
    single command to a tab created that way, every call runs into the harness
    time limit after five seconds. Measured on 2026-09-21: the same tab opened
    visibly answers immediately, and so does a separate window in the
    background. The latter leaves the user's window untouched, which is why
    exactly this one call is redirected and everything else is passed through
    unchanged. An explicitly set `newWindow` is left as it is.
    """

    def forward(method: str, session_id: str | None = None, **params: object) -> object:
        if method == "Target.createTarget" and params.get("background") and "newWindow" not in params:
            params["newWindow"] = True
        return cdp(method, session_id=session_id, **params)

    forward.__jev_mcp_own_window__ = True  # type: ignore[attr-defined]
    return forward


def install_own_window(module: object) -> None:
    """Hooks `own_window` into the `cdp` of a module exactly once."""
    existing = getattr(module, "cdp", None)
    if existing is None or getattr(existing, "__jev_mcp_own_window__", False):
        return
    module.cdp = own_window(existing)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Building the result
# ---------------------------------------------------------------------------


def _domain_stop(decision: DomainDecision) -> DomainStop:
    return DomainStop(
        verdict=str(decision.verdict.value),
        reason=decision.reason,
        moment=str(decision.moment.value),
        start_domain=decision.start_domain,
        target_domain=decision.target_domain,
        target_url=decision.target_url,
        policy_note=decision.policy_note,
        warnings=tuple(decision.warnings),
    )


def _result(
    task: _Task,
    state: _ProgressState,
    status: RunStatus,
    summary: str,
    *,
    ok: bool = False,
    budget_kind: str | None = None,
    domain_stop: DomainStop | None = None,
    planned: PlannedStep | None = None,
    error: str | None = None,
) -> RunResult:
    return RunResult(
        status=status,
        ok=ok,
        summary=summary,
        start_url=task.start_url,
        url=state.url,
        title=state.title,
        goals=task.goals,
        steps=state.steps,
        duration_ms=state.duration_ms,
        actions_used=len(state.steps),
        model_calls=state.model_calls,
        max_actions=task.max_actions,
        time_budget_s=task.time_budget_s,
        budget_exhausted=budget_kind is not None,
        budget_kind=budget_kind,
        domain_stop=domain_stop,
        planned=planned,
        notes=state.notes,
        error=error,
    )


def _seconds(state: _ProgressState) -> int:
    return max(0, round(state.duration_ms / 1000))


def _without_run(
    task: _Task,
    status: RunStatus,
    summary: str,
    *,
    notes: Sequence[str] = (),
    domain_stop: DomainStop | None = None,
) -> RunResult:
    """A result for a run that never began."""
    state = _ProgressState(
        url=task.start_url,
        title="",
        steps=(),
        model_calls=0,
        notes=tuple(dict.fromkeys((*task.notes, *notes))),
        duration_ms=0,
    )
    return _result(task, state, status, summary, domain_stop=domain_stop)


# ---------------------------------------------------------------------------
# The run itself
# ---------------------------------------------------------------------------


def _dry_run(task: _Task, guard: RunGuard, progress: _Progress, agent: object) -> RunResult:
    """Observes and obtains the first decision without executing it.

    The dry run stands where a confirmation prompt to the human would otherwise
    be. So it must also say what the prompt would say: whether the planned step
    leaves the domain this run is bound to. Otherwise a model reads `ok: true` and then starts the real
    run.
    """
    snapshot = agent.command("predict")  # type: ignore[attr-defined]
    progress.adopt(snapshot)
    decision = snapshot.get("decision") or {}
    choice = str(decision.get("choice") or "")
    page = snapshot.get("page") or {}
    action = next((a for a in (page.get("actions") or []) if a.get("id") == choice), None)
    target, note = planned_target_url(page, choice)
    progress.add_note(note)

    planned = PlannedStep(
        choice=choice,
        action=str((action or {}).get("label") or choice),
        kind=str((action or {}).get("kind") or ""),
        operation=decision.get("operation"),
        target=decision.get("target"),
        confidence=decision.get("confidence"),
        target_url=target,
        will_type=(action or {}).get("kind") == "fill",
    )

    if target is not None:
        before_check = guard.check(target, Moment.BEFORE)
        if not before_check.allowed:
            progress.add_note(before_check.policy_note)
            for warning in before_check.warnings:
                progress.add_note(warning)
            sentence = (
                "The dry run executed nothing, and this step would not be executed in the real run "
                f"either: it would leave the domain this run is bound to. {before_check.reason}"
            )
            return _result(
                task,
                progress.read(),
                RunStatus.STOPPED_DOMAIN,
                sentence,
                planned=planned,
                domain_stop=_domain_stop(before_check),
            )

    state = progress.read()
    if choice == "DONE":
        sentence = "The dry run shows that there is nothing left to do on this page."
    elif choice == "BLOCKED":
        sentence = "The dry run shows that the agent would not get any further here."
    else:
        # The label is page text. A double quote in it would end the quoting early
        # in a sentence the model reads, so it becomes a single quote.
        label = planned.action.replace('"', "'")
        sentence = f'The dry run shows as the next step: {planned.operation or planned.kind} on "{label}"' + (
            f", target {target}." if target else ". There is no target URL for it."
        )
    return _result(task, state, RunStatus.PLANNED, sentence, ok=True, planned=planned)


def _observe_again(
    task: _Task, guard: RunGuard, progress: _Progress, agent: object
) -> tuple[Mapping | None, DomainDecision | None, RunResult | None]:
    """Observes again and immediately checks where the browser stands.

    A reason to observe again is always a reason to check. "The page has
    changed" is exactly the case the domain lock exists for, and an empty
    transitional state is exactly the case in which no action is taken.

    Returns `(snapshot, decision, None)`, or `(None, None, result)` if the run
    ends here.
    """
    try:
        snapshot = agent.snapshot()  # type: ignore[attr-defined]
    except Exception as exc:
        return None, None, _failed(task, progress, exc)
    progress.adopt(snapshot)
    decision = guard.check(progress.read().url, Moment.AFTER)
    if not decision.allowed:
        return None, None, _stopped(task, progress, decision)
    return snapshot, decision, None


def _loop(
    task: _Task,
    guard: RunGuard,
    progress: _Progress,
    agent: object,
    cancel: threading.Event,
) -> RunResult:
    """The actual flow: observe, check, decide, check, act, check.

    A check follows **every** observation, including the two that follow a
    stale page, and including the one inside `predict`. Each of them may have
    found the browser somewhere else, and only the check after the load sees
    that.
    """
    snapshot = agent.snapshot()  # type: ignore[attr-defined]
    progress.adopt(snapshot)

    location_check = guard.check(progress.read().url, Moment.AFTER)
    if not location_check.allowed:
        return _stopped(task, progress, location_check)

    if task.dry_run:
        if not location_check.may_interact:
            progress.add_note(_TRANSITION_NOTE)
        return _dry_run(task, guard, progress, agent)

    stale_count = 0
    transitions = 0
    while not cancel.is_set():
        state = progress.read()
        if str(snapshot.get("status") or "") in {"done", "blocked"}:
            return _ended(task, state, str(snapshot["status"]))
        if len(state.steps) >= task.max_actions:
            return _budget_used_up(task, state)

        if not location_check.may_interact:
            # NEUTRAL is explicitly not a permission to act, and that does not
            # only mean empty pages: chrome:// and devtools:// are operable
            # interfaces. So there is no click here, the page is observed again.
            progress.add_note(_TRANSITION_NOTE)
            transitions += 1
            if transitions > _MAX_TRANSITIONS_PER_RUN:
                return _transitions_used_up(task, progress.read())
            snapshot, location_check, end_result = _observe_again(task, guard, progress, agent)
            if end_result is not None:
                return end_result
            continue

        try:
            snapshot = agent.command("predict")  # type: ignore[attr-defined]
        except Exception as exc:
            if _is_stale_page(exc) and stale_count < _MAX_STALE_PER_STEP:
                stale_count += 1
                progress.add_note(_STALE_NOTE)
                snapshot, location_check, end_result = _observe_again(task, guard, progress, agent)
                if end_result is not None:
                    return end_result
                continue
            return _failed(task, progress, exc)
        progress.adopt(snapshot)

        # `predict` observes the page again inside the library. Otherwise a
        # typing step, a selection or a click on a button without href would run
        # on a page that is meanwhile somewhere else.
        location_check = guard.check(progress.read().url, Moment.AFTER)
        if not location_check.allowed:
            return _stopped(task, progress, location_check)
        if not location_check.may_interact:
            continue

        page = snapshot.get("page") or {}
        choice = str((snapshot.get("decision") or {}).get("choice") or "")
        target, note = planned_target_url(page, choice)
        progress.add_note(note)
        if target is not None:
            before_check = guard.check(target, Moment.BEFORE)
            if not before_check.allowed:
                return _stopped(task, progress, before_check)

        if cancel.is_set():
            # The time budget ran out between observing and acting. A run whose
            # time is up does not act any more, and it does not rely on the
            # browser rejecting the call anyway.
            break

        try:
            snapshot = agent.command("act", {"fingerprint": page.get("fingerprint")})  # type: ignore[attr-defined]
        except Exception as exc:
            if _is_stale_page(exc) and stale_count < _MAX_STALE_PER_STEP:
                stale_count += 1
                progress.add_note(_STALE_NOTE)
                snapshot, location_check, end_result = _observe_again(task, guard, progress, agent)
                if end_result is not None:
                    return end_result
                continue
            return _failed(task, progress, exc)
        progress.adopt(snapshot)
        stale_count = 0

        location_check = guard.check(progress.read().url, Moment.AFTER)
        if not location_check.allowed:
            return _stopped(task, progress, location_check)

    return _time_used_up(task, progress.read())


def _ended(task: _Task, state: _ProgressState, status: str) -> RunResult:
    if status == "done":
        sentence = (
            f"The agent reached the goal: {len(state.steps)} actions in {_seconds(state)} seconds, "
            f"last on {state.url}."
        )
        return _result(task, state, RunStatus.DONE, sentence, ok=True)
    sentence = (
        f"The agent could not get any further and ended the run itself after {len(state.steps)} "
        f"actions, last on {state.url}."
    )
    return _result(task, state, RunStatus.BLOCKED, sentence)


def _budget_used_up(task: _Task, state: _ProgressState) -> RunResult:
    sentence = (
        f"The run was stopped at the action budget of {task.max_actions} actions, last on "
        f"{state.url}. Raise the budget or split the goal into smaller tasks."
    )
    return _result(task, state, RunStatus.STOPPED_BUDGET, sentence, budget_kind="actions")


def _transitions_used_up(task: _Task, state: _ProgressState) -> RunResult:
    sentence = (
        f"The browser stayed on a state on which no action is taken, last on "
        f"{state.url}. After {_MAX_TRANSITIONS_PER_RUN} new observations the run gave up "
        "instead of clicking there. Check in the browser what the page is doing right now, and "
        "then start the run again."
    )
    return _result(task, state, RunStatus.BLOCKED, sentence)


TAB_CLOSED = "The browser tab was closed."
TAB_OPEN = "The browser tab could not be closed and is probably still open."
TAB_NEVER_OPEN = (
    "No browser tab was open yet that could have been closed: the run was still being set up. "
    "If it does open one in the background after all, that tab is closed again immediately."
)


def _tab_sentence(session: "_Session", closed_ok: bool) -> str:
    """Says about the browser tab only what is really true."""
    if not session.had_agent:
        return TAB_NEVER_OPEN
    return TAB_CLOSED if closed_ok else TAB_OPEN


def _time_used_up(task: _Task, state: _ProgressState, *, tab_note: str = TAB_CLOSED) -> RunResult:
    sentence = (
        f"The run was aborted after the time budget of {task.time_budget_s:g} seconds, last "
        f"on {state.url}, after {len(state.steps)} actions. {tab_note}"
    )
    return _result(task, state, RunStatus.STOPPED_TIME, sentence, budget_kind="time")


def _without_foreign_page(state: _ProgressState, decision: DomainDecision) -> _ProgressState:
    """Removes everything from the progress that comes from the foreign page.

    When stopping after the load, `_observe_again()` has already filled the
    progress with the foreign page, and it did so before the check. Title and
    address of the foreign page would then be in the result raw and unbounded,
    although the tool description promises that nothing comes back from there.
    A title is arbitrary text: an instruction to the model, control characters,
    a direction override, any length.

    The title is therefore dropped entirely, and the address shown is the
    already defused short form from the decision, the same one that is in
    `domain_stop.target_url`. The last step carries the same address if it
    ended on it.
    """
    foreign_url = state.url
    steps = state.steps
    if steps and steps[-1].url == foreign_url:
        steps = (*steps[:-1], replace(steps[-1], url=decision.target_url))
    return replace(state, url=decision.target_url, title="", steps=steps)


def _stopped(task: _Task, progress: _Progress, decision: DomainDecision) -> RunResult:
    progress.add_note(decision.policy_note)
    for warning in decision.warnings:
        progress.add_note(warning)
    state = progress.read()
    if decision.moment is Moment.AFTER:
        # Before the click the browser still stands on the permitted page, there
        # is nothing to defuse there. After the load it stands on the foreign one.
        state = _without_foreign_page(state, decision)
    when = "before the click" if decision.moment is Moment.BEFORE else "after the load"
    sentence = f"The run was stopped {when} by the domain lock. {decision.reason}"
    return _result(task, state, RunStatus.STOPPED_DOMAIN, sentence, domain_stop=_domain_stop(decision))


def _failed(task: _Task, progress: _Progress, exc: BaseException) -> RunResult:
    state = progress.read()
    sentence = translate_error(exc)
    budget_kind = _exhausted_budget(state, task)
    status = RunStatus.STOPPED_BUDGET if budget_kind else RunStatus.FAILED
    return _result(task, state, status, sentence, budget_kind=budget_kind, error=sentence)


_NO_RESULT_NOTE = (
    "The run ended without leaving a result. Its own error branch failed as well, so the cause is "
    "only in the log on stderr. It was not a timeout. Start the run again and report the case if it "
    "happens again."
)

DOMAIN_LOCK_OFF_NOTE = (
    "The domain lock is disabled in the policy file. This operation therefore does not check whether "
    "the page switches to a foreign domain, and it also returns what it finds there. Set "
    "enforce_domain_lock in ~/.config/jev-mcp/policy.toml back to true if that is not intended."
)


def _policy_notes(guard: RunGuard) -> tuple[str, ...]:
    """What belongs in **every** result about the policy in force.

    If the domain lock is disabled, there is no blocking reason that could say
    so: nothing is ever blocked. The note is therefore not attached to the
    outcome but to the run.
    """
    if guard.policy.enforce_domain_lock:
        return ()
    return (DOMAIN_LOCK_OFF_NOTE,)


def _no_result(task: _Task, state: _ProgressState) -> RunResult:
    """The thread reported itself done and left nothing behind."""
    return _result(task, state, RunStatus.FAILED, _NO_RESULT_NOTE, error=_NO_RESULT_NOTE)


def _with_note(result: RunResult, note: str) -> RunResult:
    """Appends a note to a finished result, each wording only once."""
    if note in result.notes:
        return result
    return replace(result, notes=(*result.notes, note))


# ---------------------------------------------------------------------------
# The front door
# ---------------------------------------------------------------------------

_TAB_OPEN_NOTE = (
    "The browser tab of this run could not be closed and is probably still open. "
    "Close it by hand before you start the next run."
)

_TAB_UNCLEAR_NOTE = (
    "Closing the browser tab was still not finished after half a second. The result is correct "
    "nonetheless, it just says nothing about the tab. Check in the browser whether it is still open."
)

_CONCURRENT_NOTE = (
    "A run is already in progress, and only one can run at a time. A run operates a single "
    "browser and sets values in the environment of this process for that, two runs would overwrite "
    "each other's values. Wait until the current task is finished, and then start this one again."
)


def _release_run_lock(thread: threading.Thread | None) -> None:
    """Releases the run lock as soon as the worker thread is really done.

    After the time budget runs out, the caller returns with a result while the
    thread is still working in the browser. When the lock used to be released
    immediately, the next call started a second run in the same browser,
    although the tool description promises that only one ever runs. This is not
    an edge case: `Agent.__init__` calls `ensure_daemon()`, which can wait up to
    sixty seconds and start Chrome if necessary, and the default budget of a
    read is thirty seconds.

    So that the lock does not itself become a hang, the waiting has a cap.
    After it, the lock is released even if the thread is still alive.
    """
    if thread is None or not thread.is_alive():
        _RUN_LOCK.release()
        return

    def wait_then_release() -> None:
        try:
            thread.join(_AFTERRUN_CAP_S)
        finally:
            _RUN_LOCK.release()

    threading.Thread(target=wait_then_release, name=f"{thread.name}-linger", daemon=True).start()


def safe_text(value: object) -> str:
    """Turns anything into a text, even if its `__str__` raises."""
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return ""


def _task_from(
    start_url: object,
    goals: object,
    max_actions: object,
    time_budget_s: object,
    dry_run: object,
    agent_factory: Callable[..., object] | None,
) -> _Task:
    """Builds the checked task. May raise, the caller catches everything."""
    limit, budget_notes = _budget(max_actions)
    budget_s, time_notes = _time_budget(time_budget_s)
    goal_sentences, goal_notes = _goals(goals)  # type: ignore[arg-type]
    return _Task(
        start_url=str(start_url or ""),
        goals=goal_sentences,
        max_actions=limit,
        time_budget_s=budget_s,
        dry_run=bool(dry_run),
        agent_factory=agent_factory or _default_agent,
        notes=(*budget_notes, *time_notes, *goal_notes),
    )


def _input_failed(start_url: object, exc: BaseException) -> RunResult:
    """A result for settings that could not even be evaluated."""
    sentence = (
        "The settings for this run could not be evaluated, so no browser was "
        f"opened ({type(exc).__name__}: {condense(safe_text(exc) or 'no message')}). "
        "Check the goal, the action budget and the time budget and try again."
    )
    return RunResult(
        status=RunStatus.NOT_STARTED,
        ok=False,
        summary=sentence,
        start_url=safe_text(start_url),
        url=safe_text(start_url),
        title="",
    )


def run_task(
    start_url: str,
    goals: Sequence[str] | str,
    *,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    allow_domains: Iterable[str] | str | None = None,
    dry_run: bool = False,
    allow_unbound: bool = False,
    policy: Policy | None = None,
    policy_path: object | None = None,
    environment: EnvironmentApplication | None = None,
    agent_factory: Callable[..., object] | None = None,
) -> RunResult:
    """Carries out a complete run and returns a serializable result.

    `goals` are goals in words, single or as a list. `max_actions` is the number
    of actions, default 25, upper limit 60, higher values are capped and that is
    stated in `notes` afterwards. `time_budget_s` is the wall-clock time,
    default 120 seconds, upper limit 900 seconds, and it is enforced even when a
    single step hangs: the run executes in its own thread, and once the budget
    has run out the agent is closed and the result is built.

    `dry_run=True` observes the page and obtains the model's first decision, but
    does not execute it. The result then says what the agent would do next, and
    whether that step would leave the domain this run is bound to. There is
    deliberately no confirmation prompt to the human, the dry run takes its
    place.

    `allow_domains` permits additional domains for this run, everything else
    about the domain lock is in `guards.py`. `environment` and `agent_factory`
    are seams: without them the run applies the environment itself and builds
    the real agent.

    Only one run executes at a time. A second call that finds one running is
    rejected immediately with `not_started` instead of waiting. The reason is in
    `_CONCURRENT_NOTE`. This still holds after a timeout: the caller gets its
    result, but the lock stays until the worker thread is really done, see
    `_release_run_lock()`.

    The agent is closed in every case, including on an exception and on a
    timeout. If that fails, a note in the result says so. This function does not
    raise, every outcome is a `RunResult`.
    """
    try:
        task = _task_from(start_url, goals, max_actions, time_budget_s, dry_run, agent_factory)
    except Exception as exc:  # noqa: BLE001
        return _input_failed(start_url, exc)

    if not task.goals:
        return _without_run(
            task,
            RunStatus.NOT_STARTED,
            "No goal was given, so no browser was opened. Say in words what should happen on the page.",
        )

    if not _RUN_LOCK.acquire(blocking=False):
        return _without_run(task, RunStatus.NOT_STARTED, _CONCURRENT_NOTE)
    thread: threading.Thread | None = None
    try:
        result, thread = _execute(
            task,
            allow_domains=allow_domains,
            allow_unbound=allow_unbound,
            policy=policy,
            policy_path=policy_path,
            environment=environment,
        )
        return result
    finally:
        _release_run_lock(thread)


def _execute(
    task: _Task,
    *,
    allow_domains: Iterable[str] | str | None,
    allow_unbound: bool,
    policy: Policy | None,
    policy_path: object | None,
    environment: EnvironmentApplication | None,
) -> tuple[RunResult, threading.Thread | None]:
    """The run itself, with already checked settings and under the run lock.

    Returns the worker thread alongside the result, if one was started. The
    caller only releases the run lock once this thread is done, see
    `_release_run_lock()`.
    """
    try:
        applied = environment if environment is not None else apply_environment()
        ready = bool(applied.ok)
        environment_notes = tuple(str(note) for note in (applied.notes or ()))
    except Exception as exc:  # noqa: BLE001
        return _without_run(
            task,
            RunStatus.NOT_STARTED,
            "The prerequisites for a run could not be evaluated, so no "
            f"browser was opened ({type(exc).__name__}: "
            f"{condense(safe_text(exc) or 'no message')}).",
        ), None

    if not ready:
        return _without_run(
            task,
            RunStatus.NOT_STARTED,
            "The prerequisites for a run are not met, so no browser was opened. "
            "The notes say what is missing.",
            notes=environment_notes,
        ), None

    try:
        guard = start_run(
            task.start_url,
            allow_domains,
            policy,
            allow_unbound=allow_unbound,
            policy_path=policy_path,  # type: ignore[arg-type]
        )
        entry_check = guard.check(task.start_url, Moment.BEFORE)
    except Exception as exc:  # noqa: BLE001
        return _without_run(
            task,
            RunStatus.NOT_STARTED,
            translate_error(exc),
            notes=environment_notes,
        ), None

    environment_notes = (*environment_notes, *_policy_notes(guard))

    if not entry_check.allowed or not entry_check.may_interact:
        return _without_run(
            task,
            RunStatus.STOPPED_DOMAIN,
            f"The start URL cannot be used for a run, so no browser was opened. {entry_check.reason}",
            notes=(*environment_notes, *entry_check.warnings),
            domain_stop=_domain_stop(entry_check),
        ), None

    progress = _Progress(task.start_url, (*task.notes, *environment_notes))
    session = _Session()
    cancel = threading.Event()
    finished = threading.Event()
    closed = threading.Event()
    result_box: list[RunResult] = []
    close_outcome: list[bool] = []

    def work() -> None:
        # The latch for standard output belongs in the thread itself. The one of
        # the tool call ends with the call, this thread outlives it.
        with stdout_to_stderr():
            try:
                agent = task.agent_factory(task.start_url, list(task.goals))
                session.adopt(agent)
                result_box.append(_loop(task, guard, progress, agent, cancel))
            except BaseException as exc:  # noqa: BLE001
                # Including `KeyboardInterrupt` and `SystemExit`. If this thread
                # only caught `Exception`, it left no result but reported itself
                # done, and the caller then claimed a timeout that never
                # happened. The true cause vanished without a trace.
                result_box.append(_failed(task, progress, exc))
            finally:
                # Report first, then clean up. If closing hangs, the caller would
                # otherwise wait out the whole time budget and report a timeout,
                # although the finished result had long been available.
                finished.set()
                close_outcome.append(session.close())
                closed.set()

    thread = threading.Thread(target=work, name=f"jev-mcp-run-{next(_RUN_NUMBER)}", daemon=True)
    thread.start()
    completed = finished.wait(task.time_budget_s)
    if completed and result_box:
        result = result_box[0]
        if not closed.wait(_SETTLE_TIME_S):
            return _with_note(result, _TAB_UNCLEAR_NOTE), thread
        if close_outcome and not close_outcome[0]:
            return _with_note(result, _TAB_OPEN_NOTE), thread
        return result, thread
    if completed:
        # The thread reported itself done but left nothing behind: its own error
        # branch failed. That was not a timeout.
        return _no_result(task, progress.read()), thread

    # Time budget used up: first tell the thread that it is over, then release
    # the browser tab. A result the thread still leaves behind after that no
    # longer counts, so that the outcome stays unambiguous.
    cancel.set()
    closed_ok = session.close()
    finished.wait(_SETTLE_TIME_S)
    return (
        _time_used_up(task, progress.read(), tab_note=_tab_sentence(session, closed_ok)),
        thread,
    )


# ---------------------------------------------------------------------------
# Reading without acting
# ---------------------------------------------------------------------------
#
# `read_page()` is the second front door of this module. It opens a page,
# observes it exactly once and returns what is on it. It calls neither
# `predict` nor `act`, so there is no model call, and nothing is clicked and
# nothing is typed.
#
# The domain lock applies nonetheless. A redirect can take the page elsewhere,
# and what would then be in the result would come from a foreign domain without
# the requester ever having learned about it. That is why the check after the
# load uses `Moment.AFTER`: same domain means read and record the change,
# foreign domain means stop and return nothing. Nothing means nothing: not the
# title either, and not the address reached. Both are text an attacker sets, and
# a title carries line breaks, control characters and any length. The address
# shown is the defused short form from the decision, the same as in
# `domain_stop.target_url`.
#
# The same rules apply as for `run_task()`: this function never raises, it
# closes the agent in every case, and it holds the same run lock, because it
# operates the same single browser.

READ_GOAL = "Only look at this page. Nothing is clicked and nothing is typed."
"""The task the agent is built with. It is never executed.

`jev_ultrafast.Agent` insists on a task and rejects an empty one. The sentence
is only there so that the agent can be built: no `predict` follows, so no model
ever sees it.
"""

DEFAULT_READ_TIME_BUDGET_S = 30.0
"""Default for the wall-clock time of a read, in seconds."""

DEFAULT_TEXT_LIMIT = 4000
"""The maximum number of characters of visible text a read returns."""

MAX_TEXT_LIMIT = 20000
"""Upper limit for that number. `snapshot.js` delivers at most 6000 anyway."""

MIN_TEXT_LIMIT = 200

MAX_READ_ELEMENTS = 120
"""The maximum number of elements in the table. The library delivers up to 250."""

MAX_ELEMENT_TEXT = 200
"""The maximum length of a label or a field value in the table.

`text_limit` only bounds the visible text. Label, field value and dropdown list
were unbounded, and a page with many long dropdown lists produced a response
of 16.9 megabytes at `text_limit=200`.
"""

MAX_ELEMENT_OPTIONS = 50
"""The maximum number of entries of a dropdown list in the table."""


@dataclass(frozen=True)
class ReadElement:
    """An operable element of the page that was read.

    The same view the decision model gets: `index` is the number under which
    `jev_ultrafast` tracks the element, `operations` says what would be
    possible on it. None of it is executed while reading.
    """

    index: str
    label: str
    role: str = ""
    operations: tuple[str, ...] = ()
    value: str | None = None
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadResult:
    """The result of a read, serializable like `RunResult`.

    `text` is the visible text of the page, shortened to `text_chars`
    characters. `text_total_chars` says how long the **observed** text was
    before shortening, and that is not the same as the length of the page:
    `snapshot.js` cuts off at `LIBRARY_TEXT_LIMIT` characters before
    `text_limit` applies at all. If the observed text reaches exactly this
    length, a note says so. `text_truncated` says whether this module shortened
    it in addition. `redirected` says whether the page ended somewhere other
    than the requested address.

    What applies to secrets in the result is in the module docstring. Reading
    adds nothing to that: nothing is typed, and the task text is a fixed
    sentence.
    """

    status: RunStatus
    ok: bool
    summary: str
    start_url: str
    url: str
    title: str
    text: str = ""
    text_truncated: bool = False
    text_chars: int = 0
    text_total_chars: int = 0
    elements: tuple[ReadElement, ...] = ()
    elements_shown: int = 0
    elements_total: int = 0
    redirected: bool = False
    duration_ms: int = 0
    time_budget_s: float = DEFAULT_READ_TIME_BUDGET_S
    domain_stop: DomainStop | None = None
    notes: tuple[str, ...] = ()
    error: str | None = None


_READ_TIME_NOTE = (
    "The time budget was not a usable number, so the default of "
    f"{int(DEFAULT_READ_TIME_BUDGET_S)} seconds applies."
)

_READ_LIMIT_NOTE = (
    f"The text limit was not a usable number, so the default of {DEFAULT_TEXT_LIMIT} characters applies."
)

_OBSERVATION_LIMIT_NOTE = (
    f"The library observes at most {LIBRARY_TEXT_LIMIT} characters of visible text, and exactly "
    "this length was reached. The page may therefore be longer than what is shown here, "
    "regardless of text_limit."
)


def _read_time_budget(time_budget_s: object) -> tuple[float, tuple[str, ...]]:
    """Checks the time budget of a read."""
    value = _as_number(time_budget_s)
    if value is None or value <= 0:
        return DEFAULT_READ_TIME_BUDGET_S, (_READ_TIME_NOTE,)
    if value > MAX_TIME_BUDGET_S:
        return MAX_TIME_BUDGET_S, (
            f"The time budget of {value:g} seconds is above the upper limit and was capped at "
            f"{int(MAX_TIME_BUDGET_S)} seconds.",
        )
    return value, ()


def _text_limit(text_limit: object) -> tuple[int, tuple[str, ...]]:
    """Checks to how many characters the visible text is shortened."""
    value = _as_number(text_limit)
    if value is None:
        return DEFAULT_TEXT_LIMIT, (_READ_LIMIT_NOTE,)
    number = int(value)
    if number < MIN_TEXT_LIMIT:
        return MIN_TEXT_LIMIT, (
            f"The text limit was below {MIN_TEXT_LIMIT} characters and was raised to {MIN_TEXT_LIMIT} "
            "characters.",
        )
    if number > MAX_TEXT_LIMIT:
        return MAX_TEXT_LIMIT, (
            f"The text limit of {number} characters is above the upper limit and was capped at "
            f"{MAX_TEXT_LIMIT} characters.",
        )
    return number, ()


_ELEMENT_TEXT_NOTE = (
    f"At least one label or field value was longer than {MAX_ELEMENT_TEXT} characters and "
    "was capped for the table."
)

_ELEMENT_OPTIONS_NOTE = (
    f"At least one dropdown list had more than {MAX_ELEMENT_OPTIONS} entries. The table "
    f"shows the first {MAX_ELEMENT_OPTIONS}."
)


def _capped(value: object, capped_notes: list[str]) -> str:
    """Shortens a single value of the element table and records that."""
    text = safe_text(value)
    if len(text) <= MAX_ELEMENT_TEXT:
        return text
    capped_notes.append(_ELEMENT_TEXT_NOTE)
    return text[:MAX_ELEMENT_TEXT]


def _read_elements(snapshot: Mapping) -> tuple[tuple[ReadElement, ...], int, tuple[str, ...]]:
    """Builds the element table from `agent.snapshot()["elements"]`.

    This is the same table that `jev_ultrafast.model.action_space()` builds for
    the decision model. If it is missing, the table stays empty instead of
    something being invented here.

    Also returns the notes if something was capped. The check is for `list` and
    `tuple`, not for `Sequence`: a string is a Sequence, and the response
    therefore once claimed "The page has 57 operable elements, the table shows
    the first 0", because it had counted the string's characters.
    """
    raw = snapshot.get("elements") or []
    if not isinstance(raw, list | tuple):
        return (), 0, ()
    collected_notes: list[str] = []
    element_rows: list[ReadElement] = []
    for entry in list(raw)[:MAX_READ_ELEMENTS]:
        if not isinstance(entry, Mapping):
            continue
        raw_options = [option for option in (entry.get("options") or []) if isinstance(option, Mapping)]
        if len(raw_options) > MAX_ELEMENT_OPTIONS:
            collected_notes.append(_ELEMENT_OPTIONS_NOTE)
        option_labels = tuple(
            _capped(option.get("label"), collected_notes) for option in raw_options[:MAX_ELEMENT_OPTIONS]
        )
        value = entry.get("value")
        element_rows.append(
            ReadElement(
                index=_capped(entry.get("index") or "", collected_notes),
                label=_capped(entry.get("label") or "", collected_notes),
                role=_capped(entry.get("role") or "", collected_notes),
                operations=tuple(_capped(name, collected_notes) for name in (entry.get("operations") or [])),
                value=None if value is None else _capped(value, collected_notes),
                options=option_labels,
            )
        )
    return tuple(element_rows), len(raw), tuple(dict.fromkeys(collected_notes))


def _shortened_text(text: object, limit: int) -> tuple[str, bool, int]:
    """Shortens the visible text and says how long it was before."""
    full = safe_text(text or "")
    if len(full) <= limit:
        return full, False, len(full)
    return full[:limit], True, len(full)


def _read_result(
    start_url: str,
    budget_s: float,
    status: RunStatus,
    summary: str,
    *,
    ok: bool = False,
    url: str | None = None,
    title: str = "",
    notes: Sequence[str] = (),
    domain_stop: DomainStop | None = None,
    duration_ms: int = 0,
    error: str | None = None,
) -> ReadResult:
    """A read result without page content, for every outcome except success."""
    return ReadResult(
        status=status,
        ok=ok,
        summary=summary,
        start_url=start_url,
        url=url if url is not None else start_url,
        title=title,
        time_budget_s=budget_s,
        duration_ms=duration_ms,
        notes=tuple(dict.fromkeys(str(note) for note in notes if note)),
        domain_stop=domain_stop,
        error=error,
    )


def _observe_once(
    agent: object,
    start_url: str,
    guard: RunGuard,
    budget_s: float,
    limit: int,
    initial_notes: Sequence[str],
    started: float,
) -> ReadResult:
    """Observes the opened page exactly once and turns that into a result."""
    snapshot = agent.snapshot()  # type: ignore[attr-defined]
    page = snapshot.get("page") or {}
    reached_url = safe_text(page.get("url") or start_url)
    page_title = safe_text(page.get("title") or "")
    elapsed = round((time.monotonic() - started) * 1000)

    collected_notes = [*initial_notes]
    decision = guard.check(reached_url, Moment.AFTER)
    if decision.policy_note:
        collected_notes.append(decision.policy_note)
    collected_notes.extend(decision.warnings)

    if not decision.allowed:
        sentence = (
            "The page switched to a foreign domain while loading, so nothing was "
            f"read. {decision.reason} If that is intended, name the domain in "
            "allow_domains."
        )
        # Nothing comes back from the foreign page, not its title either and
        # not its raw address. A title is arbitrary text: an instruction to the
        # model, control characters, any length. The address shown is the
        # defused short form from the decision.
        return _read_result(
            start_url,
            budget_s,
            RunStatus.STOPPED_DOMAIN,
            sentence,
            url=decision.target_url,
            title="",
            notes=collected_notes,
            domain_stop=_domain_stop(decision),
            duration_ms=elapsed,
        )

    was_redirected = reached_url != start_url
    if was_redirected:
        collected_notes.append(
            f"The requested URL redirected to {reached_url}, and that page is the one that was read."
        )

    text, was_shortened, full_length = _shortened_text(page.get("text"), limit)
    if was_shortened:
        collected_notes.append(
            f"The observed text was {full_length} characters long and was shortened to {len(text)} "
            "characters, the rest is not in this response."
        )
    if full_length >= LIBRARY_TEXT_LIMIT:
        collected_notes.append(_OBSERVATION_LIMIT_NOTE)
    element_rows, total, cap_notes = _read_elements(snapshot)
    collected_notes.extend(cap_notes)
    if total > len(element_rows):
        collected_notes.append(
            f"The page has {total} operable elements, the table shows the first {len(element_rows)}."
        )
    omitted = page.get("omitted_actions") or 0
    if isinstance(omitted, int) and omitted > 0:
        collected_notes.append(
            f"The library did not observe {omitted} further elements at all, the page is too large for that."
        )

    sentence = (
        f"The page {reached_url} was read: {len(text)} characters of visible text and "
        f"{len(element_rows)} operable elements. Nothing was clicked and nothing was typed."
    )
    return ReadResult(
        status=RunStatus.DONE,
        ok=True,
        summary=sentence,
        start_url=start_url,
        url=reached_url,
        title=page_title,
        text=text,
        text_truncated=was_shortened,
        text_chars=len(text),
        text_total_chars=full_length,
        elements=element_rows,
        elements_shown=len(element_rows),
        elements_total=total,
        redirected=was_redirected,
        duration_ms=elapsed,
        time_budget_s=budget_s,
        notes=tuple(dict.fromkeys(str(note) for note in collected_notes if note)),
    )


def read_page(
    url: str,
    *,
    time_budget_s: float = DEFAULT_READ_TIME_BUDGET_S,
    allow_domains: Iterable[str] | str | None = None,
    text_limit: int = DEFAULT_TEXT_LIMIT,
    policy: Policy | None = None,
    policy_path: object | None = None,
    environment: EnvironmentApplication | None = None,
    agent_factory: Callable[..., object] | None = None,
) -> ReadResult:
    """Opens a page, observes it once and returns what is on it.

    Nothing is clicked, nothing is typed and nothing is selected: neither
    `predict` nor `act` is called, so there is no model call and no cost
    either. What comes back is the visible text, shortened to `text_limit`
    characters, and the element table the decision model would also see.

    The domain lock applies here as well. If the page ends up on a foreign
    domain after a redirect, the operation stops and returns nothing from
    there. A change within the same domain is read and recorded in the result,
    it is not kept quiet.

    Only one operation executes at a time, `run_task()` and `read_page()` share
    this lock because they share the browser, and after a timeout it stays held
    until the worker thread is really done. The agent is closed in every case.
    This function does not raise, every outcome is a `ReadResult`.
    """
    try:
        address = safe_text(url or "").strip()
        budget_s, time_notes = _read_time_budget(time_budget_s)
        limit, limit_notes = _text_limit(text_limit)
    except Exception as exc:  # noqa: BLE001
        sentence = (
            "The settings for this read could not be evaluated, so no "
            f"page was opened ({type(exc).__name__}: "
            f"{condense(safe_text(exc) or 'no message')})."
        )
        return _read_result(safe_text(url), DEFAULT_READ_TIME_BUDGET_S, RunStatus.NOT_STARTED, sentence)

    initial_notes = [*time_notes, *limit_notes]
    if not address:
        return _read_result(
            address,
            budget_s,
            RunStatus.NOT_STARTED,
            "No URL was given, so no page was opened.",
            notes=initial_notes,
        )

    if not _RUN_LOCK.acquire(blocking=False):
        return _read_result(address, budget_s, RunStatus.NOT_STARTED, _CONCURRENT_NOTE, notes=initial_notes)
    thread: threading.Thread | None = None
    try:
        result, thread = _perform_read(
            address,
            budget_s,
            limit,
            initial_notes,
            allow_domains=allow_domains,
            policy=policy,
            policy_path=policy_path,
            environment=environment,
            agent_factory=agent_factory,
        )
        return result
    finally:
        _release_run_lock(thread)


def _perform_read(
    address: str,
    budget_s: float,
    limit: int,
    initial_notes: Sequence[str],
    *,
    allow_domains: Iterable[str] | str | None,
    policy: Policy | None,
    policy_path: object | None,
    environment: EnvironmentApplication | None,
    agent_factory: Callable[..., object] | None,
) -> tuple[ReadResult, threading.Thread | None]:
    """The read itself, with checked settings and under the run lock.

    Returns the worker thread alongside the result, if one was started, see
    `_release_run_lock()`.
    """
    try:
        applied = environment if environment is not None else apply_environment()
        ready = bool(applied.ok)
        environment_notes = tuple(str(note) for note in (applied.notes or ()))
    except Exception as exc:  # noqa: BLE001
        sentence = (
            "The prerequisites for a read could not be evaluated, so no "
            f"page was opened ({type(exc).__name__}: "
            f"{condense(safe_text(exc) or 'no message')})."
        )
        return _read_result(address, budget_s, RunStatus.NOT_STARTED, sentence, notes=initial_notes), None

    collected_notes = [*initial_notes, *environment_notes]
    if not ready:
        return _read_result(
            address,
            budget_s,
            RunStatus.NOT_STARTED,
            "The prerequisites for a read are not met, so no page was opened. The notes say what is missing.",
            notes=collected_notes,
        ), None

    try:
        guard = start_run(
            address,
            allow_domains,
            policy,
            policy_path=policy_path,  # type: ignore[arg-type]
        )
        entry_check = guard.check(address, Moment.BEFORE)
    except Exception as exc:  # noqa: BLE001
        return _read_result(
            address, budget_s, RunStatus.NOT_STARTED, translate_error(exc), notes=collected_notes
        ), None

    collected_notes.extend(_policy_notes(guard))

    if not entry_check.allowed or not entry_check.may_interact:
        return _read_result(
            address,
            budget_s,
            RunStatus.STOPPED_DOMAIN,
            f"The URL cannot be read, so no page was opened. {entry_check.reason}",
            notes=(*collected_notes, *entry_check.warnings),
            domain_stop=_domain_stop(entry_check),
        ), None

    started = time.monotonic()
    session = _Session()
    finished = threading.Event()
    closed = threading.Event()
    result_box: list[ReadResult] = []
    close_outcome: list[bool] = []
    factory = agent_factory or _default_agent

    def work() -> None:
        # The latch for standard output belongs in the thread itself, see
        # `_execute()`.
        with stdout_to_stderr():
            try:
                agent = factory(address, [READ_GOAL])
                session.adopt(agent)
                result_box.append(
                    _observe_once(agent, address, guard, budget_s, limit, collected_notes, started)
                )
            except BaseException as exc:  # noqa: BLE001
                sentence = translate_error(exc)
                result_box.append(
                    _read_result(
                        address,
                        budget_s,
                        RunStatus.FAILED,
                        sentence,
                        notes=collected_notes,
                        duration_ms=round((time.monotonic() - started) * 1000),
                        error=sentence,
                    )
                )
            finally:
                finished.set()
                close_outcome.append(session.close())
                closed.set()

    thread = threading.Thread(target=work, name=f"jev-mcp-read-{next(_RUN_NUMBER)}", daemon=True)
    thread.start()
    completed = finished.wait(budget_s)
    if completed and result_box:
        result = result_box[0]
        if not closed.wait(_SETTLE_TIME_S):
            return _with_read_note(result, _TAB_UNCLEAR_NOTE), thread
        if close_outcome and not close_outcome[0]:
            return _with_read_note(result, _TAB_OPEN_NOTE), thread
        return result, thread
    if completed:
        return _read_result(
            address,
            budget_s,
            RunStatus.FAILED,
            _NO_RESULT_NOTE,
            notes=collected_notes,
            duration_ms=round((time.monotonic() - started) * 1000),
            error=_NO_RESULT_NOTE,
        ), thread

    closed_ok = session.close()
    finished.wait(_SETTLE_TIME_S)
    return _read_result(
        address,
        budget_s,
        RunStatus.STOPPED_TIME,
        f"The page had still not been read after the time budget of {budget_s:g} seconds, the "
        f"operation was aborted. {_tab_sentence(session, closed_ok)}",
        notes=collected_notes,
        duration_ms=round((time.monotonic() - started) * 1000),
    ), thread


def _with_read_note(result: ReadResult, note: str) -> ReadResult:
    """Appends a note to a finished read result, each wording only once."""
    if note in result.notes:
        return result
    return replace(result, notes=(*result.notes, note))
