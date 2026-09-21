"""The MCP layer: three tools over stdio.

This module is thin. It decides nothing about browsers, domains or budgets;
`config.py`, `guards.py` and `runner.py` do that. It does four things:

1. It applies the environment **exactly once**. `apply_environment()` writes
   into `os.environ` for the whole process, and the process here is the
   server, not the single call. The result is kept and passed on to every run.
   If it is not in order, that is stated in **every** response, not only in
   the first one, because a model rarely reads the first one.
2. It checks the inputs before they go any further. Everything arrives as JSON
   from a model. `run_task()` does catch every bad input, but a clear sentence
   at the door is better than a result with `not_started` that first has to be
   read.
3. It turns every result into a structure that can be sent as UTF-8. A `NaN`
   in a confidence value from the library would otherwise become `NaN` in the
   JSON, and that is not valid JSON. A lone surrogate from `os.environ` would
   make pydantic raise during serialization, and only after the tool has
   returned, where nobody can explain it any more.
4. It helps keep standard output clean. With stdio, standard output is the
   protocol channel, and anything else that goes there breaks the connection.

Standard output in detail, counted honestly
-------------------------------------------
The actual barrier does not belong to this project. The SDK (mcp 2.2.0,
`mcp/server/stdio.py`) redirects file descriptor 1 to stderr while it runs and
serves the connection from a private copy. A `print`, an `os.write(1, ...)` and
even a subprocess therefore already end up where they belong. If that foreign
barrier is missing, for example in an environment where `sys.stdout` does not
sit on descriptor 1 at all, only two partial covers remain:

* `runner.stdout_to_stderr()` points `sys.stdout` at stderr while a tool call
  or a worker thread is running. It touches `sys.stdout`, not the descriptor:
  an `os.write(1, ...)` and a subprocess get past it. The barrier deliberately
  sits in the worker thread as well, because that thread outlives the time
  budget and with it the tool call.
* `main()` explicitly sends logging to stderr.

This module contains no `print`, and two tests hold that promise against the
real connection: one in process for the thread that outlives the budget, one
as a subprocess over real stdio.

The tool descriptions are written for a model, which reads them to pick a
tool. The texts in results are written for the person at the end of the
chain, because that is where they arrive. Both are in English.
"""

import dataclasses
import logging
import math
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from threading import Lock
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import runner
from .config import DEFAULT_CONFIG_PATH, EnvironmentApplication, apply_environment, diagnose

__all__ = [
    "ToolError",
    "browser_read",
    "browser_status",
    "browser_task",
    "create_server",
    "main",
]

SERVER_NAME = "jev-mcp"

MAX_URL_LENGTH = 2048
MAX_GOALS = 20
MAX_GOAL_LENGTH = 2000
MAX_ALLOW_DOMAINS = 50
MAX_DOMAIN_LENGTH = 253

ENVIRONMENT_WARNING = (
    "The environment for the browser agent could not be set up, so no run can start. "
    'The "environment" entry says why.'
)

INSTRUCTIONS = (
    "Drives the jev-ultrafast browser agent in the user's own, already logged-in Chrome. "
    "Use browser_status when something fails or before the first run of a session, browser_read to "
    "look at a single page without touching it, and browser_task when a site has to be operated. "
    "Every run stays on the start URL's registrable domain unless allow_domains says otherwise, and "
    "only one run can be in flight at a time."
)

BROWSER_TASK_DESCRIPTION = (
    "Operate a website autonomously in the user's real, logged-in Chrome: open `url`, then pursue "
    "`goals` written as plain sentences, deciding every click, selection and keystroke without "
    "asking. Right for multi-step work on one site: filling a form, walking to a page that is only "
    "reachable by clicking, searching inside a site, reading something behind a session the user "
    "already has. Wrong for reading one page (use browser_read), for public pages that need no "
    "login (use a web fetch or search tool), and for logging in, paying, ordering or submitting "
    "anything on a banking or payment site. Never put credentials into `goals`: the goals and every "
    "typed value come back in the result. The run stays on the registrable domain of `url` and "
    "stops when a step would leave it; widen that with `allow_domains`. Bounded by `max_actions` "
    "(default 25, values above 60 are rejected) and `time_budget_s` (default 120 s, values above "
    "900 s are rejected); the call blocks until the run ends, and only one run can be in flight at "
    "a time, including any run still finishing after its budget ran out. Set `dry_run` to get "
    "the first planned step without executing anything. The underlying agent cannot see into "
    "iframes or shadow DOM, cannot do file uploads, cannot follow pop-up tabs, and only observes "
    "the visible viewport, so it scrolls but never reads what is not rendered. The result reports "
    "the status, the final URL, every executed step and the reason it stopped."
)

BROWSER_STATUS_DESCRIPTION = (
    "Report whether a browser run can work at all, before spending time on one. Says which API keys "
    "are present and where they came from, which text model would type into fields, whether the "
    "browser-harness daemon is running and whether a Chrome is connected to it, and which "
    "operations are therefore blocked. Read-only and free: it opens no browser, loads no page, "
    "calls no model and costs nothing. Call it when browser_task or browser_read fails, before the "
    "first run of a session, or when the user asks whether the browser agent is ready. It says "
    "nothing about any particular website."
)

BROWSER_READ_DESCRIPTION = (
    "Open one page in the user's real, logged-in Chrome and return its visible text plus the table "
    "of elements that can be operated. This tool clicks nothing, types nothing and selects nothing. "
    "Opening the page is itself a request, though: it opens a background tab and navigates to `url` "
    "with the user's cookies, so a URL that acts on GET will act, for example a logout, an "
    "unsubscribe, a confirmation link or a one-click action. Right for reading or checking a page, "
    "for seeing what a page offers before starting browser_task, and for content behind a login the "
    "user already has. Wrong when anything has to be clicked or filled in (use browser_task), and "
    "wasteful for public pages that need no login (use a web fetch or search tool). Redirects are "
    "followed but verified: landing on another registrable domain is reported, and none of that "
    "page's text, elements, title or URL is returned; widen that with `allow_domains`. The "
    "underlying library observes at most 6000 characters of visible text and cuts a longer page "
    "before `text_limit` (default 4000) applies, so the answer reports how long the observed text "
    "was and warns when it reached that ceiling. Labels and field values are capped at 200 "
    "characters and option lists at 50 entries per element, and the answer says when it capped "
    "something. Same limits as the agent: no iframes, no shadow DOM, no file uploads, no pop-up "
    "tabs, and only the visible viewport is observed, so content further down the page can be "
    "missing. Only one run can be in flight at a time."
)


# ---------------------------------------------------------------------------
# The environment, exactly once
# ---------------------------------------------------------------------------

_ENVIRONMENT: EnvironmentApplication | None = None
_ENVIRONMENT_LOCK = Lock()
_SNAPSHOT: dict[str, str] | None = None
_CONFIG_STATE: tuple[object, ...] | None = None


def _config_state() -> tuple[object, ...]:
    """A fingerprint of the config file that shows whether it has changed.

    Returns `(exists, modification time, size)`. If the file is missing or
    cannot be queried, that is recorded in the tuple as well, instead of
    anything raising here.
    """
    try:
        stat_result = DEFAULT_CONFIG_PATH.stat()
    except OSError:
        return (False, 0, 0)
    return (True, stat_result.st_mtime_ns, stat_result.st_size)


def environment() -> EnvironmentApplication:
    """The result of `apply_environment()`, applied once.

    The first access sets the environment; after that it is only looked up.
    `apply_environment()` does not raise; a failure is recorded in `ok` and in
    `notes`.

    Before applying, the process environment is photographed. The diagnosis
    later resolves against that snapshot and not against the changed
    environment; otherwise it would read its own writes and report every value
    from the config file as coming "from the environment".
    """
    global _ENVIRONMENT, _SNAPSHOT, _CONFIG_STATE
    with _ENVIRONMENT_LOCK:
        if _ENVIRONMENT is None:
            _SNAPSHOT = dict(os.environ)
            _CONFIG_STATE = _config_state()
            _ENVIRONMENT = apply_environment()
        return _ENVIRONMENT


RESTART_NOTE = (
    "The config file has changed since this server started. What applied at startup still "
    "applies, because the environment is set exactly once. Restart the server for the change to "
    "take effect."
)


def _provenance_notes(application: EnvironmentApplication) -> list[str]:
    """What every diagnosis needs to say about where the keys came from.

    The diagnosis resolves against the snapshot taken before startup. That it
    does not read its own writes still needs saying, because the running
    process environment does now contain those values.
    """
    notes: list[str] = []
    if application.applied:
        names = ", ".join(application.applied)
        notes.append(
            f"This server set these variables itself at startup: {names}. The sources below are "
            "therefore resolved against the environment from before startup, not against the "
            "current process environment."
        )
    if _CONFIG_STATE is not None and _config_state() != _CONFIG_STATE:
        notes.append(RESTART_NOTE)
    return notes


# ---------------------------------------------------------------------------
# Inputs are foreign data
# ---------------------------------------------------------------------------


def _url(value: object) -> str:
    """Checks the start URL of a tool call."""
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        raise ToolError(
            "No URL was given. Name the page the run should start on, for example "
            "https://example.com/contact."
        )
    if len(text) > MAX_URL_LENGTH:
        raise ToolError(
            f"The URL is {len(text)} characters long, more than the allowed {MAX_URL_LENGTH} characters."
        )
    scheme = text.split(":", 1)[0].lower() if ":" in text else ""
    if scheme not in {"http", "https"}:
        raise ToolError(
            "The URL must start with http:// or https://. This server does not open any other scheme."
        )
    return text


def _goals(value: object) -> tuple[list[str], list[str]]:
    """Turns the goals input into a list of usable sentences plus notes.

    An empty goal is still dropped, but no longer silently: `["Find the page",
    ""]` used to become a run with one goal without that being stated anywhere.
    """
    if isinstance(value, str):
        raw: list[object] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        raw = list(value)
    else:
        raise ToolError(
            "The goals must be a sentence or a list of sentences, for example "
            '["Find the contact page", "Read out the phone number"].'
        )
    if any(not isinstance(entry, str) for entry in raw):
        raise ToolError("Every goal must be a sentence in words, not a number and not an object.")
    goals = [str(entry).strip() for entry in raw if str(entry).strip()]
    if not goals:
        raise ToolError(
            "No goal was given. Write in words what should happen on the page, "
            'for example "Find the contact page and read the phone number".'
        )
    if len(goals) > MAX_GOALS:
        raise ToolError(
            f"Too many goals were given: {len(goals)}. At most {MAX_GOALS} are allowed. Split the "
            "task into several runs."
        )
    too_long = next((goal for goal in goals if len(goal) > MAX_GOAL_LENGTH), None)
    if too_long is not None:
        raise ToolError(
            f"A goal is {len(too_long)} characters long, more than the allowed {MAX_GOAL_LENGTH} "
            "characters. Make it shorter."
        )
    dropped = len(raw) - len(goals)
    notes = (
        [
            f"Of the given goals, {dropped} were empty. They were dropped, and the run uses "
            f"the remaining {len(goals)}."
        ]
        if dropped
        else []
    )
    return goals, notes


def _domains(value: object) -> tuple[list[str] | None, list[str]]:
    """Checks `allow_domains`. A single string counts as one entry.

    Empty entries are dropped, and that is stated too, instead of the list
    silently getting shorter.
    """
    if value is None:
        return None, []
    if isinstance(value, str):
        raw: list[object] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        raw = list(value)
    else:
        raise ToolError(
            "allow_domains must be a domain or a list of domains, for example "
            '["example.com", "*.example.net"].'
        )
    if any(not isinstance(entry, str) for entry in raw):
        raise ToolError("Every domain in allow_domains must be a string.")
    domains = [str(entry).strip() for entry in raw if str(entry).strip()]
    if len(domains) > MAX_ALLOW_DOMAINS:
        raise ToolError(
            f"Too many domains were given: {len(domains)}. At most {MAX_ALLOW_DOMAINS} are allowed."
        )
    too_long = next((domain for domain in domains if len(domain) > MAX_DOMAIN_LENGTH), None)
    if too_long is not None:
        raise ToolError(
            f"An entry in allow_domains is {len(too_long)} characters long, so it is not a domain."
        )
    dropped = len(raw) - len(domains)
    notes = [f"Of the entries in allow_domains, {dropped} were empty and were dropped."] if dropped else []
    return domains or None, notes


def _number(name: str, value: object, *, minimum: float, maximum: float) -> float:
    """Checks that a number is finite and within the allowed range.

    `True` and `False` are numbers in Python, and `float(True)` is a plain
    one. `max_actions: true` therefore silently produced a run with a single
    action, and `time_budget_s: true` a budget of one second. Both lie within
    the allowed range, so neither stood out anywhere.
    """
    if isinstance(value, bool):
        raise ToolError(f"{name} must be a number between {minimum:g} and {maximum:g}, not a boolean.")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ToolError(f"{name} must be a number between {minimum:g} and {maximum:g}.") from None
    if not math.isfinite(number):
        raise ToolError(f"{name} must be a finite number between {minimum:g} and {maximum:g}.")
    if number < minimum or number > maximum:
        raise ToolError(
            f"{name} is {number:g}, which is outside the allowed range of {minimum:g} to {maximum:g}."
        )
    return number


def _integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    """Like `_number`, but returns an integer."""
    return int(_number(name, value, minimum=minimum, maximum=maximum))


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def _sendable(value: object) -> str:
    """Turns a text into one that can be sent as UTF-8.

    The SDK serializes with pydantic, and pydantic writes UTF-8 directly. A
    lone surrogate in the text makes it raise, and only **after** the tool has
    returned: the client then saw nothing but "Error executing tool
    browser_status" without a reason, of all things for the tool that is meant
    to explain errors. On POSIX, `os.environ` yields exactly such characters
    for an invalid byte in a variable, so this is not a theoretical case.
    """
    return str(value).encode("utf-8", "replace").decode("utf-8")


def _json_safe(value: object) -> Any:
    """Turns a result into something that survives `json.dumps(allow_nan=False)`.

    Tuples become lists, enums become their text, `NaN` and the infinities
    become `None`, and anything unknown becomes its `str()`. The steps of a run
    come from the library, so something the JSON encoder does not know can
    show up there too.
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _sendable(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {_sendable(str(key)): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_json_safe(item) for item in value]
    return _sendable(runner.safe_text(value))


def _response(result: object) -> dict[str, Any]:
    """Turns a result into the tool's response, including the environment state.

    The state of the environment is in every response, not only in the first.
    If it is not in order, `ok` or `ready` is also flipped, because then
    nothing runs, whatever else the response says.
    """
    data = _json_safe(dataclasses.asdict(result))  # type: ignore[call-overload]
    applied = environment()
    data["environment"] = _json_safe(dataclasses.asdict(applied))
    if not applied.ok:
        notes = list(data.get("notes") or [])
        if ENVIRONMENT_WARNING not in notes:
            notes.insert(0, ENVIRONMENT_WARNING)
        data["notes"] = notes
        if "ok" in data:
            data["ok"] = False
        if "ready" in data:
            data["ready"] = False
    return data


def _stdout_guard() -> Iterator[None]:
    """The standard output barrier, for the duration of a tool call.

    The barrier itself lives in `runner.stdout_to_stderr()`, because the worker
    thread of a run needs it as well and outlives the call. Both holders are
    counted, so the output is only restored once the last one lets go.
    """
    return runner.stdout_to_stderr()


def _with_environment_warning(error: ToolError) -> ToolError:
    """Appends the state of the environment to a tool error if it is not in order.

    The module docstring promises that a broken environment is stated in
    **every** response. A raised `ToolError` is a response too, and that
    sentence used to be missing there.
    """
    text = str(error)
    if environment().ok or ENVIRONMENT_WARNING in text:
        return error
    return ToolError(f"{text} {ENVIRONMENT_WARNING}")


def _with_notes(data: dict[str, Any], notes: Sequence[str]) -> dict[str, Any]:
    """Appends notes from the tool layer to the response, each wording only once."""
    if not notes:
        return data
    merged = list(data.get("notes") or [])
    for note in notes:
        if note not in merged:
            merged.append(note)
    data["notes"] = merged
    return data


def _unexpected(name: str, error: BaseException) -> ToolError:
    """Turns an unexpected error into a readable tool error."""
    text = runner.condense(runner.safe_text(error) or "no message")
    return ToolError(
        f"The tool {name} failed unexpectedly ({type(error).__name__}: {text}). The "
        "server is still running; try again or ask browser_status first."
    )


# ---------------------------------------------------------------------------
# The three tools
# ---------------------------------------------------------------------------


def browser_task(
    url: str,
    goals: list[str] | str,
    max_actions: int = runner.DEFAULT_MAX_ACTIONS,
    time_budget_s: float = runner.DEFAULT_TIME_BUDGET_S,
    allow_domains: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """The autonomous run. See `BROWSER_TASK_DESCRIPTION` for the text a model reads."""
    try:
        address = _url(url)
        sentences, goal_notes = _goals(goals)
        actions = _integer("max_actions", max_actions, minimum=1, maximum=runner.LIBRARY_MAX_ACTIONS)
        budget = _number("time_budget_s", time_budget_s, minimum=1.0, maximum=runner.MAX_TIME_BUDGET_S)
        domains, domain_notes = _domains(allow_domains)
        with _stdout_guard():
            result = runner.run_task(
                address,
                sentences,
                max_actions=actions,
                time_budget_s=budget,
                allow_domains=domains,
                dry_run=bool(dry_run),
                environment=environment(),
            )
        return _with_notes(_response(result), [*goal_notes, *domain_notes])
    except ToolError as error:
        raise _with_environment_warning(error) from None
    except Exception as error:  # noqa: BLE001
        raise _with_environment_warning(_unexpected("browser_task", error)) from error


def browser_status() -> dict[str, Any]:
    """The diagnosis without side effects. See `BROWSER_STATUS_DESCRIPTION`."""
    try:
        application = environment()
        with _stdout_guard():
            diagnosis = diagnose(env=_SNAPSHOT)
        return _with_notes(_response(diagnosis), _provenance_notes(application))
    except ToolError as error:
        raise _with_environment_warning(error) from None
    except Exception as error:  # noqa: BLE001
        raise _with_environment_warning(_unexpected("browser_status", error)) from error


def browser_read(
    url: str,
    time_budget_s: float = runner.DEFAULT_READ_TIME_BUDGET_S,
    allow_domains: list[str] | None = None,
    text_limit: int = runner.DEFAULT_TEXT_LIMIT,
) -> dict[str, Any]:
    """Read one page without acting. See `BROWSER_READ_DESCRIPTION`."""
    try:
        address = _url(url)
        budget = _number("time_budget_s", time_budget_s, minimum=1.0, maximum=runner.MAX_TIME_BUDGET_S)
        limit = _integer(
            "text_limit",
            text_limit,
            minimum=runner.MIN_TEXT_LIMIT,
            maximum=runner.MAX_TEXT_LIMIT,
        )
        domains, domain_notes = _domains(allow_domains)
        with _stdout_guard():
            result = runner.read_page(
                address,
                time_budget_s=budget,
                allow_domains=domains,
                text_limit=limit,
                environment=environment(),
            )
        return _with_notes(_response(result), domain_notes)
    except ToolError as error:
        raise _with_environment_warning(error) from None
    except Exception as error:  # noqa: BLE001
        raise _with_environment_warning(_unexpected("browser_read", error)) from error


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


def _version() -> str:
    try:
        return version("jev-mcp")
    except PackageNotFoundError:  # Only without an installed package.
        return "0.0.0"


def create_server() -> MCPServer:
    """Builds the server and registers the three tools.

    The environment is applied here, so once at startup and not on every
    call.
    """
    environment()
    server = MCPServer(name=SERVER_NAME, version=_version(), instructions=INSTRUCTIONS)
    server.add_tool(browser_task, name="browser_task", description=BROWSER_TASK_DESCRIPTION)
    server.add_tool(browser_status, name="browser_status", description=BROWSER_STATUS_DESCRIPTION)
    server.add_tool(browser_read, name="browser_read", description=BROWSER_READ_DESCRIPTION)
    return server


_log = logging.getLogger(SERVER_NAME)


def _log_to_stderr(level: int = logging.INFO) -> None:
    """Sends all log output to stderr.

    With stdio, standard output is the protocol channel. A library that logs
    there breaks the connection, and the client only sees that nothing works
    any more.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def main() -> None:
    """The console command `jev-mcp`: starts the server over stdio.

    When the client goes away, that is not an error but the end. Without this
    handling the server ended with a traceback and exit code 1, and in Claude
    Desktop that looked like a crash. The SDK bundles exceptions from its task
    groups, so the group is split apart as well.
    """
    _log_to_stderr()
    try:
        create_server().run("stdio")
    except (BrokenPipeError, KeyboardInterrupt):
        _log.info("The client closed the connection, the server is shutting down.")
    except BaseExceptionGroup as group:
        _, rest = group.split((BrokenPipeError, KeyboardInterrupt))
        if rest is not None:
            raise rest from None
        _log.info("The client closed the connection, the server is shutting down.")


if __name__ == "__main__":  # The entry point as a module.
    main()
