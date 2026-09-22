# jev-mcp

jev-mcp is an MCP server that hands the browser agent
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast) to Claude Code and Claude Desktop. You give it a start URL and your goals in plain sentences, and it operates the
page on its own in your real, already logged-in Chrome. A TypeSafe decision model (Jev) picks every
step, and a separate text model writes the values that get typed into fields.

## Requirements

* Python 3.12 or newer
* `uv`
* Google Chrome, driven through `browser-harness`
* A TypeSafe API key (`TYPESAFE_API_KEY`). Without it no run starts at all, because nothing can
  choose an action, not even a click or a scroll.
* A text model key. Without it the agent can still click, scroll, navigate and pick from dropdowns,
  but it cannot type, so forms and search fields stay empty.
* `git` and access to GitHub during install. jev-ultrafast is not published on PyPI, so it is
  installed straight from its repository, pinned to the exact commit this server was tested against.

## Setup

```
git clone https://github.com/SAITS-Swiss-AI-Tech-Services/jev-mcp.git
cd jev-mcp
uv sync
```

Put the keys into `~/.config/jev-mcp/env`. The format is `NAME=VALUE`, one entry per line. Empty
lines and lines starting with `#` are skipped, a leading `export ` is stripped, quotes around a
value are removed, and a trailing comment is cut off. The file is read only when it is a regular
file of at most 256 KiB.

```
TYPESAFE_API_KEY=...
MOONSHOT_API_KEY=...
```

Claude Desktop inherits no shell, which is exactly why this file exists. Claude Code does inherit
the shell, so keys already exported there are found as well.

The server looks for the text model key by provider first and by source second. It tries
`TEXT_MODEL_API_KEY`, then Kimi (`MOONSHOT_API_KEY`, `KIMI_API_KEY`), then `DEEPSEEK_API_KEY`, then
`OPENROUTER_API_KEY`, and inside each of those steps the process environment before the file. The
first hit wins and decides the provider, including base URL and model name. For
`TYPESAFE_API_KEY` there is only one variable, and there the environment simply beats the file.

Register the server in Claude Code. Use the absolute path to `uv`:

```
claude mcp add jev-browser --scope user -- /absolute/path/to/uv run --directory /absolute/path/to/jev-mcp jev-mcp
```

Register it in Claude Desktop by adding the same command to
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "jev-browser": {
      "command": "/absolute/path/to/uv",
      "args": ["run", "--directory", "/absolute/path/to/jev-mcp", "jev-mcp"]
    }
  }
}
```

Both entries need absolute paths, because Claude Desktop does not inherit your shell environment
and therefore does not find `uv` on the `PATH`. `command -v uv` prints the path. Restart both
clients afterwards, they only load new servers on start.

Finally, allow Chrome to be driven. Once: open `chrome://inspect/#remote-debugging` and tick
"Allow remote debugging for this browser instance". After that, Chrome asks once per browser
session whether a connection may be made. See Troubleshooting for the reliable way to answer it.

## The three tools

### `browser_task(url, goals, max_actions, time_budget_s, allow_domains, dry_run)`

Operates a site autonomously: it opens `url` and pursues `goals`, deciding every click, selection
and keystroke without asking.

| Parameter | Default | Limits |
|---|---|---|
| `url` | required | at most 2048 characters |
| `goals` | required | a sentence or a list of them, at most 20 entries of 2000 characters each |
| `max_actions` | 25 | 1 to 60, the library's own ceiling |
| `time_budget_s` | 120 | 1 to 900, wall clock |
| `allow_domains` | none | at most 50 entries of 253 characters each |
| `dry_run` | false | observes the page, fetches the first decision, executes nothing |

The result reports the status, the final URL, the page title, every executed step, the duration,
the actions and model calls used, the reason it stopped and, for a dry run, the step it would have
taken next.

### `browser_status()`

Read-only diagnosis, and free: it opens no browser, loads no page and calls no model. It says which
keys were found and where they came from, which text model would type, whether the browser-harness
daemon is running and whether a Chrome is connected to it, and which operations are blocked as a
result. Call it before the first run of a session and whenever one of the other two tools fails.

### `browser_read(url, time_budget_s, allow_domains, text_limit)`

Opens one page and returns its visible text plus the table of operable elements, without clicking,
typing or selecting anything.

| Parameter | Default | Limits |
|---|---|---|
| `url` | required | at most 2048 characters |
| `time_budget_s` | 30 | 1 to 900 |
| `allow_domains` | none | at most 50 entries |
| `text_limit` | 4000 | 200 to 20000 characters |

The answer says how long the text was before truncation, and it lists at most 120 elements. A
redirect onto another registrable domain is reported, and nothing from that page is returned.

## Limits, stated up front

The agent underneath is an MVP, and these gaps are real:

* It cannot see into iframes, and it cannot see into shadow DOM.
* It cannot upload files.
* It cannot follow pop-up tabs.
* It only observes the visible viewport. It scrolls, but it never reads what is not rendered.
* Only one run can be in flight at a time. A second call is refused immediately with `not_started`
  instead of being queued, because a run writes process-wide values into `os.environ` and two runs
  would overwrite each other.
* A call blocks until the run ends.
* The agent works inside your real, logged-in Chrome profile. Whatever it clicks happens in your
  sessions, for real. Each run opens its own background window and closes it afterwards. It does
  not use a background tab, because Chrome 153 answers no command sent to a background tab created
  over the DevTools protocol, while a background window works and leaves your own window alone.
* Do not put credentials into `goals`. The goal text comes back verbatim in the result, and so does
  every value the agent typed into a field. Nothing of that is masked, on purpose: the typed text is
  the most important record of what the agent actually did. Password, file and hidden fields are
  excluded from observation, so nothing is typed there, but a one-time code or a customer number in
  an ordinary text field is not covered by that and will appear in the result.
* Logging in, paying, ordering and submitting forms on banking or payment sites are out of scope for
  this project.

## The safeguard: a domain lock instead of a blocklist

There is no blocklist. That was a deliberate decision: the agent only ever runs on request, so there
are no forbidden domains. Instead a run remembers the registrable domain of its start URL. If a
click or a redirect would take it to a different registrable domain, it stops and reports where it
wanted to go, rather than acting there. Subdomains of the same registrable domain count as the same
domain.

The check runs at two moments. Before a navigation it uses the address the next step would open, and
that is the check that actually protects you, because in a logged-in profile the loaded page is
already the damage: it has seen cookies, run scripts and sent requests. After every load it uses the
address the browser actually ended up on, which catches redirects, `window.location` from a script
and clicks the agent did not recognise as navigation. A block at that second moment means the damage
has already happened, and the run aborts instead of continuing.

When the module cannot read its own input reliably, it blocks. An unreadable start URL, a host with
characters outside the permitted set, an invalid port, a numeric address that does not parse: all of
those end in a block, never in a pass.

What it explicitly does not do:

* It does not judge content. What an allowed page whispers to the agent is invisible to it, and
  page content can steer the decision model.
* It knows no confirmation prompt and no step limit. Those live in the runner.
* Before a click it can only read a target address from a real link (`a[href]`). A `<button>`, a
  form submission or a click that only a script turns into navigation contributes nothing, and there
  the check first bites after the load.

Known weakness with hosting suffixes: the module does not use the Public Suffix List, which would be
a dependency. It applies the rule "the last two labels", extended by a built-in set of multi-part
suffixes that covers the common country suffixes and the common hosting suffixes such as `github.io`
and `vercel.app`. A multi-part suffix that is missing from that set, `blogspot.de` for example, is
read too loosely, and two unrelated sites under the same provider then count as the same domain.
The opposite error, a three-label host whose middle label happens to look like a suffix, makes the
check too strict, and the agent stops although it would have been allowed.

The check can be relaxed in four places: `allow_domains` per call, `allow_domains` globally in
`~/.config/jev-mcp/policy.toml`, the switch `enforce_domain_lock = false` in the same file, and
`allow_unbound` for runs that are meant to start without a domain binding.

## Choosing the text model

Measured on 2026-09-20 through a real `jev_ultrafast.model.field_text()` call, filling the field
"Where from?" in a flight search. All four runs answered correctly ("Zurich"):

| Model | `reasoning=low` | `reasoning=none` |
|---|---|---|
| `kimi-k3` | 5855 ms | 5619 ms |
| `kimi-k2.7-code-highspeed` | 1156 ms | 1138 ms |

K3 costs five times as much latency, and it costs it per typed field. K3 is the default anyway,
by explicit decision. Switching is one line in `~/.config/jev-mcp/env`, and the provider stays the
same:

```
TEXT_MODEL=kimi-k2.7-code-highspeed
```

## Troubleshooting

Call `browser_status` first. It names the missing key, the missing daemon or the missing Chrome
connection and says what is blocked because of it, which is faster than guessing.

If it reports that the browser-harness daemon is not running, or that it runs but no Chrome is
connected to it, `browser-harness --doctor` diagnoses install, daemon and browser state.

Chrome asks for approval once per browser session. The reliable way to answer it: bring a normal
Chrome window to the front, not in full screen, then run the command below and click Allow in the
dialog. It waits without a time limit.

```
cd /absolute/path/to/jev-mcp && echo 'print(page_info())' | uv run browser-harness
```

If the connection ends before approval and Chrome never asked, Chrome is most likely running in
the background without a window. That happens when "Continue running background apps when Google
Chrome is closed" is on: closing Chrome leaves it running, and an approval dialog with no window to
appear in is silently dropped. Quit Chrome fully, start it with a normal window, and try again.
Turning that setting off (Settings, System) avoids the trap for good.

`browser-harness mac-approve` can click the dialog for you on macOS, but only when Chrome's
interface is in English: it matches the English dialog text. It also needs the Accessibility
permission for your terminal.

If a run is refused with `not_started` and a note about concurrency, another run is still in flight.
Wait for it and start again.

## Development

```
uv run pytest
uv run ruff check .
```

461 tests pass as of 2026-09-21. The suite runs without a browser and without network access,
against test doubles and against contract tests that pin every assumption about jev-ultrafast to
its installed source. An end-to-end run against real Chrome 153 through the server as a separate
stdio process passed on 2026-09-21.

## Built on

jev-mcp is a thin, guarded layer. The actual work is done by these projects:

| Project | What it does here | License |
|---|---|---|
| [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) by Browser Use | The browser agent itself: the observe, decide, act loop | MIT |
| [browser-harness](https://github.com/browser-use/browser-harness) by Browser Use | Connects to Chrome over the DevTools protocol and keeps the session | MIT |
| [TypeSafe Jev](https://typesafe.ai) ([docs](https://docs.typesafe.ai)) | The decision model that picks every action and target | commercial API |
| [Model Context Protocol](https://modelcontextprotocol.io) | The protocol Claude Code and Claude Desktop speak to this server | see [repository](https://github.com/modelcontextprotocol/modelcontextprotocol) |
| [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) | Server implementation and stdio transport | MIT |
| [uv](https://github.com/astral-sh/uv) by Astral | Environment and dependency management | see repository |
| [WHATWG URL Standard](https://url.spec.whatwg.org) | The reference for how URLs are read, so the safeguard reads them the way Chrome does | CC BY 4.0 |

The text that gets typed into fields comes from an OpenAI-compatible model of your choice. The
server knows [Moonshot Kimi](https://platform.kimi.ai), [DeepSeek](https://platform.deepseek.com)
and [OpenRouter](https://openrouter.ai) by name and works with any other endpoint through
`TEXT_MODEL_API_KEY` and `TEXT_MODEL_BASE_URL`.

Related: [typesafe-mcp](https://github.com/itsmostafa/typesafe-mcp) (MIT) exposes the same Jev
model as a general judgment tool, typed answers with probabilities, for any MCP client.

## Security

This server drives a real, logged-in browser, and its only safeguard is the domain check described
above. If you find a way around it, please report it privately to hello@saits.ai instead of opening
a public issue.

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2026 R. Schröder GmbH (SAITS - Swiss AI Tech Services).
