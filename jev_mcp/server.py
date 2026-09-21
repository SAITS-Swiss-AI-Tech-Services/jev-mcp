"""Die MCP-Schicht: drei Werkzeuge über stdio.

Dieser Modul ist dünn. Er entscheidet nichts über Browser, Domains oder
Budgets, das tun `config.py`, `guards.py` und `runner.py`. Er tut vier Dinge:

1. Er wendet die Umgebung **genau einmal** an. `apply_environment()` schreibt
   prozessweit in `os.environ`, und der Prozess ist hier der Server, nicht der
   einzelne Aufruf. Das Ergebnis wird festgehalten und an jeden Lauf
   weitergereicht. Ist es nicht in Ordnung, steht das in **jeder** Antwort,
   nicht nur in der ersten, denn ein Modell liest selten die erste.
2. Er prüft die Vorgaben, bevor sie weitergehen. Alles kommt als JSON von einem
   Modell. `run_task()` fängt zwar jede Fehleingabe ab, aber ein klarer Satz an
   der Tür ist besser als ein Ergebnis mit `not_started`, das erst gelesen
   werden muss.
3. Er macht aus jedem Ergebnis eine Struktur, die sich als UTF-8 senden lässt.
   Ein `NaN` in einer Zuversicht der Bibliothek ergäbe sonst `NaN` im JSON, und
   das ist kein gültiges JSON. Ein einsames Surrogat aus `os.environ` liesse
   pydantic beim Serialisieren werfen, und zwar erst nach der Rückgabe des
   Werkzeugs, wo es niemand mehr erklären kann.
4. Er hilft, die Standardausgabe frei zu halten. Bei stdio ist sie der
   Protokollkanal, und alles, was sonst dorthin geht, zerstört die Verbindung.

Zur Standardausgabe im Einzelnen, ehrlich gerechnet
---------------------------------------------------
Der eigentliche Riegel gehört nicht diesem Projekt. Das SDK (mcp 2.2.0,
`mcp/server/stdio.py`) lenkt den Dateideskriptor 1 während des Betriebs auf
stderr um und bedient die Leitung aus einer privaten Kopie. Ein `print`, ein
`os.write(1, ...)` und selbst ein Unterprozess landen dadurch bereits dort, wo
sie hingehören. Fällt dieser fremde Riegel weg, etwa in einer Umgebung, in der
`sys.stdout` gar nicht auf Deskriptor 1 liegt, greifen nur noch zwei
Teilabdeckungen:

* `runner.ohne_stdout()` legt `sys.stdout` auf stderr, solange ein
  Werkzeugaufruf oder ein Arbeitsfaden läuft. Das fasst `sys.stdout` an, nicht
  den Deskriptor: ein `os.write(1, ...)` und ein Unterprozess gehen daran
  vorbei. Der Riegel liegt bewusst auch im Arbeitsfaden, denn der überlebt das
  Zeitbudget und damit den Werkzeugaufruf.
* `main()` legt das Protokollieren ausdrücklich auf stderr.

Ein `print` steht in diesem Modul nicht, und zwei Tests halten die Zusage gegen
die echte Leitung: einer im Prozess für den Faden, der das Budget überlebt,
einer als Unterprozess über echtes stdio.

Die Werkzeugbeschreibungen sind englisch, weil sie ein Modell liest und nicht
ein Mensch. Die Rückgabetexte sind deutsch, weil sie am Ende beim Nutzer
ankommen. Beides ist Absicht.
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

MAX_URL_LAENGE = 2048
MAX_ZIELE = 20
MAX_ZIEL_LAENGE = 2000
MAX_ALLOW_DOMAINS = 50
MAX_DOMAIN_LAENGE = 253

UMGEBUNG_WARNUNG = (
    "Die Umgebung für den Browser-Agenten liess sich nicht setzen, deshalb kann kein Lauf starten. "
    "Unter «environment» steht, woran es liegt."
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
    "page's text, elements, title or address is returned; widen that with `allow_domains`. The "
    "underlying library observes at most 6000 characters of visible text and cuts a longer page "
    "before `text_limit` (default 4000) applies, so the answer reports how long the observed text "
    "was and warns when it reached that ceiling. Labels and field values are capped at 200 "
    "characters and option lists at 50 entries per element, and the answer says when it capped "
    "something. Same limits as the agent: no iframes, no shadow DOM, no file uploads, no pop-up "
    "tabs, and only the visible viewport is observed, so content further down the page can be "
    "missing. Only one run can be in flight at a time."
)


# ---------------------------------------------------------------------------
# Die Umgebung, genau einmal
# ---------------------------------------------------------------------------

_UMGEBUNG: EnvironmentApplication | None = None
_UMGEBUNG_SCHLOSS = Lock()
_SCHNAPPSCHUSS: dict[str, str] | None = None
_KONFIG_STAND: tuple[object, ...] | None = None


def _konfig_stand() -> tuple[object, ...]:
    """Ein Abbild der Konfigurationsdatei, an dem eine Änderung erkennbar ist.

    Gibt `(vorhanden, Änderungszeit, Grösse)` zurück. Fehlt die Datei oder
    lässt sie sich nicht abfragen, steht das ebenfalls darin, statt dass hier
    etwas wirft.
    """
    try:
        zustand = DEFAULT_CONFIG_PATH.stat()
    except OSError:
        return (False, 0, 0)
    return (True, zustand.st_mtime_ns, zustand.st_size)


def environment() -> EnvironmentApplication:
    """Das einmal angewandte Ergebnis von `apply_environment()`.

    Beim ersten Zugriff wird die Umgebung gesetzt, danach wird nur noch
    nachgesehen. `apply_environment()` wirft nicht, ein Fehlschlag steht in
    `ok` und in `notes`.

    Vor dem Anwenden wird die Prozessumgebung abgelichtet. Die Diagnose löst
    später gegen dieses Abbild auf und nicht gegen die veränderte Umgebung,
    sonst läse sie ihre eigene Tat und meldete jeden Wert aus der
    Konfigurationsdatei als «aus der Umgebung».
    """
    global _UMGEBUNG, _SCHNAPPSCHUSS, _KONFIG_STAND
    with _UMGEBUNG_SCHLOSS:
        if _UMGEBUNG is None:
            _SCHNAPPSCHUSS = dict(os.environ)
            _KONFIG_STAND = _konfig_stand()
            _UMGEBUNG = apply_environment()
        return _UMGEBUNG


NEUSTART_HINWEIS = (
    "Die Konfigurationsdatei hat sich seit dem Start dieses Servers geändert. Es gilt weiterhin, "
    "was beim Start galt: die Umgebung wird genau einmal gesetzt. Starte den Server neu, damit die "
    "Änderung wirkt."
)


def _herkunft_hinweise(umgebung: EnvironmentApplication) -> list[str]:
    """Was zur Herkunft der Schlüssel in jede Diagnose gehört.

    Die Diagnose löst gegen den Schnappschuss von vor dem Start auf. Dass sie
    nicht ihre eigene Tat liest, gehört trotzdem dazu gesagt, denn in der
    laufenden Prozessumgebung stehen die Werte jetzt sehr wohl.
    """
    hinweise: list[str] = []
    if umgebung.applied:
        namen = ", ".join(umgebung.applied)
        hinweise.append(
            f"Diese Variablen hat dieser Server beim Start selbst gesetzt: {namen}. Die Herkunft "
            "unten ist deshalb gegen die Umgebung von vor dem Start aufgelöst, nicht gegen die "
            "jetzige Prozessumgebung."
        )
    if _KONFIG_STAND is not None and _konfig_stand() != _KONFIG_STAND:
        hinweise.append(NEUSTART_HINWEIS)
    return hinweise


# ---------------------------------------------------------------------------
# Eingaben sind fremde Daten
# ---------------------------------------------------------------------------


def _adresse(wert: object) -> str:
    """Prüft die Startadresse eines Werkzeugaufrufs."""
    text = wert.strip() if isinstance(wert, str) else ""
    if not text:
        raise ToolError(
            "Es wurde keine Adresse angegeben. Nenne die Seite, auf der begonnen werden soll, zum "
            "Beispiel https://example.com/kontakt."
        )
    if len(text) > MAX_URL_LAENGE:
        raise ToolError(
            f"Die Adresse ist mit {len(text)} Zeichen länger als die erlaubten {MAX_URL_LAENGE} Zeichen."
        )
    schema = text.split(":", 1)[0].lower() if ":" in text else ""
    if schema not in {"http", "https"}:
        raise ToolError(
            "Die Adresse muss mit http:// oder https:// beginnen. Andere Schemata öffnet dieser Server nicht."
        )
    return text


def _ziele(wert: object) -> tuple[list[str], list[str]]:
    """Macht aus der Zielangabe eine Liste brauchbarer Sätze samt Hinweisen.

    Ein leeres Ziel wird weiterhin verworfen, aber nicht mehr stillschweigend:
    aus `["Suche die Seite", ""]` wurde ein Lauf mit einem Ziel, ohne dass das
    irgendwo stand.
    """
    if isinstance(wert, str):
        roh: list[object] = [wert]
    elif isinstance(wert, Sequence) and not isinstance(wert, bytes | bytearray):
        roh = list(wert)
    else:
        raise ToolError(
            "Die Ziele müssen ein Satz oder eine Liste von Sätzen sein, zum Beispiel "
            '["Finde die Kontaktseite", "Lies die Telefonnummer vor"].'
        )
    if any(not isinstance(eintrag, str) for eintrag in roh):
        raise ToolError("Jedes Ziel muss ein Satz in Worten sein, keine Zahl und kein Objekt.")
    ziele = [str(eintrag).strip() for eintrag in roh if str(eintrag).strip()]
    if not ziele:
        raise ToolError(
            "Es wurde kein Ziel angegeben. Schreibe in Worten, was auf der Seite geschehen soll, "
            'zum Beispiel "Finde die Kontaktseite und lies die Telefonnummer".'
        )
    if len(ziele) > MAX_ZIELE:
        raise ToolError(
            f"Es wurden {len(ziele)} Ziele angegeben, erlaubt sind höchstens {MAX_ZIELE}. Teile den "
            "Auftrag in mehrere Läufe."
        )
    zu_lang = next((ziel for ziel in ziele if len(ziel) > MAX_ZIEL_LAENGE), None)
    if zu_lang is not None:
        raise ToolError(
            f"Ein Ziel ist mit {len(zu_lang)} Zeichen länger als die erlaubten {MAX_ZIEL_LAENGE} "
            "Zeichen. Fasse es kürzer."
        )
    verworfen = len(roh) - len(ziele)
    hinweise = (
        [
            f"Von den angegebenen Zielen waren {verworfen} leer. Sie wurden verworfen, gelaufen "
            f"wird mit den übrigen {len(ziele)}."
        ]
        if verworfen
        else []
    )
    return ziele, hinweise


def _domains(wert: object) -> tuple[list[str] | None, list[str]]:
    """Prüft `allow_domains`. Eine einzelne Zeichenkette gilt als ein Eintrag.

    Leere Einträge werden verworfen, und auch das wird gesagt, statt dass die
    Liste stillschweigend kürzer wird.
    """
    if wert is None:
        return None, []
    if isinstance(wert, str):
        roh: list[object] = [wert]
    elif isinstance(wert, Sequence) and not isinstance(wert, bytes | bytearray):
        roh = list(wert)
    else:
        raise ToolError(
            "allow_domains muss eine Domain oder eine Liste von Domains sein, zum Beispiel "
            '["example.com", "*.example.net"].'
        )
    if any(not isinstance(eintrag, str) for eintrag in roh):
        raise ToolError("Jede Domain in allow_domains muss eine Zeichenkette sein.")
    domains = [str(eintrag).strip() for eintrag in roh if str(eintrag).strip()]
    if len(domains) > MAX_ALLOW_DOMAINS:
        raise ToolError(
            f"Es wurden {len(domains)} Domains angegeben, erlaubt sind höchstens {MAX_ALLOW_DOMAINS}."
        )
    zu_lang = next((domain for domain in domains if len(domain) > MAX_DOMAIN_LAENGE), None)
    if zu_lang is not None:
        raise ToolError(f"Ein Eintrag in allow_domains ist mit {len(zu_lang)} Zeichen keine Domain.")
    verworfen = len(roh) - len(domains)
    hinweise = (
        [f"Von den Einträgen in allow_domains waren {verworfen} leer und wurden verworfen."]
        if verworfen
        else []
    )
    return domains or None, hinweise


def _zahl(name: str, wert: object, *, minimum: float, maximum: float) -> float:
    """Prüft eine Zahl auf endlich und im erlaubten Bereich.

    `True` und `False` sind in Python Zahlen, und `float(True)` ist eine glatte
    Eins. `max_actions: true` ergab dadurch kommentarlos einen Lauf mit einer
    einzigen Aktion, `time_budget_s: true` ein Budget von einer Sekunde. Beides
    liegt im erlaubten Bereich und fiel deshalb nirgends auf.
    """
    if isinstance(wert, bool):
        raise ToolError(
            f"{name} muss eine Zahl zwischen {minimum:g} und {maximum:g} sein, kein Wahrheitswert."
        )
    try:
        zahl = float(wert)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        raise ToolError(f"{name} muss eine Zahl zwischen {minimum:g} und {maximum:g} sein.") from None
    if not math.isfinite(zahl):
        raise ToolError(f"{name} muss eine endliche Zahl zwischen {minimum:g} und {maximum:g} sein.")
    if zahl < minimum or zahl > maximum:
        raise ToolError(
            f"{name} liegt mit {zahl:g} ausserhalb des erlaubten Bereichs von {minimum:g} bis {maximum:g}."
        )
    return zahl


def _ganze_zahl(name: str, wert: object, *, minimum: int, maximum: int) -> int:
    """Wie `_zahl`, gibt aber eine ganze Zahl zurück."""
    return int(_zahl(name, wert, minimum=minimum, maximum=maximum))


# ---------------------------------------------------------------------------
# Antworten
# ---------------------------------------------------------------------------


def _sendbar(wert: object) -> str:
    """Macht aus einem Text einen, der sich als UTF-8 senden lässt.

    Das SDK serialisiert mit pydantic, und pydantic schreibt unmittelbar UTF-8.
    Ein einsames Surrogat darin wirft, und zwar erst **nach** der Rückgabe des
    Werkzeugs: der Client sah dann nur «Error executing tool browser_status»
    ohne Begründung, ausgerechnet bei dem Werkzeug, das Fehler erklären soll.
    Auf POSIX liefert `os.environ` genau solche Zeichen für ein ungültiges Byte
    in einer Variablen, das ist also kein theoretischer Fall.
    """
    return str(wert).encode("utf-8", "replace").decode("utf-8")


def _json_tauglich(wert: object) -> Any:
    """Macht aus einem Ergebnis etwas, das `json.dumps(allow_nan=False)` übersteht.

    Tupel werden Listen, Aufzählungen werden ihr Text, `NaN` und die
    Unendlichkeiten werden `None`, und alles Unbekannte wird sein `str()`. Die
    Schritte eines Laufs kommen aus der Bibliothek, dort kann auch etwas stehen,
    das der JSON-Kodierer nicht kennt.
    """
    if wert is None or isinstance(wert, bool):
        return wert
    if isinstance(wert, str):
        return _sendbar(wert)
    if isinstance(wert, int):
        return int(wert)
    if isinstance(wert, float):
        return float(wert) if math.isfinite(wert) else None
    if isinstance(wert, Mapping):
        return {_sendbar(str(schluessel)): _json_tauglich(inhalt) for schluessel, inhalt in wert.items()}
    if isinstance(wert, list | tuple | set | frozenset):
        return [_json_tauglich(inhalt) for inhalt in wert]
    return _sendbar(runner.sicherer_text(wert))


def _antwort(ergebnis: object) -> dict[str, Any]:
    """Macht aus einem Ergebnis die Antwort des Werkzeugs, samt Umgebungszustand.

    Der Zustand der Umgebung steht in jeder Antwort, nicht nur in der ersten.
    Ist er nicht in Ordnung, kippt ausserdem `ok` beziehungsweise `ready`, denn
    dann läuft nichts, egal was sonst in der Antwort steht.
    """
    daten = _json_tauglich(dataclasses.asdict(ergebnis))  # type: ignore[call-overload]
    umgebung = environment()
    daten["environment"] = _json_tauglich(dataclasses.asdict(umgebung))
    if not umgebung.ok:
        hinweise = list(daten.get("notes") or [])
        if UMGEBUNG_WARNUNG not in hinweise:
            hinweise.insert(0, UMGEBUNG_WARNUNG)
        daten["notes"] = hinweise
        if "ok" in daten:
            daten["ok"] = False
        if "ready" in daten:
            daten["ready"] = False
    return daten


def _ohne_stdout() -> Iterator[None]:
    """Der Riegel für die Standardausgabe, für die Dauer eines Werkzeugaufrufs.

    Der Riegel selbst steht in `runner.ohne_stdout()`, denn der Arbeitsfaden
    eines Laufs braucht ihn ebenfalls und überlebt den Aufruf. Beide Halter
    zählen mit, damit die Ausgabe erst zurückgelegt wird, wenn der letzte
    loslässt.
    """
    return runner.ohne_stdout()


def _mit_umgebungswarnung(fehler: ToolError) -> ToolError:
    """Hängt an einen Werkzeugfehler den Zustand der Umgebung, wenn er nicht stimmt.

    Der Modul-Docstring sagt zu, dass eine kaputte Umgebung in **jeder** Antwort
    steht. Eine geworfene `ToolError` ist auch eine Antwort, und dort fehlte der
    Satz bisher.
    """
    text = str(fehler)
    if environment().ok or UMGEBUNG_WARNUNG in text:
        return fehler
    return ToolError(f"{text} {UMGEBUNG_WARNUNG}")


def _mit_hinweisen(daten: dict[str, Any], hinweise: Sequence[str]) -> dict[str, Any]:
    """Hängt Hinweise der Werkzeugschicht an die Antwort, jeden Wortlaut nur einmal."""
    if not hinweise:
        return daten
    vorhanden = list(daten.get("notes") or [])
    for hinweis in hinweise:
        if hinweis not in vorhanden:
            vorhanden.append(hinweis)
    daten["notes"] = vorhanden
    return daten


def _unerwartet(name: str, fehler: BaseException) -> ToolError:
    """Macht aus einem unerwarteten Fehler einen lesbaren Werkzeugfehler."""
    text = runner.kurzfassung(runner.sicherer_text(fehler) or "ohne Text")
    return ToolError(
        f"Das Werkzeug {name} ist unerwartet gescheitert ({type(fehler).__name__}: {text}). Der "
        "Server läuft weiter, versuche es erneut oder frage zuerst browser_status."
    )


# ---------------------------------------------------------------------------
# Die drei Werkzeuge
# ---------------------------------------------------------------------------


def browser_task(
    url: str,
    goals: list[str] | str,
    max_actions: int = runner.DEFAULT_MAX_ACTIONS,
    time_budget_s: float = runner.DEFAULT_TIME_BUDGET_S,
    allow_domains: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Der autonome Lauf. Siehe `BROWSER_TASK_DESCRIPTION` für den Text, den ein Modell liest."""
    try:
        adresse = _adresse(url)
        saetze, ziel_hinweise = _ziele(goals)
        aktionen = _ganze_zahl("max_actions", max_actions, minimum=1, maximum=runner.LIBRARY_MAX_ACTIONS)
        zeit = _zahl("time_budget_s", time_budget_s, minimum=1.0, maximum=runner.MAX_TIME_BUDGET_S)
        domains, domain_hinweise = _domains(allow_domains)
        with _ohne_stdout():
            ergebnis = runner.run_task(
                adresse,
                saetze,
                max_actions=aktionen,
                time_budget_s=zeit,
                allow_domains=domains,
                dry_run=bool(dry_run),
                environment=environment(),
            )
        return _mit_hinweisen(_antwort(ergebnis), [*ziel_hinweise, *domain_hinweise])
    except ToolError as fehler:
        raise _mit_umgebungswarnung(fehler) from None
    except Exception as fehler:  # noqa: BLE001
        raise _mit_umgebungswarnung(_unerwartet("browser_task", fehler)) from fehler


def browser_status() -> dict[str, Any]:
    """Die Diagnose ohne Nebenwirkung. Siehe `BROWSER_STATUS_DESCRIPTION`."""
    try:
        umgebung = environment()
        with _ohne_stdout():
            befund = diagnose(env=_SCHNAPPSCHUSS)
        return _mit_hinweisen(_antwort(befund), _herkunft_hinweise(umgebung))
    except ToolError as fehler:
        raise _mit_umgebungswarnung(fehler) from None
    except Exception as fehler:  # noqa: BLE001
        raise _mit_umgebungswarnung(_unerwartet("browser_status", fehler)) from fehler


def browser_read(
    url: str,
    time_budget_s: float = runner.DEFAULT_READ_TIME_BUDGET_S,
    allow_domains: list[str] | None = None,
    text_limit: int = runner.DEFAULT_TEXT_LIMIT,
) -> dict[str, Any]:
    """Eine Seite lesen, ohne zu handeln. Siehe `BROWSER_READ_DESCRIPTION`."""
    try:
        adresse = _adresse(url)
        zeit = _zahl("time_budget_s", time_budget_s, minimum=1.0, maximum=runner.MAX_TIME_BUDGET_S)
        grenze = _ganze_zahl(
            "text_limit",
            text_limit,
            minimum=runner.MIN_TEXT_LIMIT,
            maximum=runner.MAX_TEXT_LIMIT,
        )
        domains, domain_hinweise = _domains(allow_domains)
        with _ohne_stdout():
            ergebnis = runner.read_page(
                adresse,
                time_budget_s=zeit,
                allow_domains=domains,
                text_limit=grenze,
                environment=environment(),
            )
        return _mit_hinweisen(_antwort(ergebnis), domain_hinweise)
    except ToolError as fehler:
        raise _mit_umgebungswarnung(fehler) from None
    except Exception as fehler:  # noqa: BLE001
        raise _mit_umgebungswarnung(_unerwartet("browser_read", fehler)) from fehler


# ---------------------------------------------------------------------------
# Der Server
# ---------------------------------------------------------------------------


def _version() -> str:
    try:
        return version("jev-mcp")
    except PackageNotFoundError:  # Nur ohne installiertes Paket.
        return "0.0.0"


def create_server() -> MCPServer:
    """Baut den Server und meldet die drei Werkzeuge an.

    Die Umgebung wird hier angewandt, also einmal beim Start und nicht bei jedem
    Aufruf.
    """
    environment()
    server = MCPServer(name=SERVER_NAME, version=_version(), instructions=INSTRUCTIONS)
    server.add_tool(browser_task, name="browser_task", description=BROWSER_TASK_DESCRIPTION)
    server.add_tool(browser_status, name="browser_status", description=BROWSER_STATUS_DESCRIPTION)
    server.add_tool(browser_read, name="browser_read", description=BROWSER_READ_DESCRIPTION)
    return server


_protokoll = logging.getLogger(SERVER_NAME)


def _protokoll_auf_stderr(level: int = logging.INFO) -> None:
    """Legt alle Protokollausgaben auf stderr.

    Bei stdio ist die Standardausgabe der Protokollkanal. Eine Bibliothek, die
    dorthin protokolliert, zerstört die Verbindung, und der Client sieht nur,
    dass nichts mehr geht.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)


def main() -> None:
    """Der Konsolenbefehl `jev-mcp`: startet den Server über stdio.

    Geht der Client weg, ist das kein Fehler, sondern das Ende. Ohne diese
    Behandlung endete der Server mit einem Traceback und Exitcode 1, und in
    Claude Desktop sah das aus wie ein Absturz. Das SDK bündelt Ausnahmen aus
    seinen Aufgabengruppen, deshalb wird auch die Gruppe aufgetrennt.
    """
    _protokoll_auf_stderr()
    try:
        create_server().run("stdio")
    except (BrokenPipeError, KeyboardInterrupt):
        _protokoll.info("Der Client hat die Leitung geschlossen, der Server endet.")
    except BaseExceptionGroup as gruppe:
        _, rest = gruppe.split((BrokenPipeError, KeyboardInterrupt))
        if rest is not None:
            raise rest from None
        _protokoll.info("Der Client hat die Leitung geschlossen, der Server endet.")


if __name__ == "__main__":  # Der Einstieg als Modul.
    main()
