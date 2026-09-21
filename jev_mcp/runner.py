"""Ein Lauf des Browser-Agenten: begrenzt, überwacht und in Worten erklärt.

Diese Schicht liegt zwischen der MCP-Werkzeugschicht und `jev_ultrafast`. Sie
tut vier Dinge, die die Bibliothek nicht tut:

1. Sie wendet die Umgebung an, bevor irgendetwas läuft. `jev_ultrafast/model.py`
   liest seine Variablen erst beim Aufruf aus `os.environ`, und seine eingebaute
   Vorgabe zeigt auf DeepSeek. Ohne `apply_environment()` liefe ein fremder
   Schlüssel still gegen den falschen Endpunkt. Meldet die Anwendung `ok=False`,
   startet kein Lauf, und die Hinweise stehen in der Antwort.
2. Sie hängt die Domain-Treue aus `guards.py` an beide vorgesehenen Zeitpunkte.
3. Sie begrenzt den Lauf zweifach: in Aktionen und in Wanduhrzeit. Die
   Bibliothek kennt nur das Aktionsbudget, und ein einzelner Schritt kann hängen.
4. Sie übersetzt die knappen englischen Ausnahmen der Bibliothek in ganze
   deutsche Sätze, die sagen, was passiert ist und was man tun kann.

Die Zieladresse vor dem Klick: was die Bibliothek hergibt
---------------------------------------------------------
Nachgesehen am 20.09.2026 in `jev_ultrafast/snapshot.js` und `browser.py`.

Die Elementtabelle, die das Entscheidungsmodell sieht (`page["actions"]`,
gebaut in `snapshot.js` Zeile 61), trägt **keine** Zieladresse. Sie enthält
`node`, `role`, `label`, `kind` und `value`, mehr nicht.

Die Zieladresse gibt es trotzdem, an einer anderen Stelle: `cache.guard()` in
`snapshot.js` Zeile 47 bis 54 legt für jedes Element ein Tupel aus vierzehn
Feldern an, und an Position 12 steht `e.getAttribute('href')`. Diese Tupel
kommen als `page["guards"]` mit, wo `browser.py` Zeile 97 sie zur
Veraltet-Prüfung benutzt. Wir lesen sie dort mit, siehe `planned_target_url()`.

Der Schutz greift damit vor dem Klick, aber nicht lückenlos. Ehrlich gesagt:

* Nur `a[href]` liefert eine Adresse. Ein `<button>`, ein Formular-Absenden oder
  ein Klick, den erst ein Skript zur Navigation macht, trägt nichts bei. Dort
  greift die Domain-Treue erst nach dem Laden, und im eingeloggten Profil ist
  die geladene Seite bereits der Schaden.
* `href="javascript:..."` sagt nicht, wohin es geht. Wir prüfen es nicht als
  Navigation, sonst hielte jeder gewöhnliche Schaltflächen-Link den Lauf an,
  und vermerken es stattdessen als Hinweis.
* Die Position 12 ist eine Stelle in einem unbenannten Tupel der Bibliothek.
  Ändert sich dort die Reihenfolge, lesen wir das falsche Feld. Deshalb prüfen
  wir zweierlei: die Länge des Tupels, die Einfügen und Entfernen fängt, und die
  Form des gelesenen Werts, die ein Vertauschen fängt. Direkt neben der Adresse
  steht der Text der Umgebung des Elements, und daraus würde sonst eine
  plausible Adresse auf der eigenen Domain. Beides endet in einem Hinweis, nie
  in einem stillen Durchwinken. Ein Vertragstest hält Länge und Position
  ausserdem gegen die installierte `snapshot.js`.

Wie Adressen gelesen werden
---------------------------
Es gibt im ganzen Projekt genau eine Lesart von Adressen, und sie steht in
`guards.py`. Dieser Modul löst deshalb nie selbst auf, sondern ruft
`guards.resolve_url()`. Der Grund steht dort ausführlich: `urllib.parse.urljoin`
liest den Backslash als gewöhnliches Zeichen, der Browser macht einen
Schrägstrich daraus, und wer die erste Lesart prüft und die zweite ausführt,
prüft die falsche Adresse.

Was im Ergebnis steht, und was das über Geheimnisse heisst
----------------------------------------------------------
`RunResult` ist über `dataclasses.asdict()` ohne Sonderbehandlung in JSON zu
bringen. Es entsteht aus einer festen Liste von Feldern, nie aus einem
durchgereichten Zustand der Bibliothek. Schlüssel aus der Umgebung oder aus der
Konfiguration stehen deshalb nicht darin.

Zwei Felder tragen aber sehr wohl Inhalt, den der Auftrag hervorgebracht hat,
und das ist Absicht:

* `RunResult.goals` trägt den Auftragstext wörtlich, so wie er hereinkam.
* `StepRecord.text` trägt jeden Wert, den der Agent in ein Feld getippt hat.

`snapshot.js` Zeile 9 nimmt Felder der Art `password`, `file` und `hidden` von
der Beobachtung aus, dort wird also nichts getippt. Ein Einmalkennwort, eine
Kundennummer oder eine Ausweisnummer in einem gewöhnlichen Textfeld ist davon
nicht gedeckt und steht anschliessend im Ergebnis.

Bewusst wird nichts davon maskiert. Der getippte Text ist die wichtigste
Angabe, um einen Lauf nachzuvollziehen, und das Textmodell erfindet keine
Zugangsdaten: es kann nur tippen, was aus dem Auftrag folgt. Wer keinen
vertraulichen Wert im Ergebnis haben will, schreibt ihn nicht in den Auftrag.
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
    "eigenes_fenster",
    "installiere_eigenes_fenster",
    "kurzfassung",
    "ohne_stdout",
    "planned_target_url",
    "read_page",
    "run_task",
    "sicherer_text",
    "translate_error",
    "wait_until_idle",
]

LIBRARY_MAX_ACTIONS = 60
"""Die Obergrenze der Bibliothek, `jev_ultrafast.questions.MAX_STEPS`.

Der Wert ist hier festgeschrieben, damit dieser Modul ohne den Browser-Harness
ladbar bleibt. Ein Vertragstest hält ihn gegen die installierte Bibliothek.
"""

LIBRARY_MAX_MODEL_CALLS = LIBRARY_MAX_ACTIONS * 2
"""Das Modellaufruf-Budget der Bibliothek, `MAX_STEPS * 2` in `agent.py`."""

DEFAULT_MAX_ACTIONS = 25
"""Vorgabe für einen Lauf. Bewusst unter der Obergrenze, damit ein Aufruf nicht lange blockiert."""

DEFAULT_TIME_BUDGET_S = 120.0
"""Vorgabe für die Wanduhrzeit eines Laufs, in Sekunden."""

MAX_TIME_BUDGET_S = 900.0
"""Obergrenze für die Wanduhrzeit. Länger wartet kein Aufrufer sinnvoll."""

_GUARD_ENTRY_LENGTH = 14
"""Länge des Tupels aus `cache.guard()` in `snapshot.js`."""

_GUARD_HREF_INDEX = 12
"""Position der Zieladresse in diesem Tupel."""

_MAX_HREF_LAENGE = 2048
"""Längstes href, das noch plausibel ist. Darüber ist es kein Verweisziel."""

_NAVIGIERBARE_SCHEMATA = frozenset({"http", "https"})

_SCHEMA_ZEICHEN = "abcdefghijklmnopqrstuvwxyz0123456789+.-"

_MAX_STALE_JE_SCHRITT = 5
_MAX_UEBERGANG_JE_LAUF = 5
"""So oft beobachtet ein Lauf neu, wenn er auf einem Übergangszustand steht."""

_MAX_FEHLERTEXT = 240
_ABKLINGZEIT_S = 0.5
"""So lange wartet der Aufrufer noch auf den Faden, wenn er selbst schon fertig ist."""

LIBRARY_TEXT_LIMIT = 6000
"""So viele Zeichen sichtbaren Text beobachtet `jev_ultrafast/snapshot.js` höchstens.

Die Grenze steht dort in `words.join('\n').slice(0,6000)` und greift, bevor
`text_limit` überhaupt etwas zu kürzen hat. Erreicht der beobachtete Text genau
diese Länge, ist die Seite vermutlich länger, und das Ergebnis sagt es. Ein
Vertragstest hält die Zahl gegen die installierte Datei.
"""

_LAUF_SCHLOSS = threading.Lock()
"""Nur ein Lauf gleichzeitig, siehe `run_task`."""

_LAUF_NUMMER = count(1)

_NACHLAUF_DECKEL_S = 120.0
"""So lange wartet die Freigabe des Lauf-Schlosses höchstens auf den Arbeitsfaden.

Das Schloss wird erst freigegeben, wenn der Faden wirklich fertig ist, sonst
arbeiten nach einer Zeitüberschreitung zwei Läufe im selben Browser. Der Deckel
sorgt dafür, dass ein endgültig hängender Faden das Schloss nicht für immer
behält.
"""

_STDOUT_SCHLOSS = threading.Lock()
_STDOUT_TIEFE = 0
_STDOUT_ORIGINAL: object = None


@contextlib.contextmanager
def ohne_stdout() -> Iterator[None]:
    """Lenkt die Standardausgabe auf stderr, solange irgendwer diesen Riegel hält.

    Bei stdio ist die Standardausgabe der Protokollkanal, und jedes fremde Byte
    darin zerstört die Verbindung. Der Riegel zählt mit, wie viele ihn gerade
    halten, und stellt die Ausgabe erst wieder her, wenn der letzte ihn loslässt.
    Das Zählen ist nötig, weil sich zwei Halter überschneiden: der Werkzeugaufruf
    und der Arbeitsfaden, der das Zeitbudget überlebt. Ein einfaches
    `redirect_stdout` würde dabei in der falschen Reihenfolge zurückgelegt und
    liesse `sys.stdout` am Ende auf stderr stehen.

    Der Riegel ist kein vollständiger Schutz. Er fasst `sys.stdout` an, nicht den
    Dateideskriptor 1: ein `os.write(1, ...)` oder ein Unterprozess geht daran
    vorbei. Dass auch das nicht in der Leitung landet, liegt allein am SDK, das
    den Deskriptor 1 während des Betriebs auf stderr legt.
    """
    global _STDOUT_TIEFE, _STDOUT_ORIGINAL
    ziel = sys.stderr
    if ziel is None:  # Nur in Umgebungen ohne stderr.
        yield
        return
    with _STDOUT_SCHLOSS:
        if _STDOUT_TIEFE == 0:
            _STDOUT_ORIGINAL = sys.stdout
            sys.stdout = ziel
        _STDOUT_TIEFE += 1
    try:
        yield
    finally:
        with _STDOUT_SCHLOSS:
            _STDOUT_TIEFE -= 1
            if _STDOUT_TIEFE <= 0:
                _STDOUT_TIEFE = 0
                sys.stdout = _STDOUT_ORIGINAL  # type: ignore[assignment]
                _STDOUT_ORIGINAL = None


def wait_until_idle(timeout: float = 30.0) -> bool:
    """Wartet, bis kein Lauf und kein Lesevorgang mehr in Arbeit ist.

    Gedacht für Aufrufer, die wissen müssen, ob der Browser wieder frei ist,
    etwa eine Testsuite zwischen zwei Fällen. Gibt `True` zurück, wenn das
    Schloss innerhalb der Frist frei war.
    """
    if not _LAUF_SCHLOSS.acquire(timeout=timeout):
        return False
    _LAUF_SCHLOSS.release()
    return True


class RunStatus(StrEnum):
    """Wie ein Lauf geendet hat."""

    DONE = "done"
    """Der Agent hat das Ziel für erreicht erklärt."""

    BLOCKED = "blocked"
    """Der Agent kam selbst nicht weiter."""

    PLANNED = "planned"
    """Trockenlauf: es wurde geplant, aber nichts ausgeführt."""

    STOPPED_DOMAIN = "stopped_domain"
    """Die Domain-Treue hat den Lauf angehalten."""

    STOPPED_BUDGET = "stopped_budget"
    """Ein Aktions- oder Modellaufruf-Budget war erschöpft."""

    STOPPED_TIME = "stopped_time"
    """Das Zeitbudget war erschöpft."""

    FAILED = "failed"
    """Der Lauf ist an einem Fehler gescheitert."""

    NOT_STARTED = "not_started"
    """Es wurde kein Browser geöffnet, die Voraussetzungen stimmten nicht."""


@dataclass(frozen=True)
class StepRecord:
    """Ein ausgeführter Schritt, in lesbarer Form."""

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
    """Die Domain-Entscheidung, die einen Lauf angehalten hat.

    Eine serialisierbare Kopie von `guards.DomainDecision`. Die Adressen darin
    sind die bereits entschärften Kurzformen des Wächters.
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
    """Was der Agent als Nächstes täte. Ergebnis eines Trockenlaufs."""

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
    """Das Ergebnis eines Laufs, vollständig und ohne Geheimnisse."""

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
# Fehlerübersetzung
# ---------------------------------------------------------------------------

_SCHLUESSEL_ORT = f"~{str(DEFAULT_CONFIG_PATH).replace(str(DEFAULT_CONFIG_PATH.home()), '', 1)}"

_UEBERSETZUNGEN: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"^stopped at the \d+-action demo budget"),
        "Die Bibliothek hat ihr eigenes Aktionsbudget ausgeschöpft und den Lauf beendet. "
        "Formuliere das Ziel enger oder teile es in mehrere Aufrufe auf.",
    ),
    (
        re.compile(r"^reached the demo's model-call budget"),
        "Das Budget für Modellaufrufe der Bibliothek ist erschöpft, der Lauf endet hier. "
        "Das passiert, wenn der Agent viele Schritte verwirft, weil sich die Seite laufend ändert. "
        "Versuche es mit einem engeren Ziel oder auf einer ruhigeren Seite erneut.",
    ),
    (
        re.compile(r"^this run has stopped\b"),
        "Dieser Lauf war bereits beendet, es kann darin kein weiterer Schritt mehr ausgeführt werden. "
        "Starte einen neuen Lauf, wenn noch etwas zu tun ist.",
    ),
    (
        re.compile(r"^type_text needs text_model_api_key\b"),
        "Der nächste Schritt wäre Tippen gewesen, dafür fehlt der Schlüssel des Textmodells "
        "(TEXT_MODEL_API_KEY). Klicken, Auswählen und Navigieren gehen weiterhin, nur Formulare und "
        f"Suchfelder nicht. Hinterlege den Schlüssel in {_SCHLUESSEL_ORT} oder in der Umgebung, "
        "dann kann der Agent auch tippen.",
    ),
    (
        re.compile(r"^text helper returned no valid field value\b"),
        "Das Textmodell hat keinen brauchbaren Feldwert geliefert, es wurde deshalb nichts getippt. "
        "Sage im Ziel genauer, was in das Feld gehört, und versuche es erneut.",
    ),
    (
        re.compile(r"^invalid typesafe response\b"),
        "Das Entscheidungsmodell hat eine Antwort geliefert, die sich nicht auswerten liess, "
        "es wurde nichts ausgeführt. Ein erneuter Versuch hilft hier meistens.",
    ),
    (
        re.compile(r"^model connection failed\b"),
        "Das Entscheidungs- oder Textmodell war nicht erreichbar, es wurde nichts ausgeführt. "
        "Prüfe die Internetverbindung und die hinterlegten Schlüssel, dann starte den Lauf neu.",
    ),
    (
        re.compile(r"^model provider returned http \d+"),
        "Der Modellanbieter hat den Aufruf mit einem Fehler beantwortet, es wurde nichts ausgeführt. "
        "Das ist meistens ein abgelaufener Schlüssel, ein leeres Guthaben oder eine Drosselung. "
        "Prüfe das Konto beim Anbieter und versuche es danach erneut.",
    ),
    (
        re.compile(r"^model unavailable$"),
        "Der Modellanbieter war auch nach mehreren Versuchen nicht verfügbar, es wurde nichts ausgeführt. "
        "Warte einen Moment und starte den Lauf danach neu.",
    ),
    (
        re.compile(r"^observe and choose before acting$"),
        "Der Lauf hat versucht zu handeln, ohne vorher eine gültige Entscheidung zu haben. "
        "Das ist ein Fehler in der Ablaufsteuerung dieses Servers, nicht auf der Seite. "
        "Starte den Lauf neu und melde den Fall, wenn er sich wiederholt.",
    ),
    (
        re.compile(r"^supply a task$"),
        "Es wurde kein Ziel an den Agenten übergeben. Nenne in Worten, was auf der Seite geschehen soll.",
    ),
    (
        re.compile(r"^invalid observed node$"),
        "Der Agent wollte ein Element bedienen, das sich nicht mehr eindeutig zuordnen liess, "
        "es wurde nichts ausgeführt. Starte den Lauf auf der Seite neu.",
    ),
    (
        re.compile(r"^target changed or is covered\b"),
        "Das Element hat sich bewegt oder liegt unter einem anderen, der Klick wurde nicht ausgeführt. "
        "Schliesse Einblendungen wie Cookie-Banner oder Chat-Fenster und starte den Lauf neu.",
    ),
    (
        re.compile(r"^dropdown execution was (?:interrupted|not confirmed)\b"),
        "Eine Auswahlliste liess sich nicht sicher bedienen, der Zustand des Feldes ist unklar. "
        "Sieh im Browser nach, was dort jetzt ausgewählt ist, bevor du den Lauf wiederholst.",
    ),
    (
        re.compile(r"\brequired daemon \S+ is (?:not running|unhealthy)\b|^daemon-starting\b"),
        "Der Browser-Harness-Dienst läuft nicht oder ist nicht gesund, deshalb liess sich kein Chrome "
        "ansprechen. Starte ihn mit `browser-harness daemon start` und versuche es erneut.",
    ),
    (
        re.compile(r"\bdaemon \S+ didn't come up\b"),
        "Der Browser-Harness-Dienst ist nicht hochgekommen, deshalb liess sich kein Chrome ansprechen. "
        "Sieh in sein Protokoll, starte ihn neu und versuche es danach erneut.",
    ),
    (
        re.compile(r"^permission-blocked\b|^remote debugging is turned off\b"),
        "Chrome lässt die Fernsteuerung nicht zu, deshalb konnte der Lauf nicht beginnen. "
        "Erlaube sie in Chrome unter chrome://inspect und bestätige die Rückfrage, dann starte den "
        "Lauf neu.",
    ),
    (
        re.compile(r"^devtoolsactiveport not found\b"),
        "Chrome läuft ohne offene Fernsteuerung, der Lauf konnte deshalb nicht beginnen. "
        "Schalte sie unter chrome://inspect ein und starte den Lauf danach neu.",
    ),
    (
        re.compile(r"^bu_cdp_url=\S* unreachable\b|^cdp ws handshake failed\b"),
        "Die Verbindung zu Chrome hat nicht geantwortet, der Lauf wurde abgebrochen. "
        "Prüfe, ob Chrome läuft und mit dem Harness verbunden ist, dann starte den Lauf neu.",
    ),
    (
        re.compile(r"^javascript evaluation failed\b"),
        "Die Seite hat die Beobachtung abgewiesen, meistens weil sie gerade neu lädt. "
        "Warte kurz und starte den Lauf danach neu.",
    ),
    (
        re.compile(r"\btimed out waiting for\b"),
        "Ein Aufruf hat die Zeitgrenze überschritten, es wurde nichts weiter ausgeführt. "
        "Prüfe Netz und Browser und starte den Lauf danach neu.",
    ),
)
"""Die Wortlaute der Bibliothek, jeder an seinem Satzanfang verankert.

Verankert, nicht als lose Teilzeichenkette gesucht. Vorher entschied das Wort
`chrome` irgendwo im Text, und die Meldung "Konnte Element 'Zur
Chrome-Erweiterung' nicht anklicken" wurde zu "Chrome war nicht ansprechbar".
Ebenso wurde aus "Element not found: a[href='/connection-settings']" ein
Verbindungsausfall. Passt kein Muster, sagt der Satz das ehrlich, statt eine
falsche Ursache zu behaupten.

Ein Vertragstest hält jedes Muster gegen das installierte `jev_ultrafast` und
`browser_harness`. Benennt ein Update dort eine Meldung um, fällt das auf.
"""

_TYP_UEBERSETZUNGEN: tuple[tuple[type[BaseException], str], ...] = (
    (
        TimeoutError,
        "Ein Aufruf hat die Zeitgrenze überschritten, es wurde nichts weiter ausgeführt. "
        "Prüfe Netz und Browser und starte den Lauf danach neu.",
    ),
    (
        ConnectionError,
        "Eine Verbindung ist ausgefallen, der Lauf wurde abgebrochen. "
        "Prüfe Netz und Browser und starte den Lauf danach neu.",
    ),
)
"""Fälle, die sich am Typ der Ausnahme sicherer erkennen lassen als am Text."""


def kurzfassung(text: str) -> str:
    """Eine einzeilige, gekürzte Fassung eines fremden Fehlertexts."""
    sauber = " ".join(str(text).split())
    return sauber[: _MAX_FEHLERTEXT - 1] + "…" if len(sauber) > _MAX_FEHLERTEXT else sauber


def translate_error(error: BaseException) -> str:
    """Übersetzt eine Ausnahme der Bibliothek in einen ganzen deutschen Satz.

    Der Satz sagt, was passiert ist, und was der Nutzer tun kann. Passt nichts,
    kommt der ursprüngliche Text gekürzt und einzeilig mit, damit nichts
    verschwindet, was beim Suchen hilft.
    """
    text = " ".join(str(error).split()).lower()
    for muster, satz in _UEBERSETZUNGEN:
        if muster.search(text):
            return satz
    for typ, satz in _TYP_UEBERSETZUNGEN:
        if isinstance(error, typ):
            return satz
    return (
        "Der Lauf ist an einer Stelle gescheitert, die dieser Server nicht einordnen kann "
        f"({type(error).__name__}: {kurzfassung(str(error) or 'ohne Text')}). "
        "Starte den Lauf neu und melde den Fall, wenn er sich wiederholt."
    )


def _budgetart(stand: "_Stand", auftrag: "_Auftrag") -> str | None:
    """Sagt aus den **eigenen** Zählern, ob ein Budget erschöpft war.

    Bewusst nicht aus dem Fehlertext der Bibliothek. Der Status eines Laufs darf
    nicht davon abhängen, wie eine fremde Meldung gerade formuliert ist: eine
    harmlose Umbenennung von "demo budget" zu "step budget" liess den Status
    vorher von `stopped_budget` auf `failed` kippen, ohne dass sich am Lauf
    etwas geändert hätte. Gezählt wird, was dieser Modul ohnehin mitführt: die
    ausgeführten Schritte und die Entscheidungen des Modells.
    """
    if len(stand.steps) >= min(auftrag.max_actions, LIBRARY_MAX_ACTIONS):
        return "actions"
    if stand.model_calls >= LIBRARY_MAX_MODEL_CALLS:
        return "model_calls"
    return None


def _ist_veraltete_seite(error: BaseException) -> bool:
    """True für `jev_ultrafast.browser.StalePage`, ohne die Bibliothek zu importieren.

    Erkannt wird am Klassennamen in der Ableitungskette. Das hält diesen Modul
    frei von einem Import, der den Browser-Harness mitzieht, und lässt Tests
    einen eigenen Doppelgänger derselben Bezeichnung verwenden.
    """
    return any(klasse.__name__ == "StalePage" for klasse in type(error).__mro__)


# ---------------------------------------------------------------------------
# Die Zieladresse vor dem Klick
# ---------------------------------------------------------------------------


def _schema(adresse: str) -> str:
    """Das Schema einer Adresse, kleingeschrieben, oder ein leerer Text."""
    kopf, trenner, _ = adresse.partition(":")
    if not trenner or not kopf:
        return ""
    klein = kopf.lower()
    if not klein[0].isalpha() or any(zeichen not in _SCHEMA_ZEICHEN for zeichen in klein):
        return ""
    return klein


_KEIN_ZIEL_HINWEIS = (
    "Für mindestens einen Klick lag vorab keine Zieladresse vor. Nur Verweise mit href liefern eine, "
    "Schaltflächen und Formulare nicht. Die Domain-Treue greift für solche Klicks erst nach dem Laden, "
    "und im eingeloggten Profil ist die geladene Seite bereits der Schaden."
)

_UEBERGANG_HINWEIS = (
    "Der Browser stand zwischendurch auf einem Übergangszustand, der kein Ziel des Auftrags ist, etwa "
    "about:blank oder eine Oberfläche des Browsers selbst. Das hält den Lauf nicht an, gehandelt wird "
    "dort aber nicht: der Lauf hat stattdessen neu beobachtet und auf die nächste richtige Adresse "
    "gewartet."
)

_VERALTET_HINWEIS = (
    "Die Seite hat sich während eines Schritts geändert, der Schritt wurde deshalb verworfen und neu "
    "beobachtet. Das ist der vorgesehene Wiederholungsfall und kein Fehler."
)

_UNPLAUSIBLES_ZIEL_HINWEIS = (
    "An der Stelle der Zieladresse stand etwas, das keine Adresse sein kann, etwa ein Stück "
    "Seitentext. Vermutlich hat sich die Reihenfolge der Felder in der Bibliothek geändert. Die "
    "Zieladresse wurde deshalb nicht geprüft, und die Domain-Treue greift für diesen Schritt erst "
    "nach dem Laden."
)

_NICHT_AUFLOESBAR_HINWEIS = (
    "Eine Zieladresse liess sich nicht zu einer vollständigen Adresse auflösen, sie wurde deshalb "
    "vor dem Klick nicht geprüft. Die Domain-Treue greift für diesen Schritt erst nach dem Laden."
)


def _plausibles_href(wert: str) -> bool:
    """Sagt, ob ein gelesener Wert überhaupt eine Zieladresse sein kann.

    Die Position der Adresse im Tupel der Bibliothek ist nur über die Länge
    abgesichert, und eine Länge fängt kein Vertauschen. Direkt neben der Adresse
    steht der Text der Umgebung des Elements. Wird beides vertauscht, entsteht
    aus "Jetzt anmelden und Konto bestaetigen" eine scheinbar harmlose Adresse
    auf der eigenen Domain, die anstandslos durchginge.

    Unterschieden wird deshalb nach der Form: eine Adresse trägt keinen
    Leerraum, und sie ist nicht beliebig lang. Fliesstext trägt beides.
    """
    if not wert or len(wert) > _MAX_HREF_LAENGE:
        return False
    return not any(zeichen.isspace() for zeichen in wert)


def planned_target_url(page: Mapping, choice: str) -> tuple[str | None, str | None]:
    """Liest die Zieladresse, die ein geplanter Klick ansteuern würde.

    Gibt `(Adresse, Hinweis)` zurück. Die Adresse ist absolut und trägt ein
    Schema, mit dem der Browser wirklich navigiert, sonst ist sie `None`. Der
    Hinweis ist ein deutscher Satz für das Ergebnis, sobald es etwas zu sagen
    gibt, sonst `None`.

    `None` ohne Hinweis heisst: hier ist keine Navigation zu prüfen, etwa bei
    einem Textfeld, einem Sprung innerhalb der Seite oder `mailto:`. `None` mit
    Hinweis heisst: es wäre etwas zu prüfen gewesen, die Bibliothek gibt es aber
    nicht her. Siehe den Modul-Docstring, dort steht die Herkunft der Daten.
    """
    aktion = next((a for a in (page.get("actions") or []) if a.get("id") == choice), None)
    if aktion is None or aktion.get("kind") != "click":
        return None, None

    eintrag = (page.get("guards") or {}).get(str(aktion.get("node")))
    if eintrag is None:
        return None, _KEIN_ZIEL_HINWEIS
    if not isinstance(eintrag, list | tuple) or len(eintrag) != _GUARD_ENTRY_LENGTH:
        return None, (
            "Die Elementtabelle der Bibliothek hat eine unerwartete Form, die Zieladresse liess sich "
            "vor dem Klick nicht ablesen. Die Domain-Treue greift für diesen Schritt erst nach dem Laden."
        )

    href = eintrag[_GUARD_HREF_INDEX]
    if not isinstance(href, str) or not href.strip():
        return None, _KEIN_ZIEL_HINWEIS

    ziel = href.strip()
    if not _plausibles_href(ziel):
        return None, _UNPLAUSIBLES_ZIEL_HINWEIS
    if ziel.startswith("#"):
        return None, None

    schema = _schema(ziel)
    if schema == "javascript":
        return None, (
            "Mindestens ein Klick führte auf ein Skript-Ziel (javascript:), das vorab nicht verrät, "
            "wohin es geht. Die Domain-Treue greift für solche Klicks erst nach dem Laden."
        )
    if schema and schema not in _NAVIGIERBARE_SCHEMATA:
        return None, None

    if schema:
        aufgeloest: str | None = ziel
    else:
        aufgeloest = resolve_url(str(page.get("url") or ""), ziel)
    if aufgeloest is None or urlsplit(aufgeloest).scheme.lower() not in _NAVIGIERBARE_SCHEMATA:
        return None, _NICHT_AUFLOESBAR_HINWEIS
    return aufgeloest, None


# ---------------------------------------------------------------------------
# Fortschritt, der zwischen Faden und Aufrufer geteilt wird
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Stand:
    """Eine Momentaufnahme des Fortschritts, gefahrlos zu lesen."""

    url: str
    title: str
    steps: tuple[StepRecord, ...]
    model_calls: int
    notes: tuple[str, ...]
    duration_ms: int


def _schritt_von(eintrag: Mapping) -> StepRecord:
    """Baut einen Schritt aus einem Eintrag der Historie, Feld für Feld."""
    return StepRecord(
        step=int(eintrag.get("step") or 0),
        action=str(eintrag.get("action") or ""),
        kind=str(eintrag.get("kind") or ""),
        url=str(eintrag.get("url") or ""),
        page_changed=eintrag.get("page_changed"),
        text=eintrag.get("text"),
        operation=eintrag.get("operation"),
        target=eintrag.get("target"),
        confidence=eintrag.get("confidence"),
        elapsed_ms=int(eintrag.get("elapsed_ms") or 0),
    )


class _Fortschritt:
    """Was bisher geschah, unter einem Schloss, weil zwei Fäden darauf sehen."""

    def __init__(self, url: str, notes: Sequence[str]) -> None:
        self._schloss = threading.Lock()
        self._url = url
        self._titel = ""
        self._steps: tuple[StepRecord, ...] = ()
        self._model_calls = 0
        self._notes: list[str] = list(notes)
        self._begonnen = time.monotonic()

    def uebernimm(self, snapshot: Mapping) -> None:
        """Liest Adresse, Titel, Historie und Modellaufrufe aus einer Momentaufnahme."""
        seite = snapshot.get("page") or {}
        schritte = tuple(_schritt_von(eintrag) for eintrag in (snapshot.get("history") or []))
        aufrufe = len(snapshot.get("decisions") or [])
        with self._schloss:
            self._url = str(seite.get("url") or self._url)
            self._titel = str(seite.get("title") or self._titel)
            self._steps = schritte
            self._model_calls = max(self._model_calls, aufrufe)

    def notiere(self, hinweis: str | None) -> None:
        """Nimmt einen Hinweis auf, jeden Wortlaut nur einmal."""
        if not hinweis:
            return
        with self._schloss:
            if hinweis not in self._notes:
                self._notes.append(hinweis)

    def lies(self) -> _Stand:
        with self._schloss:
            return _Stand(
                url=self._url,
                title=self._titel,
                steps=self._steps,
                model_calls=self._model_calls,
                notes=tuple(self._notes),
                duration_ms=round((time.monotonic() - self._begonnen) * 1000),
            )


class _Sitzung:
    """Hält den Agenten und schliesst ihn genau einmal, egal von welchem Faden."""

    def __init__(self) -> None:
        self._schloss = threading.Lock()
        self._agent: object | None = None
        self._geschlossen = False
        self._erfolg = True
        self._hatte_agent = False

    @property
    def hatte_agent(self) -> bool:
        """True, sobald je ein Agent übergeben wurde, also je ein Tab offen war.

        Wird ein Lauf abgebrochen, während `Agent.__init__` noch im
        `ensure_daemon()` der Bibliothek steckt, gab es nie einen Tab. Das
        Ergebnis darf dann nicht behaupten, es habe einen geschlossen.
        """
        with self._schloss:
            return self._hatte_agent

    def uebernimm(self, agent: object) -> None:
        with self._schloss:
            self._hatte_agent = True
            if not self._geschlossen:
                self._agent = agent
                return
        _schliesse(agent)

    def schliesse(self) -> bool:
        """Schliesst den Agenten und sagt, ob das gelungen ist."""
        with self._schloss:
            if self._geschlossen:
                return self._erfolg
            self._geschlossen = True
            agent = self._agent
            self._agent = None
        erfolg = _schliesse(agent)
        with self._schloss:
            self._erfolg = erfolg
        return erfolg


def _schliesse(agent: object | None) -> bool:
    """Schliesst einen Agenten, schluckt dabei jeden Fehler und meldet den Ausgang.

    Beim Aufräumen darf nichts mehr scheitern, ein Fehler hier würde das
    eigentliche Ergebnis des Laufs überschreiben. Verschwiegen wird er trotzdem
    nicht: scheitert das Schliessen, bleibt der Browser-Tab offen, und der
    Aufrufer erfährt das über den Rückgabewert und am Ende über einen Hinweis im
    Ergebnis.
    """
    if agent is None:
        return True
    try:
        agent.close()  # type: ignore[attr-defined]
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Der Auftrag
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Auftrag:
    """Die geprüften Vorgaben eines Laufs."""

    start_url: str
    goals: tuple[str, ...]
    max_actions: int
    time_budget_s: float
    dry_run: bool
    agent_factory: Callable[..., object]
    notes: tuple[str, ...] = field(default=())


def _ziele(goals: Sequence[str] | str | None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Macht aus der Zielangabe eine Liste nicht leerer Sätze samt Hinweisen.

    Ein leeres Ziel verschwand bisher stillschweigend, und aus
    `["Suche die Seite", ""]` wurde ein Lauf mit einem Ziel, ohne dass das
    irgendwo stand. Verworfen wird es weiterhin, aber es wird gesagt.
    """
    if goals is None:
        return (), ()
    roh = (goals,) if isinstance(goals, str) else tuple(goals)
    ziele = tuple(text.strip() for text in roh if isinstance(text, str) and text.strip())
    verworfen = len(roh) - len(ziele)
    if not verworfen:
        return ziele, ()
    return ziele, (
        f"Von den angegebenen Zielen waren {verworfen} leer oder kein Satz in Worten. Sie wurden "
        f"verworfen, gelaufen wird mit den übrigen {len(ziele)}.",
    )


def _als_zahl(wert: object) -> float | None:
    """Liest eine endliche Zahl, oder `None`.

    `None` steht auch für `nan` und für die Unendlichkeiten. Sie sind Zahlen im
    Sinne von `float()`, aber keine Budgets: `nan <= 0` ist falsch, also käme
    `nan` durch jede Untergrenze, `Event.wait(nan)` kehrt sofort zurück, und
    `nan` im Ergebnis ergibt `NaN` im JSON, was kein gültiges JSON ist. Ein
    strenger Aufrufer lehnt eine solche Antwort ab. `json.loads` liefert genau
    diese Werte, `{"max_actions": Infinity}` ist dafür schon genug.

    `True` und `False` sind in Python ebenfalls Zahlen, und `float(True)` ist
    eine glatte Eins. `max_actions: true` ergab dadurch kommentarlos einen Lauf
    mit einer einzigen Aktion und `time_budget_s: true` ein Budget von einer
    Sekunde, das danach als Zeitüberschreitung endete, ohne dass irgendwo
    stand, warum. Ein Wahrheitswert ist deshalb hier keine Zahl.
    """
    if isinstance(wert, bool):
        return None
    try:
        zahl = float(wert)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return zahl if math.isfinite(zahl) else None


def _budget(max_actions: object) -> tuple[int, tuple[str, ...]]:
    """Prüft das Aktionsbudget und kappt es an der Obergrenze der Bibliothek."""
    zahl = _als_zahl(max_actions)
    if zahl is None:
        return DEFAULT_MAX_ACTIONS, (
            f"Das Aktionsbudget war keine endliche Zahl, es gilt deshalb die Vorgabe von "
            f"{DEFAULT_MAX_ACTIONS} Aktionen.",
        )
    wert = int(zahl)
    if wert > LIBRARY_MAX_ACTIONS:
        return LIBRARY_MAX_ACTIONS, (
            f"Das Aktionsbudget von {wert} liegt über der Obergrenze der Bibliothek und wurde auf "
            f"{LIBRARY_MAX_ACTIONS} Aktionen gekappt.",
        )
    if wert < 1:
        return 1, ("Das Aktionsbudget lag unter einer Aktion und wurde auf eine Aktion angehoben.",)
    return wert, ()


def _zeitbudget(time_budget_s: object) -> tuple[float, tuple[str, ...]]:
    """Prüft das Zeitbudget und hält es zwischen einer Sekunde und der Obergrenze."""
    wert = _als_zahl(time_budget_s)
    if wert is None:
        return DEFAULT_TIME_BUDGET_S, (
            f"Das Zeitbudget war keine endliche Zahl, es gilt deshalb die Vorgabe von "
            f"{int(DEFAULT_TIME_BUDGET_S)} Sekunden.",
        )
    if wert <= 0:
        return DEFAULT_TIME_BUDGET_S, (
            "Das Zeitbudget war nicht positiv, es gilt deshalb die Vorgabe von "
            f"{int(DEFAULT_TIME_BUDGET_S)} Sekunden.",
        )
    if wert > MAX_TIME_BUDGET_S:
        return MAX_TIME_BUDGET_S, (
            f"Das Zeitbudget von {wert:g} Sekunden liegt über der Obergrenze und wurde auf "
            f"{int(MAX_TIME_BUDGET_S)} Sekunden gekappt.",
        )
    return wert, ()


def _standard_agent(url: str, goals: list[str]) -> object:
    """Baut den echten Agenten. Der Import bleibt hier, nicht auf Modulebene.

    `jev_ultrafast` zieht beim Import den Browser-Harness mit. Der Runner soll
    auch dann geladen und geprüft werden können, wenn kein Browser in der Nähe
    ist, deshalb geschieht der Import erst, wenn wirklich ein Lauf beginnt.
    """
    import jev_ultrafast.browser as bibliothek_browser
    from jev_ultrafast import Agent

    installiere_eigenes_fenster(bibliothek_browser)
    return Agent(url, goals)


def eigenes_fenster(cdp: Callable[..., object]) -> Callable[..., object]:
    """Legt Hintergrund-Tabs als eigenes Fenster im Hintergrund an.

    Die Bibliothek öffnet ihren Tab mit `background=True`, damit sie dem Nutzer
    nicht den sichtbaren Tab wegnimmt. Chrome 153 beantwortet aber keinen
    einzigen Befehl an einen so angelegten Tab, jeder Aufruf läuft nach fünf
    Sekunden in die Zeitgrenze des Harness. Gemessen am 21.09.2026: derselbe Tab
    sichtbar geöffnet antwortet sofort, und ein eigenes Fenster im Hintergrund
    ebenfalls. Letzteres lässt das Fenster des Nutzers unberührt, deshalb wird
    genau dieser eine Aufruf umgebogen und alles andere unverändert durchgereicht.
    Ein ausdrücklich gesetztes `newWindow` bleibt stehen.
    """

    def weiter(method: str, session_id: str | None = None, **params: object) -> object:
        if method == "Target.createTarget" and params.get("background") and "newWindow" not in params:
            params["newWindow"] = True
        return cdp(method, session_id=session_id, **params)

    weiter.__jev_mcp_eigenes_fenster__ = True  # type: ignore[attr-defined]
    return weiter


def installiere_eigenes_fenster(modul: object) -> None:
    """Hängt `eigenes_fenster` genau einmal in das `cdp` eines Moduls ein."""
    vorhanden = getattr(modul, "cdp", None)
    if vorhanden is None or getattr(vorhanden, "__jev_mcp_eigenes_fenster__", False):
        return
    modul.cdp = eigenes_fenster(vorhanden)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Ergebnisbau
# ---------------------------------------------------------------------------


def _domain_stop(entscheidung: DomainDecision) -> DomainStop:
    return DomainStop(
        verdict=str(entscheidung.verdict.value),
        reason=entscheidung.reason,
        moment=str(entscheidung.moment.value),
        start_domain=entscheidung.start_domain,
        target_domain=entscheidung.target_domain,
        target_url=entscheidung.target_url,
        policy_note=entscheidung.policy_note,
        warnings=tuple(entscheidung.warnings),
    )


def _ergebnis(
    auftrag: _Auftrag,
    stand: _Stand,
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
        start_url=auftrag.start_url,
        url=stand.url,
        title=stand.title,
        goals=auftrag.goals,
        steps=stand.steps,
        duration_ms=stand.duration_ms,
        actions_used=len(stand.steps),
        model_calls=stand.model_calls,
        max_actions=auftrag.max_actions,
        time_budget_s=auftrag.time_budget_s,
        budget_exhausted=budget_kind is not None,
        budget_kind=budget_kind,
        domain_stop=domain_stop,
        planned=planned,
        notes=stand.notes,
        error=error,
    )


def _sekunden(stand: _Stand) -> int:
    return max(0, round(stand.duration_ms / 1000))


def _ohne_lauf(
    auftrag: _Auftrag,
    status: RunStatus,
    summary: str,
    *,
    notes: Sequence[str] = (),
    domain_stop: DomainStop | None = None,
) -> RunResult:
    """Ein Ergebnis für einen Lauf, der gar nicht erst begonnen hat."""
    stand = _Stand(
        url=auftrag.start_url,
        title="",
        steps=(),
        model_calls=0,
        notes=tuple(dict.fromkeys((*auftrag.notes, *notes))),
        duration_ms=0,
    )
    return _ergebnis(auftrag, stand, status, summary, domain_stop=domain_stop)


# ---------------------------------------------------------------------------
# Der Lauf selbst
# ---------------------------------------------------------------------------


def _trockenlauf(auftrag: _Auftrag, wache: RunGuard, fortschritt: _Fortschritt, agent: object) -> RunResult:
    """Beobachtet und holt die erste Entscheidung ein, ohne sie auszuführen.

    Der Trockenlauf steht an der Stelle, an der sonst eine Rückfrage beim
    Menschen stünde. Also muss er auch sagen, was die Rückfrage sagen würde:
    ob der geplante Schritt den Auftrag verlässt. Sonst liest ein Modell
    `ok: true` und startet danach den echten Lauf.
    """
    momentaufnahme = agent.command("predict")  # type: ignore[attr-defined]
    fortschritt.uebernimm(momentaufnahme)
    entscheidung = momentaufnahme.get("decision") or {}
    auswahl = str(entscheidung.get("choice") or "")
    seite = momentaufnahme.get("page") or {}
    aktion = next((a for a in (seite.get("actions") or []) if a.get("id") == auswahl), None)
    ziel, hinweis = planned_target_url(seite, auswahl)
    fortschritt.notiere(hinweis)

    geplant = PlannedStep(
        choice=auswahl,
        action=str((aktion or {}).get("label") or auswahl),
        kind=str((aktion or {}).get("kind") or ""),
        operation=entscheidung.get("operation"),
        target=entscheidung.get("target"),
        confidence=entscheidung.get("confidence"),
        target_url=ziel,
        will_type=(aktion or {}).get("kind") == "fill",
    )

    if ziel is not None:
        vorher = wache.check(ziel, Moment.BEFORE)
        if not vorher.allowed:
            fortschritt.notiere(vorher.policy_note)
            for warnung in vorher.warnings:
                fortschritt.notiere(warnung)
            satz = (
                "Der Trockenlauf hat nichts ausgeführt, und dieser Schritt würde auch im echten Lauf "
                f"nicht ausgeführt: er verlässt den Auftrag. {vorher.reason}"
            )
            return _ergebnis(
                auftrag,
                fortschritt.lies(),
                RunStatus.STOPPED_DOMAIN,
                satz,
                planned=geplant,
                domain_stop=_domain_stop(vorher),
            )

    stand = fortschritt.lies()
    if auswahl == "DONE":
        satz = "Der Trockenlauf ergibt, dass auf dieser Seite nichts mehr zu tun ist."
    elif auswahl == "BLOCKED":
        satz = "Der Trockenlauf ergibt, dass der Agent hier nicht weiterkäme."
    else:
        satz = (
            f"Der Trockenlauf ergibt als nächsten Schritt: {geplant.operation or geplant.kind} auf "
            f"«{geplant.action}»"
            + (f", Ziel {ziel}." if ziel else ". Eine Zieladresse liegt dafür nicht vor.")
        )
    return _ergebnis(auftrag, stand, RunStatus.PLANNED, satz, ok=True, planned=geplant)


def _neu_beobachten(
    auftrag: _Auftrag, wache: RunGuard, fortschritt: _Fortschritt, agent: object
) -> tuple[Mapping | None, DomainDecision | None, RunResult | None]:
    """Beobachtet neu und prüft sofort, wo der Browser dabei steht.

    Ein Grund, neu zu beobachten, ist immer ein Grund zu prüfen. "Die Seite hat
    gewechselt" ist genau der Fall, für den die Domain-Treue da ist, und ein
    leerer Übergangszustand ist genau der Fall, in dem nicht gehandelt wird.

    Gibt `(Momentaufnahme, Entscheidung, None)` zurück, oder
    `(None, None, Ergebnis)`, wenn der Lauf hier endet.
    """
    try:
        momentaufnahme = agent.snapshot()  # type: ignore[attr-defined]
    except Exception as fehler:
        return None, None, _gescheitert(auftrag, fortschritt, fehler)
    fortschritt.uebernimm(momentaufnahme)
    entscheidung = wache.check(fortschritt.lies().url, Moment.AFTER)
    if not entscheidung.allowed:
        return None, None, _angehalten(auftrag, fortschritt, entscheidung)
    return momentaufnahme, entscheidung, None


def _schleife(
    auftrag: _Auftrag,
    wache: RunGuard,
    fortschritt: _Fortschritt,
    agent: object,
    abbruch: threading.Event,
) -> RunResult:
    """Der eigentliche Ablauf: beobachten, prüfen, entscheiden, prüfen, handeln, prüfen.

    Geprüft wird nach **jeder** Beobachtung, auch nach den beiden, die auf eine
    veraltete Seite folgen, und auch nach der, die in `predict` steckt. Jede
    davon kann den Browser woanders angetroffen haben, und nur die Prüfung nach
    dem Laden sieht das.
    """
    momentaufnahme = agent.snapshot()  # type: ignore[attr-defined]
    fortschritt.uebernimm(momentaufnahme)

    stehen = wache.check(fortschritt.lies().url, Moment.AFTER)
    if not stehen.allowed:
        return _angehalten(auftrag, fortschritt, stehen)

    if auftrag.dry_run:
        if not stehen.may_interact:
            fortschritt.notiere(_UEBERGANG_HINWEIS)
        return _trockenlauf(auftrag, wache, fortschritt, agent)

    veraltet = 0
    uebergaenge = 0
    while not abbruch.is_set():
        stand = fortschritt.lies()
        if str(momentaufnahme.get("status") or "") in {"done", "blocked"}:
            return _beendet(auftrag, stand, str(momentaufnahme["status"]))
        if len(stand.steps) >= auftrag.max_actions:
            return _budget_erschoepft(auftrag, stand)

        if not stehen.may_interact:
            # NEUTRAL ist ausdrücklich keine Freigabe zum Handeln, und das sind
            # nicht nur leere Seiten: chrome:// und devtools:// sind bedienbare
            # Oberflächen. Also wird hier nicht geklickt, sondern neu beobachtet.
            fortschritt.notiere(_UEBERGANG_HINWEIS)
            uebergaenge += 1
            if uebergaenge > _MAX_UEBERGANG_JE_LAUF:
                return _uebergang_erschoepft(auftrag, fortschritt.lies())
            momentaufnahme, stehen, ende = _neu_beobachten(auftrag, wache, fortschritt, agent)
            if ende is not None:
                return ende
            continue

        try:
            momentaufnahme = agent.command("predict")  # type: ignore[attr-defined]
        except Exception as fehler:
            if _ist_veraltete_seite(fehler) and veraltet < _MAX_STALE_JE_SCHRITT:
                veraltet += 1
                fortschritt.notiere(_VERALTET_HINWEIS)
                momentaufnahme, stehen, ende = _neu_beobachten(auftrag, wache, fortschritt, agent)
                if ende is not None:
                    return ende
                continue
            return _gescheitert(auftrag, fortschritt, fehler)
        fortschritt.uebernimm(momentaufnahme)

        # `predict` beobachtet die Seite in der Bibliothek neu. Ein Tippen, ein
        # Auswählen oder ein Klick auf eine Schaltfläche ohne href liefe sonst
        # auf einer Seite, die inzwischen woanders steht.
        stehen = wache.check(fortschritt.lies().url, Moment.AFTER)
        if not stehen.allowed:
            return _angehalten(auftrag, fortschritt, stehen)
        if not stehen.may_interact:
            continue

        seite = momentaufnahme.get("page") or {}
        auswahl = str((momentaufnahme.get("decision") or {}).get("choice") or "")
        ziel, hinweis = planned_target_url(seite, auswahl)
        fortschritt.notiere(hinweis)
        if ziel is not None:
            vorher = wache.check(ziel, Moment.BEFORE)
            if not vorher.allowed:
                return _angehalten(auftrag, fortschritt, vorher)

        if abbruch.is_set():
            # Das Zeitbudget ist zwischen Beobachten und Handeln abgelaufen. Ein
            # abgelaufener Lauf handelt nicht mehr, und er verlässt sich dafür
            # auch nicht darauf, dass der Browser den Aufruf schon abweisen wird.
            break

        try:
            momentaufnahme = agent.command("act", {"fingerprint": seite.get("fingerprint")})  # type: ignore[attr-defined]
        except Exception as fehler:
            if _ist_veraltete_seite(fehler) and veraltet < _MAX_STALE_JE_SCHRITT:
                veraltet += 1
                fortschritt.notiere(_VERALTET_HINWEIS)
                momentaufnahme, stehen, ende = _neu_beobachten(auftrag, wache, fortschritt, agent)
                if ende is not None:
                    return ende
                continue
            return _gescheitert(auftrag, fortschritt, fehler)
        fortschritt.uebernimm(momentaufnahme)
        veraltet = 0

        stehen = wache.check(fortschritt.lies().url, Moment.AFTER)
        if not stehen.allowed:
            return _angehalten(auftrag, fortschritt, stehen)

    return _zeit_erschoepft(auftrag, fortschritt.lies())


def _beendet(auftrag: _Auftrag, stand: _Stand, status: str) -> RunResult:
    if status == "done":
        satz = (
            f"Der Agent hat das Ziel erreicht: {len(stand.steps)} Aktionen in {_sekunden(stand)} Sekunden, "
            f"zuletzt auf {stand.url}."
        )
        return _ergebnis(auftrag, stand, RunStatus.DONE, satz, ok=True)
    satz = (
        f"Der Agent kam nicht weiter und hat den Lauf nach {len(stand.steps)} Aktionen selbst beendet, "
        f"zuletzt auf {stand.url}."
    )
    return _ergebnis(auftrag, stand, RunStatus.BLOCKED, satz)


def _budget_erschoepft(auftrag: _Auftrag, stand: _Stand) -> RunResult:
    satz = (
        f"Der Lauf wurde beim Aktionsbudget von {auftrag.max_actions} Aktionen angehalten, zuletzt auf "
        f"{stand.url}. Erhöhe das Budget oder teile das Ziel in kleinere Aufträge."
    )
    return _ergebnis(auftrag, stand, RunStatus.STOPPED_BUDGET, satz, budget_kind="actions")


def _uebergang_erschoepft(auftrag: _Auftrag, stand: _Stand) -> RunResult:
    satz = (
        f"Der Browser blieb auf einem Zustand stehen, auf dem nicht gehandelt wird, zuletzt auf "
        f"{stand.url}. Nach {_MAX_UEBERGANG_JE_LAUF} neuen Beobachtungen hat der Lauf aufgegeben, "
        "statt dort zu klicken. Sieh im Browser nach, was die Seite gerade tut, und starte den Lauf "
        "danach neu."
    )
    return _ergebnis(auftrag, stand, RunStatus.BLOCKED, satz)


TAB_GESCHLOSSEN = "Der Browser-Tab wurde geschlossen."
TAB_OFFEN = "Der Browser-Tab liess sich nicht schliessen und ist vermutlich noch offen."
TAB_NIE_OFFEN = (
    "Es war noch kein Browser-Tab offen, den man hätte schliessen können: der Lauf steckte noch im "
    "Aufbau. Öffnet er im Hintergrund doch noch einen, wird er sofort wieder geschlossen."
)


def _schlusssatz(sitzung: "_Sitzung", erfolg: bool) -> str:
    """Sagt über den Browser-Tab nur das, was wirklich zutrifft."""
    if not sitzung.hatte_agent:
        return TAB_NIE_OFFEN
    return TAB_GESCHLOSSEN if erfolg else TAB_OFFEN


def _zeit_erschoepft(auftrag: _Auftrag, stand: _Stand, *, schluss: str = TAB_GESCHLOSSEN) -> RunResult:
    satz = (
        f"Der Lauf wurde nach dem Zeitbudget von {auftrag.time_budget_s:g} Sekunden abgebrochen, zuletzt "
        f"auf {stand.url}, nach {len(stand.steps)} Aktionen. {schluss}"
    )
    return _ergebnis(auftrag, stand, RunStatus.STOPPED_TIME, satz, budget_kind="time")


def _ohne_fremde_seite(stand: _Stand, entscheidung: DomainDecision) -> _Stand:
    """Nimmt alles aus dem Fortschritt, was von der fremden Seite stammt.

    Beim Anhalten nach dem Laden hat `_neu_beobachten()` den Fortschritt bereits
    mit der fremden Seite gefüllt, und zwar bevor geprüft wurde. Titel und
    Adresse der fremden Seite stünden danach roh und unbegrenzt im Ergebnis,
    obwohl die Werkzeugbeschreibung zusagt, dass von dort nichts zurückkommt.
    Ein Titel ist beliebiger Text: eine Anweisung an das Modell, Steuerzeichen,
    eine Richtungsumkehr, beliebige Länge.

    Der Titel fällt deshalb ganz weg, und als Adresse steht die bereits
    entschärfte Kurzform aus der Entscheidung da, dieselbe, die auch in
    `domain_stop.target_url` steht. Der letzte Schritt trägt dieselbe Adresse,
    wenn er auf ihr geendet ist.
    """
    fremd = stand.url
    schritte = stand.steps
    if schritte and schritte[-1].url == fremd:
        schritte = (*schritte[:-1], replace(schritte[-1], url=entscheidung.target_url))
    return replace(stand, url=entscheidung.target_url, title="", steps=schritte)


def _angehalten(auftrag: _Auftrag, fortschritt: _Fortschritt, entscheidung: DomainDecision) -> RunResult:
    fortschritt.notiere(entscheidung.policy_note)
    for warnung in entscheidung.warnings:
        fortschritt.notiere(warnung)
    stand = fortschritt.lies()
    if entscheidung.moment is Moment.AFTER:
        # Vor dem Klick steht der Browser noch auf der erlaubten Seite, dort gibt
        # es nichts zu entschärfen. Nach dem Laden steht er auf der fremden.
        stand = _ohne_fremde_seite(stand, entscheidung)
    wann = "vor dem Klick" if entscheidung.moment is Moment.BEFORE else "nach dem Laden"
    satz = f"Der Lauf wurde {wann} von der Domain-Treue angehalten. {entscheidung.reason}"
    return _ergebnis(auftrag, stand, RunStatus.STOPPED_DOMAIN, satz, domain_stop=_domain_stop(entscheidung))


def _gescheitert(auftrag: _Auftrag, fortschritt: _Fortschritt, fehler: BaseException) -> RunResult:
    stand = fortschritt.lies()
    satz = translate_error(fehler)
    art = _budgetart(stand, auftrag)
    status = RunStatus.STOPPED_BUDGET if art else RunStatus.FAILED
    return _ergebnis(auftrag, stand, status, satz, budget_kind=art, error=satz)


_OHNE_ERGEBNIS_HINWEIS = (
    "Der Lauf hat sich beendet, ohne ein Ergebnis abzulegen. Auch sein eigener Fehlerzweig ist "
    "gescheitert, die Ursache steht deshalb nur im Protokoll auf stderr. Eine Zeitüberschreitung "
    "war es nicht. Starte den Lauf neu und melde den Fall, wenn er sich wiederholt."
)

DOMAIN_TREUE_AUS_HINWEIS = (
    "Die Domain-Treue ist in der Policy-Datei abgeschaltet. Dieser Vorgang prüft deshalb nicht, ob "
    "die Seite auf eine fremde Domain wechselt, und gibt auch von dort zurück, was er findet. "
    "Setze enforce_domain_lock in ~/.config/jev-mcp/policy.toml wieder auf true, wenn das nicht "
    "gewollt ist."
)


def _policy_hinweise(wache: RunGuard) -> tuple[str, ...]:
    """Was über die geltende Policy in **jedes** Ergebnis gehört.

    Ist die Domain-Treue abgeschaltet, gibt es keinen Sperrgrund, der das sagen
    könnte: es wird ja nie gesperrt. Der Hinweis hängt deshalb nicht am Ausgang,
    sondern am Lauf.
    """
    if wache.policy.enforce_domain_lock:
        return ()
    return (DOMAIN_TREUE_AUS_HINWEIS,)


def _ohne_ergebnis(auftrag: _Auftrag, stand: _Stand) -> RunResult:
    """Der Faden hat sich fertig gemeldet und nichts abgelegt."""
    return _ergebnis(auftrag, stand, RunStatus.FAILED, _OHNE_ERGEBNIS_HINWEIS, error=_OHNE_ERGEBNIS_HINWEIS)


def _mit_notiz(ergebnis: RunResult, hinweis: str) -> RunResult:
    """Hängt einen Hinweis an ein fertiges Ergebnis, jeden Wortlaut nur einmal."""
    if hinweis in ergebnis.notes:
        return ergebnis
    return replace(ergebnis, notes=(*ergebnis.notes, hinweis))


# ---------------------------------------------------------------------------
# Die Eingangstür
# ---------------------------------------------------------------------------

_TAB_OFFEN_HINWEIS = (
    "Der Browser-Tab dieses Laufs liess sich nicht schliessen und ist vermutlich noch offen. "
    "Schliesse ihn von Hand, bevor du den nächsten Lauf startest."
)

_TAB_UNKLAR_HINWEIS = (
    "Das Schliessen des Browser-Tabs war nach einer halben Sekunde noch nicht fertig. Das Ergebnis "
    "stimmt trotzdem, nur über den Tab sagt es nichts. Sieh im Browser nach, ob er noch offen ist."
)

_GLEICHZEITIG_HINWEIS = (
    "Es läuft gerade schon ein Lauf, und es kann immer nur einer laufen. Ein Lauf bedient einen "
    "einzigen Browser und setzt dafür Werte in der Umgebung dieses Prozesses, zwei Läufe würden "
    "einander diese Werte überschreiben. Warte, bis der laufende Auftrag fertig ist, und starte "
    "diesen danach erneut."
)


def _gib_schloss_frei(faden: threading.Thread | None) -> None:
    """Gibt das Lauf-Schloss frei, sobald der Arbeitsfaden wirklich fertig ist.

    Nach Ablauf des Zeitbudgets kehrt der Aufrufer mit einem Ergebnis zurück,
    während der Faden noch im Browser arbeitet. Wurde das Schloss dabei sofort
    freigegeben, startete der nächste Aufruf einen zweiten Lauf im selben
    Browser, obwohl die Werkzeugbeschreibung zusagt, dass immer nur einer läuft.
    Das ist kein Randfall: `Agent.__init__` ruft `ensure_daemon()`, das bis zu
    sechzig Sekunden warten und notfalls Chrome starten kann, und das
    Vorgabebudget eines Lesevorgangs ist dreissig Sekunden.

    Damit das Schloss nicht selbst zum Hänger wird, hat das Warten einen Deckel.
    Danach wird freigegeben, auch wenn der Faden noch lebt.
    """
    if faden is None or not faden.is_alive():
        _LAUF_SCHLOSS.release()
        return

    def warte() -> None:
        try:
            faden.join(_NACHLAUF_DECKEL_S)
        finally:
            _LAUF_SCHLOSS.release()

    threading.Thread(target=warte, name=f"{faden.name}-nachlauf", daemon=True).start()


def sicherer_text(wert: object) -> str:
    """Macht aus irgendetwas einen Text, auch wenn dessen `__str__` wirft."""
    try:
        return str(wert)
    except Exception:  # noqa: BLE001
        return ""


def _auftrag_aus(
    start_url: object,
    goals: object,
    max_actions: object,
    time_budget_s: object,
    dry_run: object,
    agent_factory: Callable[..., object] | None,
) -> _Auftrag:
    """Baut den geprüften Auftrag. Darf werfen, der Aufrufer fängt alles."""
    grenze, budget_notes = _budget(max_actions)
    zeit, zeit_notes = _zeitbudget(time_budget_s)
    saetze, ziel_notes = _ziele(goals)  # type: ignore[arg-type]
    return _Auftrag(
        start_url=str(start_url or ""),
        goals=saetze,
        max_actions=grenze,
        time_budget_s=zeit,
        dry_run=bool(dry_run),
        agent_factory=agent_factory or _standard_agent,
        notes=(*budget_notes, *zeit_notes, *ziel_notes),
    )


def _eingabe_gescheitert(start_url: object, fehler: BaseException) -> RunResult:
    """Ein Ergebnis für Vorgaben, die sich gar nicht erst auswerten liessen."""
    satz = (
        "Die Vorgaben für diesen Lauf liessen sich nicht auswerten, deshalb wurde kein Browser "
        f"geöffnet ({type(fehler).__name__}: {kurzfassung(sicherer_text(fehler) or 'ohne Text')}). "
        "Prüfe Ziel, Aktionsbudget und Zeitbudget und versuche es erneut."
    )
    return RunResult(
        status=RunStatus.NOT_STARTED,
        ok=False,
        summary=satz,
        start_url=sicherer_text(start_url),
        url=sicherer_text(start_url),
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
    """Führt einen vollständigen Lauf aus und gibt ein serialisierbares Ergebnis zurück.

    `goals` sind Ziele in Worten, einzeln oder als Liste. `max_actions` ist die
    Zahl der Aktionen, Vorgabe 25, Obergrenze 60, höhere Werte werden gekappt
    und das steht danach in `notes`. `time_budget_s` ist die Wanduhrzeit,
    Vorgabe 120 Sekunden, Obergrenze 900 Sekunden, und sie wird auch dann
    durchgesetzt, wenn ein einzelner Schritt hängt: der Lauf läuft in einem
    eigenen Faden, und nach Ablauf des Budgets wird der Agent geschlossen und
    das Ergebnis gebaut.

    `dry_run=True` beobachtet die Seite und holt die erste Entscheidung des
    Modells ein, führt sie aber nicht aus. Das Ergebnis sagt dann, was der Agent
    als Nächstes täte, und ob dieser Schritt den Auftrag verlassen würde. Eine
    Rückfrage beim Menschen gibt es bewusst nicht, der Trockenlauf tritt an ihre
    Stelle.

    `allow_domains` erlaubt zusätzliche Domänen für diesen Lauf, alles Weitere
    zur Domain-Treue steht in `guards.py`. `environment` und `agent_factory`
    sind Nähte: ohne sie wendet der Lauf die Umgebung selbst an und baut den
    echten Agenten.

    Es läuft immer nur ein Lauf gleichzeitig. Ein zweiter Aufruf, der einen
    laufenden antrifft, wird sofort mit `not_started` abgewiesen, statt zu
    warten. Der Grund steht in `_GLEICHZEITIG_HINWEIS`. Nach einer
    Zeitüberschreitung gilt das weiter: der Aufrufer bekommt sein Ergebnis, das
    Schloss bleibt aber, bis der Arbeitsfaden wirklich fertig ist, siehe
    `_gib_schloss_frei()`.

    Der Agent wird in jedem Fall geschlossen, auch bei einer Ausnahme und auch
    bei Zeitüberschreitung. Gelingt das nicht, sagt ein Hinweis im Ergebnis das.
    Diese Funktion wirft nicht, jeder Ausgang ist ein `RunResult`.
    """
    try:
        auftrag = _auftrag_aus(start_url, goals, max_actions, time_budget_s, dry_run, agent_factory)
    except Exception as fehler:  # noqa: BLE001
        return _eingabe_gescheitert(start_url, fehler)

    if not auftrag.goals:
        return _ohne_lauf(
            auftrag,
            RunStatus.NOT_STARTED,
            "Es wurde kein Ziel angegeben, deshalb wurde kein Browser geöffnet. Nenne in Worten, was "
            "auf der Seite geschehen soll.",
        )

    if not _LAUF_SCHLOSS.acquire(blocking=False):
        return _ohne_lauf(auftrag, RunStatus.NOT_STARTED, _GLEICHZEITIG_HINWEIS)
    faden: threading.Thread | None = None
    try:
        ergebnis, faden = _fuehre_aus(
            auftrag,
            allow_domains=allow_domains,
            allow_unbound=allow_unbound,
            policy=policy,
            policy_path=policy_path,
            environment=environment,
        )
        return ergebnis
    finally:
        _gib_schloss_frei(faden)


def _fuehre_aus(
    auftrag: _Auftrag,
    *,
    allow_domains: Iterable[str] | str | None,
    allow_unbound: bool,
    policy: Policy | None,
    policy_path: object | None,
    environment: EnvironmentApplication | None,
) -> tuple[RunResult, threading.Thread | None]:
    """Der Lauf selbst, mit bereits geprüften Vorgaben und unter dem Lauf-Schloss.

    Gibt neben dem Ergebnis den Arbeitsfaden zurück, sofern einer gestartet
    wurde. Der Aufrufer gibt das Lauf-Schloss erst frei, wenn dieser Faden
    fertig ist, siehe `_gib_schloss_frei()`.
    """
    try:
        angewandt = environment if environment is not None else apply_environment()
        bereit = bool(angewandt.ok)
        umgebungs_notes = tuple(str(hinweis) for hinweis in (angewandt.notes or ()))
    except Exception as fehler:  # noqa: BLE001
        return _ohne_lauf(
            auftrag,
            RunStatus.NOT_STARTED,
            "Die Voraussetzungen für einen Lauf liessen sich nicht auswerten, deshalb wurde kein "
            f"Browser geöffnet ({type(fehler).__name__}: "
            f"{kurzfassung(sicherer_text(fehler) or 'ohne Text')}).",
        ), None

    if not bereit:
        return _ohne_lauf(
            auftrag,
            RunStatus.NOT_STARTED,
            "Die Voraussetzungen für einen Lauf stimmen nicht, deshalb wurde kein Browser geöffnet. "
            "Die Hinweise sagen, was fehlt.",
            notes=umgebungs_notes,
        ), None

    try:
        wache = start_run(
            auftrag.start_url,
            allow_domains,
            policy,
            allow_unbound=allow_unbound,
            policy_path=policy_path,  # type: ignore[arg-type]
        )
        eingang = wache.check(auftrag.start_url, Moment.BEFORE)
    except Exception as fehler:  # noqa: BLE001
        return _ohne_lauf(
            auftrag,
            RunStatus.NOT_STARTED,
            translate_error(fehler),
            notes=umgebungs_notes,
        ), None

    umgebungs_notes = (*umgebungs_notes, *_policy_hinweise(wache))

    if not eingang.allowed or not eingang.may_interact:
        return _ohne_lauf(
            auftrag,
            RunStatus.STOPPED_DOMAIN,
            f"Die Start-Adresse taugt nicht als Auftrag, es wurde kein Browser geöffnet. {eingang.reason}",
            notes=(*umgebungs_notes, *eingang.warnings),
            domain_stop=_domain_stop(eingang),
        ), None

    fortschritt = _Fortschritt(auftrag.start_url, (*auftrag.notes, *umgebungs_notes))
    sitzung = _Sitzung()
    abbruch = threading.Event()
    fertig = threading.Event()
    geschlossen = threading.Event()
    kasten: list[RunResult] = []
    schluss: list[bool] = []

    def arbeite() -> None:
        # Der Riegel für die Standardausgabe gehört in den Faden selbst. Der des
        # Werkzeugaufrufs endet mit dem Aufruf, dieser Faden überlebt ihn.
        with ohne_stdout():
            try:
                agent = auftrag.agent_factory(auftrag.start_url, list(auftrag.goals))
                sitzung.uebernimm(agent)
                kasten.append(_schleife(auftrag, wache, fortschritt, agent, abbruch))
            except BaseException as fehler:  # noqa: BLE001
                # Auch `KeyboardInterrupt` und `SystemExit`. Fing dieser Faden nur
                # `Exception`, legte er kein Ergebnis ab, meldete sich aber als
                # fertig, und der Aufrufer behauptete danach eine Zeitüberschreitung,
                # die es nie gab. Die wahre Ursache verschwand dabei spurlos.
                kasten.append(_gescheitert(auftrag, fortschritt, fehler))
            finally:
                # Erst melden, dann aufräumen. Hängt das Schliessen, wartete der
                # Aufrufer sonst das ganze Zeitbudget ab und meldete eine
                # Zeitüberschreitung, obwohl das fertige Ergebnis längst vorlag.
                fertig.set()
                schluss.append(sitzung.schliesse())
                geschlossen.set()

    faden = threading.Thread(target=arbeite, name=f"jev-mcp-run-{next(_LAUF_NUMMER)}", daemon=True)
    faden.start()
    beendet = fertig.wait(auftrag.time_budget_s)
    if beendet and kasten:
        ergebnis = kasten[0]
        if not geschlossen.wait(_ABKLINGZEIT_S):
            return _mit_notiz(ergebnis, _TAB_UNKLAR_HINWEIS), faden
        if schluss and not schluss[0]:
            return _mit_notiz(ergebnis, _TAB_OFFEN_HINWEIS), faden
        return ergebnis, faden
    if beendet:
        # Der Faden hat sich fertig gemeldet, aber nichts abgelegt: sein eigener
        # Fehlerzweig ist gescheitert. Eine Zeitüberschreitung war das nicht.
        return _ohne_ergebnis(auftrag, fortschritt.lies()), faden

    # Zeitbudget abgelaufen: erst dem Faden sagen, dass Schluss ist, dann den
    # Browser-Tab freigeben. Ein Ergebnis, das der Faden danach noch ablegt,
    # zählt nicht mehr, damit der Ausgang eindeutig bleibt.
    abbruch.set()
    erfolg = sitzung.schliesse()
    fertig.wait(_ABKLINGZEIT_S)
    return (
        _zeit_erschoepft(auftrag, fortschritt.lies(), schluss=_schlusssatz(sitzung, erfolg)),
        faden,
    )


# ---------------------------------------------------------------------------
# Lesen, ohne zu handeln
# ---------------------------------------------------------------------------
#
# `read_page()` ist die zweite Eingangstür dieses Moduls. Sie öffnet eine Seite,
# beobachtet sie genau einmal und gibt zurück, was dort steht. Sie ruft weder
# `predict` noch `act` auf, es fällt also kein Modellaufruf an und es wird
# nichts geklickt und nichts getippt.
#
# Die Domain-Treue gilt trotzdem. Eine Weiterleitung kann die Seite woanders
# hinführen, und was dann im Ergebnis stünde, käme von einer fremden Domain,
# ohne dass der Auftraggeber das je erfahren hätte. Deshalb wird nach dem Laden
# mit `Moment.AFTER` geprüft: gleiche Domain heisst lesen und den Wechsel
# vermerken, fremde Domain heisst anhalten und nichts zurückgeben. Nichts heisst
# nichts: auch nicht den Titel und auch nicht die erreichte Adresse. Beides ist
# Text, den ein Angreifer setzt, und ein Titel trägt Zeilenumbrüche,
# Steuerzeichen und beliebige Länge. Als Adresse steht die entschärfte Kurzform
# aus der Entscheidung da, dieselbe wie in `domain_stop.target_url`.
#
# Es gelten dieselben Regeln wie für `run_task()`: diese Funktion wirft nie, sie
# schliesst den Agenten in jedem Fall, und sie hält dasselbe Lauf-Schloss, denn
# sie bedient denselben einen Browser.

READ_GOAL = "Diese Seite nur ansehen. Es wird nichts geklickt und nichts getippt."
"""Der Auftrag, mit dem der Agent gebaut wird. Er wird nie ausgeführt.

`jev_ultrafast.Agent` besteht auf einem Auftrag und lehnt einen leeren ab. Der
Satz steht also nur da, damit der Agent sich bauen lässt: es folgt kein
`predict`, also sieht ihn auch kein Modell.
"""

DEFAULT_READ_TIME_BUDGET_S = 30.0
"""Vorgabe für die Wanduhrzeit eines Lesevorgangs, in Sekunden."""

DEFAULT_TEXT_LIMIT = 4000
"""So viele Zeichen sichtbaren Text gibt ein Lesevorgang höchstens zurück."""

MAX_TEXT_LIMIT = 20000
"""Obergrenze für diese Zahl. `snapshot.js` liefert ohnehin höchstens 6000."""

MIN_TEXT_LIMIT = 200

MAX_READ_ELEMENTS = 120
"""So viele Elemente stehen höchstens in der Tabelle. Die Bibliothek liefert bis zu 250."""

MAX_ELEMENT_TEXT = 200
"""So lang darf eine Beschriftung oder ein Feldwert in der Tabelle höchstens sein.

`text_limit` begrenzt nur den sichtbaren Text. Beschriftung, Feldwert und
Auswahlliste waren unbegrenzt, und eine Seite mit vielen langen Auswahllisten
ergab bei `text_limit=200` eine Antwort von 16,9 Megabyte.
"""

MAX_ELEMENT_OPTIONS = 50
"""So viele Einträge einer Auswahlliste stehen höchstens in der Tabelle."""


@dataclass(frozen=True)
class ReadElement:
    """Ein bedienbares Element der gelesenen Seite.

    Dieselbe Sicht, die auch das Entscheidungsmodell bekommt: `index` ist die
    Nummer, unter der `jev_ultrafast` das Element führt, `operations` sagt, was
    darauf möglich wäre. Ausgeführt wird davon beim Lesen nichts.
    """

    index: str
    label: str
    role: str = ""
    operations: tuple[str, ...] = ()
    value: str | None = None
    options: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReadResult:
    """Das Ergebnis eines Lesevorgangs, serialisierbar wie `RunResult`.

    `text` ist der sichtbare Text der Seite, gekürzt auf `text_chars` Zeichen.
    `text_total_chars` sagt, wie lang der **beobachtete** Text vor dem Kürzen
    war, und das ist nicht dasselbe wie die Länge der Seite: `snapshot.js`
    schneidet bei `LIBRARY_TEXT_LIMIT` Zeichen ab, bevor `text_limit` überhaupt
    greift. Erreicht der beobachtete Text genau diese Länge, sagt ein Hinweis
    das. `text_truncated` sagt, ob dieser Modul zusätzlich gekürzt hat.
    `redirected` sagt, ob die Seite woanders geendet ist als die angefragte
    Adresse.

    Was über Geheimnisse im Ergebnis gilt, steht im Modul-Docstring. Für das
    Lesen kommt nichts hinzu: es wird nichts getippt, und der Auftragstext ist
    ein fester Satz.
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


_LESE_ZEIT_HINWEIS = (
    "Das Zeitbudget war keine brauchbare Zahl, es gilt deshalb die Vorgabe von "
    f"{int(DEFAULT_READ_TIME_BUDGET_S)} Sekunden."
)

_LESE_GRENZE_HINWEIS = (
    f"Die Textgrenze war keine brauchbare Zahl, es gilt deshalb die Vorgabe von {DEFAULT_TEXT_LIMIT} Zeichen."
)

_BEOBACHTUNGSGRENZE_HINWEIS = (
    f"Die Bibliothek beobachtet höchstens {LIBRARY_TEXT_LIMIT} Zeichen sichtbaren Text, und genau "
    "diese Länge wurde erreicht. Die Seite kann also länger sein, als hier steht, und zwar "
    "unabhängig von text_limit."
)


def _lese_zeitbudget(time_budget_s: object) -> tuple[float, tuple[str, ...]]:
    """Prüft das Zeitbudget eines Lesevorgangs."""
    wert = _als_zahl(time_budget_s)
    if wert is None or wert <= 0:
        return DEFAULT_READ_TIME_BUDGET_S, (_LESE_ZEIT_HINWEIS,)
    if wert > MAX_TIME_BUDGET_S:
        return MAX_TIME_BUDGET_S, (
            f"Das Zeitbudget von {wert:g} Sekunden liegt über der Obergrenze und wurde auf "
            f"{int(MAX_TIME_BUDGET_S)} Sekunden gekappt.",
        )
    return wert, ()


def _textgrenze(text_limit: object) -> tuple[int, tuple[str, ...]]:
    """Prüft, auf wie viele Zeichen der sichtbare Text gekürzt wird."""
    wert = _als_zahl(text_limit)
    if wert is None:
        return DEFAULT_TEXT_LIMIT, (_LESE_GRENZE_HINWEIS,)
    zahl = int(wert)
    if zahl < MIN_TEXT_LIMIT:
        return MIN_TEXT_LIMIT, (
            f"Die Textgrenze lag unter {MIN_TEXT_LIMIT} Zeichen und wurde auf {MIN_TEXT_LIMIT} "
            "Zeichen angehoben.",
        )
    if zahl > MAX_TEXT_LIMIT:
        return MAX_TEXT_LIMIT, (
            f"Die Textgrenze von {zahl} Zeichen liegt über der Obergrenze und wurde auf "
            f"{MAX_TEXT_LIMIT} Zeichen gekappt.",
        )
    return zahl, ()


_ELEMENT_TEXT_HINWEIS = (
    f"Mindestens eine Beschriftung oder ein Feldwert war länger als {MAX_ELEMENT_TEXT} Zeichen und "
    "wurde für die Tabelle gekappt."
)

_ELEMENT_OPTIONEN_HINWEIS = (
    f"Mindestens eine Auswahlliste hatte mehr als {MAX_ELEMENT_OPTIONS} Einträge. In der Tabelle "
    f"stehen die ersten {MAX_ELEMENT_OPTIONS}."
)


def _gekappt(wert: object, gekappte: list[str]) -> str:
    """Kürzt einen einzelnen Wert der Elementtabelle und vermerkt das."""
    text = sicherer_text(wert)
    if len(text) <= MAX_ELEMENT_TEXT:
        return text
    gekappte.append(_ELEMENT_TEXT_HINWEIS)
    return text[:MAX_ELEMENT_TEXT]


def _leseelemente(momentaufnahme: Mapping) -> tuple[tuple[ReadElement, ...], int, tuple[str, ...]]:
    """Baut die Elementtabelle aus `agent.snapshot()["elements"]`.

    Das ist dieselbe Tabelle, die `jev_ultrafast.model.action_space()` für das
    Entscheidungsmodell baut. Fehlt sie, bleibt die Tabelle leer, statt dass
    hier etwas erfunden wird.

    Gibt zusätzlich die Hinweise zurück, wenn etwas gekappt wurde. Geprüft wird
    auf `list` und `tuple`, nicht auf `Sequence`: eine Zeichenkette ist eine
    Sequence, und die Antwort behauptete deshalb einmal «57 bedienbare Elemente,
    in der Tabelle stehen die ersten 0», weil sie deren Zeichen gezählt hatte.
    """
    roh = momentaufnahme.get("elements") or []
    if not isinstance(roh, list | tuple):
        return (), 0, ()
    hinweise: list[str] = []
    elemente: list[ReadElement] = []
    for eintrag in list(roh)[:MAX_READ_ELEMENTS]:
        if not isinstance(eintrag, Mapping):
            continue
        rohe_optionen = [option for option in (eintrag.get("options") or []) if isinstance(option, Mapping)]
        if len(rohe_optionen) > MAX_ELEMENT_OPTIONS:
            hinweise.append(_ELEMENT_OPTIONEN_HINWEIS)
        optionen = tuple(
            _gekappt(option.get("label"), hinweise) for option in rohe_optionen[:MAX_ELEMENT_OPTIONS]
        )
        wert = eintrag.get("value")
        elemente.append(
            ReadElement(
                index=_gekappt(eintrag.get("index") or "", hinweise),
                label=_gekappt(eintrag.get("label") or "", hinweise),
                role=_gekappt(eintrag.get("role") or "", hinweise),
                operations=tuple(_gekappt(name, hinweise) for name in (eintrag.get("operations") or [])),
                value=None if wert is None else _gekappt(wert, hinweise),
                options=optionen,
            )
        )
    return tuple(elemente), len(roh), tuple(dict.fromkeys(hinweise))


def _gekuerzter_text(text: object, grenze: int) -> tuple[str, bool, int]:
    """Kürzt den sichtbaren Text und sagt, wie lang er vorher war."""
    ganz = sicherer_text(text or "")
    if len(ganz) <= grenze:
        return ganz, False, len(ganz)
    return ganz[:grenze], True, len(ganz)


def _lese_ergebnis(
    start_url: str,
    zeit: float,
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
    """Ein Leseergebnis ohne Seiteninhalt, für jeden Ausgang ausser dem Erfolg."""
    return ReadResult(
        status=status,
        ok=ok,
        summary=summary,
        start_url=start_url,
        url=url if url is not None else start_url,
        title=title,
        time_budget_s=zeit,
        duration_ms=duration_ms,
        notes=tuple(dict.fromkeys(str(hinweis) for hinweis in notes if hinweis)),
        domain_stop=domain_stop,
        error=error,
    )


def _beobachte_einmal(
    agent: object,
    start_url: str,
    wache: RunGuard,
    zeit: float,
    grenze: int,
    notizen: Sequence[str],
    begonnen: float,
) -> ReadResult:
    """Beobachtet die geöffnete Seite genau einmal und macht daraus ein Ergebnis."""
    momentaufnahme = agent.snapshot()  # type: ignore[attr-defined]
    seite = momentaufnahme.get("page") or {}
    erreicht = sicherer_text(seite.get("url") or start_url)
    titel = sicherer_text(seite.get("title") or "")
    dauer = round((time.monotonic() - begonnen) * 1000)

    hinweise = [*notizen]
    entscheidung = wache.check(erreicht, Moment.AFTER)
    if entscheidung.policy_note:
        hinweise.append(entscheidung.policy_note)
    hinweise.extend(entscheidung.warnings)

    if not entscheidung.allowed:
        satz = (
            "Die Seite ist beim Laden auf eine fremde Domain gewechselt, deshalb wurde nichts "
            f"gelesen. {entscheidung.reason} Wenn das gewollt ist, nenne die Domain in "
            "allow_domains."
        )
        # Von der fremden Seite kommt nichts zurück, auch nicht ihr Titel und
        # nicht ihre rohe Adresse. Ein Titel ist beliebiger Text: eine Anweisung
        # an das Modell, Steuerzeichen, beliebige Länge. Als Adresse steht die
        # entschärfte Kurzform aus der Entscheidung da.
        return _lese_ergebnis(
            start_url,
            zeit,
            RunStatus.STOPPED_DOMAIN,
            satz,
            url=entscheidung.target_url,
            title="",
            notes=hinweise,
            domain_stop=_domain_stop(entscheidung),
            duration_ms=dauer,
        )

    umgeleitet = erreicht != start_url
    if umgeleitet:
        hinweise.append(
            f"Die angefragte Adresse hat auf {erreicht} weitergeleitet, gelesen wurde diese Seite."
        )

    text, gekuerzt, ganze_laenge = _gekuerzter_text(seite.get("text"), grenze)
    if gekuerzt:
        hinweise.append(
            f"Der beobachtete Text war {ganze_laenge} Zeichen lang und wurde auf {len(text)} Zeichen "
            "gekürzt, der Rest steht nicht in dieser Antwort."
        )
    if ganze_laenge >= LIBRARY_TEXT_LIMIT:
        hinweise.append(_BEOBACHTUNGSGRENZE_HINWEIS)
    elemente, gesamt, kapp_hinweise = _leseelemente(momentaufnahme)
    hinweise.extend(kapp_hinweise)
    if gesamt > len(elemente):
        hinweise.append(
            f"Die Seite hat {gesamt} bedienbare Elemente, in der Tabelle stehen die ersten {len(elemente)}."
        )
    ausgelassen = seite.get("omitted_actions") or 0
    if isinstance(ausgelassen, int) and ausgelassen > 0:
        hinweise.append(
            f"Die Bibliothek hat {ausgelassen} weitere Elemente gar nicht erst beobachtet, die "
            "Seite ist dafür zu gross."
        )

    satz = (
        f"Die Seite {erreicht} wurde gelesen: {len(text)} Zeichen sichtbarer Text und "
        f"{len(elemente)} bedienbare Elemente. Es wurde nichts geklickt und nichts getippt."
    )
    return ReadResult(
        status=RunStatus.DONE,
        ok=True,
        summary=satz,
        start_url=start_url,
        url=erreicht,
        title=titel,
        text=text,
        text_truncated=gekuerzt,
        text_chars=len(text),
        text_total_chars=ganze_laenge,
        elements=elemente,
        elements_shown=len(elemente),
        elements_total=gesamt,
        redirected=umgeleitet,
        duration_ms=dauer,
        time_budget_s=zeit,
        notes=tuple(dict.fromkeys(str(hinweis) for hinweis in hinweise if hinweis)),
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
    """Öffnet eine Seite, beobachtet sie einmal und gibt zurück, was dort steht.

    Es wird nicht geklickt, nicht getippt und nichts ausgewählt: weder `predict`
    noch `act` werden aufgerufen, es entsteht also auch kein Modellaufruf und
    keine Kosten. Zurück kommen der sichtbare Text, auf `text_limit` Zeichen
    gekürzt, und die Elementtabelle, die auch das Entscheidungsmodell sähe.

    Die Domain-Treue gilt auch hier. Endet die Seite nach einer Weiterleitung
    auf einer fremden Domain, hält der Vorgang an und gibt nichts von dort
    zurück. Ein Wechsel innerhalb derselben Domain wird gelesen und im Ergebnis
    vermerkt, verschwiegen wird er nicht.

    Es läuft immer nur ein Vorgang gleichzeitig, `run_task()` und `read_page()`
    teilen sich dieses Schloss, denn sie teilen sich den Browser, und es bleibt
    nach einer Zeitüberschreitung so lange gehalten, bis der Arbeitsfaden
    wirklich fertig ist. Der Agent wird in jedem Fall geschlossen. Diese
    Funktion wirft nicht, jeder Ausgang ist ein `ReadResult`.
    """
    try:
        adresse = sicherer_text(url or "").strip()
        zeit, zeit_notes = _lese_zeitbudget(time_budget_s)
        grenze, grenzen_notes = _textgrenze(text_limit)
    except Exception as fehler:  # noqa: BLE001
        satz = (
            "Die Vorgaben für diesen Lesevorgang liessen sich nicht auswerten, deshalb wurde keine "
            f"Seite geöffnet ({type(fehler).__name__}: "
            f"{kurzfassung(sicherer_text(fehler) or 'ohne Text')})."
        )
        return _lese_ergebnis(sicherer_text(url), DEFAULT_READ_TIME_BUDGET_S, RunStatus.NOT_STARTED, satz)

    notizen = [*zeit_notes, *grenzen_notes]
    if not adresse:
        return _lese_ergebnis(
            adresse,
            zeit,
            RunStatus.NOT_STARTED,
            "Es wurde keine Adresse angegeben, deshalb wurde keine Seite geöffnet.",
            notes=notizen,
        )

    if not _LAUF_SCHLOSS.acquire(blocking=False):
        return _lese_ergebnis(adresse, zeit, RunStatus.NOT_STARTED, _GLEICHZEITIG_HINWEIS, notes=notizen)
    faden: threading.Thread | None = None
    try:
        ergebnis, faden = _lies_seite(
            adresse,
            zeit,
            grenze,
            notizen,
            allow_domains=allow_domains,
            policy=policy,
            policy_path=policy_path,
            environment=environment,
            agent_factory=agent_factory,
        )
        return ergebnis
    finally:
        _gib_schloss_frei(faden)


def _lies_seite(
    adresse: str,
    zeit: float,
    grenze: int,
    notizen: Sequence[str],
    *,
    allow_domains: Iterable[str] | str | None,
    policy: Policy | None,
    policy_path: object | None,
    environment: EnvironmentApplication | None,
    agent_factory: Callable[..., object] | None,
) -> tuple[ReadResult, threading.Thread | None]:
    """Der Lesevorgang selbst, mit geprüften Vorgaben und unter dem Lauf-Schloss.

    Gibt neben dem Ergebnis den Arbeitsfaden zurück, sofern einer gestartet
    wurde, siehe `_gib_schloss_frei()`.
    """
    try:
        angewandt = environment if environment is not None else apply_environment()
        bereit = bool(angewandt.ok)
        umgebungs_notes = tuple(str(hinweis) for hinweis in (angewandt.notes or ()))
    except Exception as fehler:  # noqa: BLE001
        satz = (
            "Die Voraussetzungen für einen Lesevorgang liessen sich nicht auswerten, deshalb wurde "
            f"keine Seite geöffnet ({type(fehler).__name__}: "
            f"{kurzfassung(sicherer_text(fehler) or 'ohne Text')})."
        )
        return _lese_ergebnis(adresse, zeit, RunStatus.NOT_STARTED, satz, notes=notizen), None

    hinweise = [*notizen, *umgebungs_notes]
    if not bereit:
        return _lese_ergebnis(
            adresse,
            zeit,
            RunStatus.NOT_STARTED,
            "Die Voraussetzungen für einen Lesevorgang stimmen nicht, deshalb wurde keine Seite "
            "geöffnet. Die Hinweise sagen, was fehlt.",
            notes=hinweise,
        ), None

    try:
        wache = start_run(
            adresse,
            allow_domains,
            policy,
            policy_path=policy_path,  # type: ignore[arg-type]
        )
        eingang = wache.check(adresse, Moment.BEFORE)
    except Exception as fehler:  # noqa: BLE001
        return _lese_ergebnis(
            adresse, zeit, RunStatus.NOT_STARTED, translate_error(fehler), notes=hinweise
        ), None

    hinweise.extend(_policy_hinweise(wache))

    if not eingang.allowed or not eingang.may_interact:
        return _lese_ergebnis(
            adresse,
            zeit,
            RunStatus.STOPPED_DOMAIN,
            f"Die Adresse taugt nicht als Leseauftrag, es wurde keine Seite geöffnet. {eingang.reason}",
            notes=(*hinweise, *eingang.warnings),
            domain_stop=_domain_stop(eingang),
        ), None

    begonnen = time.monotonic()
    sitzung = _Sitzung()
    fertig = threading.Event()
    geschlossen = threading.Event()
    kasten: list[ReadResult] = []
    schluss: list[bool] = []
    fabrik = agent_factory or _standard_agent

    def arbeite() -> None:
        # Der Riegel für die Standardausgabe gehört in den Faden selbst, siehe
        # `_fuehre_aus()`.
        with ohne_stdout():
            try:
                agent = fabrik(adresse, [READ_GOAL])
                sitzung.uebernimm(agent)
                kasten.append(_beobachte_einmal(agent, adresse, wache, zeit, grenze, hinweise, begonnen))
            except BaseException as fehler:  # noqa: BLE001
                satz = translate_error(fehler)
                kasten.append(
                    _lese_ergebnis(
                        adresse,
                        zeit,
                        RunStatus.FAILED,
                        satz,
                        notes=hinweise,
                        duration_ms=round((time.monotonic() - begonnen) * 1000),
                        error=satz,
                    )
                )
            finally:
                fertig.set()
                schluss.append(sitzung.schliesse())
                geschlossen.set()

    faden = threading.Thread(target=arbeite, name=f"jev-mcp-read-{next(_LAUF_NUMMER)}", daemon=True)
    faden.start()
    beendet = fertig.wait(zeit)
    if beendet and kasten:
        ergebnis = kasten[0]
        if not geschlossen.wait(_ABKLINGZEIT_S):
            return _mit_lese_notiz(ergebnis, _TAB_UNKLAR_HINWEIS), faden
        if schluss and not schluss[0]:
            return _mit_lese_notiz(ergebnis, _TAB_OFFEN_HINWEIS), faden
        return ergebnis, faden
    if beendet:
        return _lese_ergebnis(
            adresse,
            zeit,
            RunStatus.FAILED,
            _OHNE_ERGEBNIS_HINWEIS,
            notes=hinweise,
            duration_ms=round((time.monotonic() - begonnen) * 1000),
            error=_OHNE_ERGEBNIS_HINWEIS,
        ), faden

    erfolg = sitzung.schliesse()
    fertig.wait(_ABKLINGZEIT_S)
    return _lese_ergebnis(
        adresse,
        zeit,
        RunStatus.STOPPED_TIME,
        f"Die Seite war nach dem Zeitbudget von {zeit:g} Sekunden noch nicht gelesen, der Vorgang "
        f"wurde abgebrochen. {_schlusssatz(sitzung, erfolg)}",
        notes=hinweise,
        duration_ms=round((time.monotonic() - begonnen) * 1000),
    ), faden


def _mit_lese_notiz(ergebnis: ReadResult, hinweis: str) -> ReadResult:
    """Hängt einen Hinweis an ein fertiges Leseergebnis, jeden Wortlaut nur einmal."""
    if hinweis in ergebnis.notes:
        return ergebnis
    return replace(ergebnis, notes=(*ergebnis.notes, hinweis))
