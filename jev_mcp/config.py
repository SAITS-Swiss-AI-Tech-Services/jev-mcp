"""Voraussetzungen für einen jev-ultrafast-Lauf auflösen, anwenden und diagnostizieren.

Befund aus `jev_ultrafast/model.py`, gelesen am 20.09.2026:
Die Bibliothek liest ihre Variablen erst beim Aufruf, nicht beim Import. Auf
Modulebene steht nur der httpx-Client, der keine Variable anfasst. Erst in
`choose()` wird `os.environ["TYPESAFE_API_KEY"]` gelesen, dazu
`os.environ.get("TYPESAFE_MODEL", "jev-latest")`. Erst in `field_text()` werden
`TEXT_MODEL_API_KEY`, `TEXT_MODEL_BASE_URL` (Vorgabe `https://api.deepseek.com/v1`),
`TEXT_MODEL` (Vorgabe `deepseek-chat`) und `TEXT_MODEL_REASONING` gelesen.
Daraus folgt: es genügt, `os.environ` vor dem Aufruf zu setzen. Ob jev_ultrafast
vorher oder nachher importiert wurde, spielt keine Rolle. `apply_environment()`
setzt trotzdem Basis-URL und Modellnamen immer mit, denn die Vorgaben der
Bibliothek zeigen auf DeepSeek. Ein Schlüssel eines anderen Anbieters würde
sonst still gegen den falschen Endpunkt laufen.

Reihenfolge der Auflösung für den Textmodell-Schlüssel: der Anbieter geht vor
der Quelle. Zuerst zählt `TEXT_MODEL_API_KEY`, dann Kimi, dann DeepSeek, zuletzt
OpenRouter, und innerhalb jeder Stufe gilt zuerst die Umgebung und danach die
Konfigurationsdatei. Wer einen Schlüssel eigens für dieses Projekt in die Datei
legt, meint diesen Anbieter, auch wenn in der Shell noch ein alter Schlüssel
eines anderen Anbieters steht. Für `TYPESAFE_API_KEY` gibt es nur eine Variable,
dort gilt schlicht Umgebung vor Datei.

Geheimnisse verlassen dieses Modul nur über `apply_environment()`, das sie nach
`os.environ` schreibt. Keine öffentliche Datenstruktur und keine Meldung dieses
Moduls enthält einen Schlüsselwert oder ein Stück davon. Das gilt auch für die
Basis-URL: nach aussen geht nur eine gesäuberte Fassung ohne Anmeldedaten, ohne
Abfrage und ohne Fragment, denn beides sind übliche Verstecke für Geheimnisse.
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

SOURCE_ENVIRONMENT = "Umgebung"

TYPESAFE_VARIABLE = "TYPESAFE_API_KEY"

# Grössere Dateien werden nicht gelesen. Eine Konfigurationsdatei mit
# Umgebungsvariablen ist nie so gross, alles darüber ist ein Versehen.
MAX_CONFIG_FILE_BYTES = 256 * 1024

# Hartes Gesamtbudget für die Browser-Prüfung. browser_harness setzt sein
# Zeitlimit je Socket-Aufruf, nicht für den ganzen Vorgang, deshalb braucht es
# hier eine eigene Obergrenze.
BROWSER_PROBE_BUDGET_SECONDS = 3.0

# browser_harness gibt je Socket-Aufruf eine Sekunde. Antwortet der Daemon erst
# kurz davor mit Nein, war das vermutlich sein Zeitlimit und keine echte
# Auskunft, deshalb liegt die Schwelle knapp darunter.
_DAEMON_SLOW_ANSWER_SECONDS = 0.9

# Anbieter-Vorgaben. Base-URL und Modellname bleiben über TEXT_MODEL_BASE_URL
# und TEXT_MODEL überschreibbar, ohne dass der Anbieter gewechselt wird.
KIMI = ("https://api.moonshot.ai/v1", "kimi-k3")
DEEPSEEK = ("https://api.deepseek.com/v1", "deepseek-chat")
OPENROUTER = ("https://openrouter.ai/api/v1", "inception/mercury-2.5")

# Vorgabe der Bibliothek selbst, siehe Modul-Docstring. Gilt, wenn jemand
# TEXT_MODEL_API_KEY setzt, ohne Base-URL und Modell dazuzuschreiben.
UPSTREAM_DEFAULT = DEEPSEEK


@dataclass(frozen=True)
class _Tier:
    """Eine Stufe der Schlüsselsuche, eine Stufe je Anbieter."""

    variables: tuple[str, ...]
    defaults: tuple[str, str]
    provider: str | None


# Erste Fundstelle gewinnt. Die Stufen werden von oben nach unten abgearbeitet,
# innerhalb einer Stufe gilt je Variable zuerst die Umgebung und dann die Datei.
TEXT_MODEL_TIERS: tuple[_Tier, ...] = (
    _Tier(("TEXT_MODEL_API_KEY",), UPSTREAM_DEFAULT, None),
    _Tier(("MOONSHOT_API_KEY", "KIMI_API_KEY"), KIMI, "kimi"),
    _Tier(("DEEPSEEK_API_KEY",), DEEPSEEK, "deepseek"),
    _Tier(("OPENROUTER_API_KEY",), OPENROUTER, "openrouter"),
)

TEXT_MODEL_VARIABLES: tuple[str, ...] = tuple(name for tier in TEXT_MODEL_TIERS for name in tier.variables)

# Anbietername nach Host der Basis-URL. So stimmt die Anzeige auch dann, wenn
# jemand TEXT_MODEL_API_KEY zusammen mit einer eigenen Basis-URL setzt.
PROVIDERS_BY_HOST = {
    "api.moonshot.ai": "kimi",
    "api.deepseek.com": "deepseek",
    "openrouter.ai": "openrouter",
}

# Woran ein Modellname seines Anbieters zu erkennen ist. Nur für einen Hinweis,
# nie für eine Entscheidung, denn Modellnamen sind frei wählbar. OpenRouter
# fehlt bewusst, dort tragen die Namen den fremden Anbieter im Namen.
PROVIDER_MODEL_MARKERS = {
    "kimi": ("kimi", "moonshot"),
    "deepseek": ("deepseek",),
}

# Nicht geheime Zusatzvariablen, die aus der Datei mit übernommen werden.
PASSTHROUGH_VARIABLES = ("TYPESAFE_MODEL", "TEXT_MODEL_REASONING")

_SETTINGS_VARIABLES = ("TEXT_MODEL_BASE_URL", "TEXT_MODEL", *PASSTHROUGH_VARIABLES)
_ALL_VARIABLES = (
    TYPESAFE_VARIABLE,
    *TEXT_MODEL_VARIABLES,
    *_SETTINGS_VARIABLES,
)

# Variablen, die apply_environment() setzt oder, wenn kein Wert vorliegt,
# aus der Zielumgebung entfernt.
_MANAGED_VARIABLES = (TYPESAFE_VARIABLE, "TEXT_MODEL_API_KEY", *PASSTHROUGH_VARIABLES)

_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_EXPORT_PREFIX = re.compile(r"\Aexport\s+")


@dataclass(frozen=True)
class KeyStatus:
    """Ob ein Schlüssel da ist und woher. Nie der Schlüssel selbst."""

    present: bool
    source: str | None
    variable: str | None
    detail: str


@dataclass(frozen=True)
class TextModelAccess:
    """Der Textmodell-Zugang ohne den Schlüssel.

    `base_url` ist die gesäuberte Fassung: Schema, Host, Port und Pfad, sonst
    nichts. Anmeldedaten, Abfrage und Fragment werden vorher entfernt.
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
    """Zustand des Browser-Harness-Daemons.

    `known` ist falsch, wenn die Prüfung in ihr Zeitbudget gelaufen ist. Dann
    sagen `daemon_running` und `browser_connected` nichts aus und der Zustand
    darf einen Lauf nicht blockieren.
    """

    daemon_running: bool
    browser_connected: bool
    detail: str
    known: bool = True


@dataclass(frozen=True)
class Diagnosis:
    """Alles, was vor einem Lauf über die Voraussetzungen bekannt ist."""

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
    """Ergebnis von `apply_environment()`. Enthält nur Namen, nie Werte.

    `ok` unterscheidet einen geglückten Lauf, bei dem es auch nichts zu tun
    geben kann, von einem gescheiterten. Ist `ok` falsch, wurde die Umgebung
    nicht verändert und `notes` sagt, woran es lag.
    """

    ok: bool
    applied: tuple[str, ...] = ()
    removed: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def display_path(path: object) -> str:
    """Pfad für Menschen, mit ~ statt des Heimatverzeichnisses.

    Verglichen wird auf Pfadebene, damit aus einem Verzeichnis, das nur zufällig
    mit dem Heimatverzeichnis anfängt, kein falsches ~ wird.
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
        # Auch ein Pfadobjekt, dessen __fspath__ selbst fliegt, darf eine
        # Diagnose nicht zum Absturz bringen.
        try:
            return str(path)
        except Exception:
            return "<unbrauchbarer Pfad>"


def file_source(path: object) -> str:
    """Quellenangabe für Werte, die aus der Konfigurationsdatei stammen."""
    return f"Datei {display_path(path)}"


def sanitized_url(value: object) -> str | None:
    """Die URL ohne Anmeldedaten, Abfrage und Fragment, sonst None.

    Nur diese Fassung darf in eine Rückgabe oder in eine Meldung. Der Rohwert
    kann Benutzername, Passwort oder einen Schlüssel in der Abfrage enthalten.
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
        # Ohne Netzteil ist es keine brauchbare Basis-URL, und was dann im Pfad
        # steht, könnte ungeprüft Anmeldedaten enthalten.
        return None
    if ":" in host:
        host = f"[{host}]"
    if port is not None:
        host = f"{host}:{port}"
    cleaned = urlunsplit((parts.scheme, host, parts.path, "", ""))
    return cleaned or None


def read_config_file(path: object) -> tuple[dict[str, str], tuple[str, ...]]:
    """Liest `KEY=VALUE` je Zeile. Wirft nie, meldet Probleme als Hinweise.

    Kommentarzeilen mit `#` und Leerzeilen werden übersprungen, ein führendes
    `export ` wird entfernt, Anführungszeichen um Werte werden entfernt und ein
    Kommentar am Zeilenende wird abgeschnitten. Zeilen mit einem Namen, der
    nicht dem Muster `[A-Za-z_][A-Za-z0-9_]*` entspricht, zählen als
    übersprungen, damit eine Datei in falscher Kodierung auffällt. Gelesen wird
    nur eine gewöhnliche Datei bis `MAX_CONFIG_FILE_BYTES`, damit eine FIFO die
    Diagnose nicht zum Stehen bringt. Eine fehlende Datei ist kein Problem und
    erzeugt keinen Hinweis, eine unlesbare oder teilweise kaputte Datei schon.
    """
    try:
        candidate = Path(path)  # type: ignore[arg-type]
    except Exception:
        return {}, ("Der Pfad zur Konfigurationsdatei ist unbrauchbar, es wird keine Datei gelesen.",)
    label = display_path(candidate)
    try:
        info = candidate.stat()
    except FileNotFoundError:
        return {}, ()
    except OSError as exc:
        return {}, (
            f"Die Konfigurationsdatei {label} liess sich nicht prüfen "
            f"({type(exc).__name__}), sie wird deshalb nicht gelesen.",
        )
    if not stat.S_ISREG(info.st_mode):
        return {}, (f"Der Pfad {label} ist keine gewöhnliche Datei, er wird deshalb nicht gelesen.",)
    if info.st_size > MAX_CONFIG_FILE_BYTES:
        return {}, (
            f"Die Konfigurationsdatei {label} ist mit {info.st_size} Bytes grösser als die "
            f"erlaubten {MAX_CONFIG_FILE_BYTES} Bytes und wird deshalb nicht gelesen.",
        )
    try:
        raw = candidate.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        return {}, (
            f"Die Konfigurationsdatei {label} ist vorhanden, "
            f"konnte aber nicht gelesen werden ({type(exc).__name__}).",
        )
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
            f"In der Konfigurationsdatei {label} wurde 1 Zeile übersprungen, weil sie nicht dem "
            "Muster NAME=WERT entspricht.",
        )
    elif skipped:
        notes = (
            f"In der Konfigurationsdatei {label} wurden {skipped} Zeilen übersprungen, weil sie "
            "nicht dem Muster NAME=WERT entsprechen.",
        )
    return values, notes


def _value_of(raw: str) -> str:
    """Den Wert einer Zeile auspacken, Anführungszeichen und Kommentar weg."""
    value = raw.strip()
    quote = value[:1]
    if quote in ('"', "'"):
        closing = value.find(quote, 1)
        if closing != -1:
            return value[1:closing]
        return _without_comment(value[1:])
    return _without_comment(value)


def _without_comment(value: str) -> str:
    """Alles ab einem Kommentarzeichen abschneiden, das nach Leerraum steht."""
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
    """Nur die Variablen abgreifen, die uns etwas angehen."""
    source = os.environ if env is None else env
    values: dict[str, str] = {}
    for name in _ALL_VARIABLES:
        value = _clean(source.get(name))
        if value is not None:
            values[name] = value
    return values


@dataclass(frozen=True)
class _Layers:
    """Die beiden Fundorte in ihrer Reihenfolge, Umgebung zuerst."""

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
    """Anbietername nach Host, None wenn der Host unbekannt ist."""
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
    return PROVIDERS_BY_HOST.get(host, host or "unbekannt")


@dataclass(frozen=True)
class _TextModelPlan:
    """Der aufgelöste Textmodell-Zugang, innen mit Schlüssel und Rohwerten."""

    secret: str | None = None
    variable: str | None = None
    source: str | None = None
    provider: str | None = None
    raw_base_url: str | None = None
    base_url: str | None = None
    model: str | None = None
    notes: tuple[str, ...] = ()


def _find_text_secret(layers: _Layers) -> tuple[str | None, str | None, str | None, _Tier]:
    """Den Textmodell-Schlüssel finden. Anbieter vor Quelle, siehe Docstring."""
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
                f"Die Basis-URL aus der Quelle {given_label} ist unbrauchbar, deshalb gilt die "
                f"Vorgabe {sanitized_url(default_url)} des Anbieters."
            )
        else:
            found = _host_provider(shown)
            if tier.provider is not None and found is not None and found != tier.provider:
                notes.append(
                    f"Die Basis-URL {shown} aus der Quelle {given_label} gehört zum Anbieter "
                    f"{found}, der Schlüssel stammt aber aus der Variable {variable} des Anbieters "
                    f"{tier.provider}, deshalb gilt die Vorgabe {sanitized_url(default_url)}."
                )
            elif tier.provider is not None and found is None:
                notes.append(
                    f"Die Basis-URL {shown} aus der Quelle {given_label} gehört zu keinem bekannten "
                    f"Anbieter, sie wird mit dem Schlüssel aus der Variable {variable} trotzdem "
                    "verwendet."
                )
                raw_base_url = given
            else:
                raw_base_url = given

    base_url = sanitized_url(raw_base_url)
    provider = tier.provider or (_provider_name(base_url) if base_url else "unbekannt")
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
    """Hinweis, wenn der Modellname offensichtlich zu einem anderen Anbieter gehört."""
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
                f"Der Modellname {model} sieht nach dem Anbieter {other} aus, angesprochen wird "
                f"aber der Anbieter {provider}, bitte prüfe diese Kombination."
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
            detail=(
                "Es wurde kein Textmodell-Schlüssel gefunden, weder in der Umgebung noch in der "
                "Konfigurationsdatei."
            ),
        )
    return TextModelAccess(
        present=True,
        source=plan.source,
        variable=plan.variable,
        provider=plan.provider,
        model=plan.model,
        base_url=plan.base_url,
        detail=(
            f"Der Textmodell-Zugang stammt aus der Quelle {plan.source} über die Variable "
            f"{plan.variable}, Anbieter {plan.provider}, Modell {plan.model}."
        ),
    )


def resolve_text_model(
    env: Mapping[str, str] | None = None,
    config_path: object | None = None,
) -> TextModelAccess:
    """Den Textmodell-Zugang auflösen, ohne den Schlüssel herauszugeben."""
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
                "Der TypeSafe-Schlüssel fehlt, weder die Umgebung noch die Konfigurationsdatei "
                f"enthalten {TYPESAFE_VARIABLE}."
            ),
        )
    return KeyStatus(
        present=True,
        source=label,
        variable=TYPESAFE_VARIABLE,
        detail=f"Der TypeSafe-Schlüssel wurde gefunden, Quelle {label}, Variable {TYPESAFE_VARIABLE}.",
    )


def resolve_typesafe(
    env: Mapping[str, str] | None = None,
    config_path: object | None = None,
) -> KeyStatus:
    """Den TypeSafe-Schlüssel auflösen, ohne den Schlüssel herauszugeben."""
    layers, _ = _layers(env, DEFAULT_CONFIG_PATH if config_path is None else config_path)
    return _typesafe_status(layers)


def probe_browser() -> BrowserStatus:
    """Prüft den Browser-Harness, ohne etwas zu starten.

    Bewusst nicht `browser_harness.admin.ensure_daemon`: die Funktion startet
    einen Daemon, startet notfalls Chrome und wartet dabei bis zu sechzig
    Sekunden. Für eine Diagnose ohne Nebenwirkung ist das der falsche Weg.
    `daemon_alive()` macht einen Ping, `daemon_browser_ready()` fragt den
    laufenden Daemon nach seiner Browser-Verbindung. Beide setzen ihr Zeitlimit
    von einer Sekunde je Socket-Aufruf, nicht für den ganzen Vorgang, deshalb
    deckelt `_browser_status()` die Prüfung zusätzlich mit
    `BROWSER_PROBE_BUDGET_SECONDS`.

    Kommt das Nein auf die Frage nach der Browser-Verbindung erst kurz vor
    diesem Zeitlimit, war vermutlich der Daemon beschäftigt und es ist keine
    Auskunft. Dieser Fall gilt als unbekannt und blockiert keinen Lauf.
    """
    from browser_harness.admin import daemon_alive, daemon_browser_ready

    if not daemon_alive():
        return BrowserStatus(
            daemon_running=False,
            browser_connected=False,
            detail="Der Browser-Harness-Daemon läuft nicht und antwortet deshalb auf keinen Ping.",
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
                    "Der Browser-Harness-Daemon läuft, hat aber nicht innerhalb seines Zeitlimits "
                    "geantwortet, sein Browserzustand ist deshalb unbekannt."
                ),
                known=False,
            )
        return BrowserStatus(
            daemon_running=True,
            browser_connected=False,
            detail="Der Browser-Harness-Daemon läuft, es ist aber kein Chrome mit ihm verbunden.",
        )
    return BrowserStatus(
        daemon_running=True,
        browser_connected=True,
        detail="Der Browser-Harness-Daemon läuft und ein Chrome ist mit ihm verbunden.",
    )


_UNKNOWN_BROWSER_DETAIL = "Der Zustand des Browser-Harness liess sich nicht ermitteln."


def _browser_status(probe: Callable[[], BrowserStatus] | None) -> tuple[BrowserStatus, tuple[str, ...]]:
    """Die Browser-Prüfung mit hartem Gesamtbudget ausführen.

    Die Prüfung läuft in einem eigenen Faden. Läuft das Budget ab, ist der
    Zustand unbekannt. Unbekannt ist nicht dasselbe wie nicht verbunden und
    blockiert deshalb keinen Lauf.
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
                detail=(
                    "Die Prüfung des Browser-Harness hat zu lange gedauert, sein Zustand ist "
                    "deshalb unbekannt."
                ),
                known=False,
            ),
            (
                "Die Prüfung des Browser-Harness wurde nach "
                f"{BROWSER_PROBE_BUDGET_SECONDS} Sekunden abgebrochen, der Browserzustand gilt "
                "deshalb als unbekannt und blockiert keinen Lauf.",
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
                "Die Prüfung des Browser-Harness ist mit einem Fehler abgebrochen "
                f"({type(error).__name__}), der Browser gilt deshalb als nicht verfügbar.",
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
            ("Die Prüfung des Browser-Harness hat eine unerwartete Antwort geliefert.",),
        )
    if not status.known:
        return status, (
            "Der Zustand des Browser-Harness ist unbekannt, der Agent versucht den Lauf trotzdem.",
        )
    return status, ()


def _blocked_operations(
    typesafe: KeyStatus, text_model: TextModelAccess, browser: BrowserStatus
) -> tuple[str, ...]:
    blocked: list[str] = []
    if not typesafe.present:
        blocked.append(
            "Kein Browser-Lauf ist möglich, weil der TypeSafe-Schlüssel fehlt. Ohne ihn kann der "
            "Agent keine einzige Aktion auswählen, auch nicht Klicken oder Scrollen. Lege "
            f"{TYPESAFE_VARIABLE} in die Umgebung oder in die Datei {display_path(DEFAULT_CONFIG_PATH)}."
        )
    if not text_model.present:
        blocked.append(
            "Tippen in Felder ist nicht möglich, weil kein Textmodell-Schlüssel gefunden wurde. "
            "Klicken, Scrollen, Navigieren und die Auswahl in Dropdowns funktionieren weiterhin. "
            "Formulare und Suchfelder bleiben bis dahin leer."
        )
    if not browser.known:
        return tuple(blocked)
    if not browser.daemon_running:
        blocked.append(
            "Kein Browser-Lauf ist möglich, weil der Browser-Harness-Daemon nicht läuft. Starte "
            "ihn, danach kann der Agent die Seite öffnen und bedienen."
        )
    elif not browser.browser_connected:
        blocked.append(
            "Kein Browser-Lauf ist möglich, weil der Daemon zwar läuft, aber kein Chrome mit ihm "
            "verbunden ist. Verbinde Chrome, danach funktionieren Klicken, Tippen und Navigieren."
        )
    return tuple(blocked)


def _summary(ready: bool, text_model: TextModelAccess, browser: BrowserStatus) -> str:
    if ready and not browser.known:
        return (
            "Ein Lauf ist möglich, der Zustand des Browsers liess sich aber nicht klären, der "
            "Agent versucht es trotzdem."
        )
    if ready and text_model.present:
        return (
            "Alle Voraussetzungen sind erfüllt, der Agent kann klicken, tippen, scrollen und "
            f"navigieren. Getippt wird mit dem Anbieter {text_model.provider} und dem Modell "
            f"{text_model.model}."
        )
    if ready:
        return "Ein Lauf ist möglich, aber eingeschränkt, weil kein Textmodell-Zugang gefunden wurde."
    return "Ein Lauf ist im Moment nicht möglich, die Gründe stehen in der Liste der blockierten Operationen."


def diagnose(
    env: Mapping[str, str] | None = None,
    config_path: object | None = None,
    probe_browser: Callable[[], BrowserStatus] | None = None,
) -> Diagnosis:
    """Den Zustand aller Voraussetzungen ermitteln. Wirft unter keinen Umständen.

    Jeder Fehler landet als ganzer Satz in `notes`, statt nach oben zu fliegen.
    Der Parametername verdeckt bewusst die Funktion `probe_browser`, denn er
    benennt genau deren Rolle.
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
            "Die Umgebung und die Konfigurationsdatei liessen sich nicht auswerten "
            f"({type(exc).__name__}), es wird deshalb angenommen, dass kein Schlüssel vorliegt."
        )
        typesafe = KeyStatus(
            present=False,
            source=None,
            variable=None,
            detail="Der TypeSafe-Schlüssel konnte nicht ermittelt werden.",
        )
        text_model = TextModelAccess(
            present=False,
            source=None,
            variable=None,
            provider=None,
            model=None,
            base_url=None,
            detail="Der Textmodell-Zugang konnte nicht ermittelt werden.",
        )
    browser, browser_notes = _browser_status(probe_browser)
    notes.extend(browser_notes)
    # Ein unbekannter Browserzustand ist kein Nein und blockiert deshalb nicht.
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
    """Alle Werte vorbereiten, bevor auch nur einer gesetzt wird."""
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
    """Prüft einen Wert, bevor er gesetzt wird. Nennt nie den Wert selbst."""
    if not isinstance(value, str):
        return f"Der Wert für die Variable {name} ist keine Zeichenkette und wird deshalb nicht gesetzt."
    if "\x00" in value:
        return (
            f"Der Wert für die Variable {name} enthält ein Nullbyte und lässt sich deshalb nicht "
            "in die Umgebung schreiben."
        )
    if "=" in name or not _NAME_PATTERN.match(name):
        return f"Der Name {name} ist als Umgebungsvariable nicht zulässig und wird nicht gesetzt."
    return None


def apply_environment(
    env: Mapping[str, str] | None = None,
    config_path: object | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> EnvironmentApplication:
    """Schreibt die aufgelösten Werte nach `os.environ`. Wirft nie.

    Muss laufen, bevor `jev_ultrafast` benutzt wird, denn die Bibliothek liest
    ihre Variablen beim Aufruf aus der Umgebung, siehe Modul-Docstring.

    Es wird erst alles vorbereitet und geprüft und danach in einem Rutsch
    gesetzt. Scheitert dabei etwas, wird der vorherige Zustand wiederhergestellt
    und `ok` ist falsch. Eine halb gesetzte Umgebung, in der zum Beispiel der
    Schlüssel des einen Anbieters auf die Basis-URL eines anderen trifft, kann
    es dadurch nicht geben. Variablen, für die kein brauchbarer Wert vorliegt,
    werden aus der Zielumgebung entfernt, damit die Diagnose und die Wirklichkeit
    dasselbe sagen. Gibt nur Namen zurück, niemals Werte.
    """
    target = os.environ if environ is None else environ
    path = DEFAULT_CONFIG_PATH if config_path is None else config_path
    try:
        values, remove, notes = _plan_environment(env, path)
    except Exception as exc:
        return EnvironmentApplication(
            ok=False,
            notes=(
                "Die Umgebung liess sich nicht vorbereiten "
                f"({type(exc).__name__}), es wurde deshalb nichts gesetzt.",
            ),
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
                "Der bisherige Zustand der Umgebung liess sich nicht sichern "
                f"({type(exc).__name__}), es wurde deshalb nichts gesetzt.",
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
                "Das Setzen der Umgebungsvariablen ist fehlgeschlagen "
                f"({type(exc).__name__}), es wurde deshalb keine einzige Variable verändert.",
                *rollback,
            ),
        )
    return EnvironmentApplication(ok=True, applied=tuple(applied), removed=tuple(removed), notes=notes)


def _rollback(
    target: MutableMapping[str, str], touched: list[str], before: dict[str, str]
) -> tuple[str, ...]:
    """Den gesicherten Zustand zurückschreiben, so weit es geht."""
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
        return (
            "Der vorherige Zustand liess sich für diese Variablen nicht wiederherstellen: "
            f"{', '.join(failed)}.",
        )
    return ()
