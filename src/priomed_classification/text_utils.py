"""Normalización de texto en español para extracción de señales."""
from __future__ import annotations

import re
import unicodedata


def _strip_accents_lower(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Minúsculas, sin tildes, sin puntuación, espacios colapsados (entrada del modelo ML)."""
    text = re.sub(r"[^a-z0-9\s]", " ", _strip_accents_lower(text))
    return re.sub(r"\s+", " ", text).strip()


# Separador de cláusula que se conserva en el texto normalizado para el guardrail.
CLAUSE_BREAK = "|"
_CLAUSE_PUNCT = re.compile(r"[.;,:!?()\[\]\n\r]+")


def normalize_with_clauses(text: str) -> str:
    """Como `normalize`, pero la puntuación que separa cláusulas se conserva como `|`.

    Así la negación de una cláusula ("Sin fiebre.") no alcanza a la siguiente
    ("Disnea severa"). Las frases de alarma no contienen `|`, por lo que la búsqueda
    de frases no se ve afectada.
    """
    text = _CLAUSE_PUNCT.sub(f" {CLAUSE_BREAK} ", _strip_accents_lower(text))
    text = re.sub(r"[^a-z0-9\s|]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Marcadores de negación que anulan una señal si aparecen justo antes de ella.
NEGATION_CUES = ("niega", "sin", "no presenta", "no refiere", "descarta", "ausencia de", "no hay", "no tiene")

# Los marcadores solo cuentan como palabras completas: "sin" no debe coincidir dentro de
# "sintomas" o "sintetico", porque ocultaría una señal de alarma real. "sin embargo" es
# un conector adversativo, no una negación.
_CUE_REGEX = {cue: re.escape(cue) for cue in NEGATION_CUES}
_CUE_REGEX["sin"] = r"sin(?! embargo\b)"
_NEGATION_PATTERN = re.compile(r"\b(?:" + "|".join(_CUE_REGEX.values()) + r")\b")

# Palabras que cierran el alcance de una negación (estilo NegEx): lo que viene después ya
# no está negado. "Paciente sin fiebre pero con disnea" -> la disnea es real.
# "ni" NO es terminador: "no refiere disnea ni desmayo" niega ambas.
SCOPE_TERMINATORS = ("pero", "aunque", "sin embargo", "y", "con", "presenta", "refiere", "ademas", "actualmente", "hoy")
_TERMINATOR_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in SCOPE_TERMINATORS) + r")\b")


def is_negated(normalized_text: str, start: int, window: int = 30) -> bool:
    """True si la ocurrencia que empieza en `start` está dentro del alcance de una negación.

    Heurística deliberadamente simple (prototipo). Una negación mal detectada que
    OCULTA una alarma real es el error costoso; por eso:
      * la ventana es corta y no cruza separadores de cláusula (`|`, ver
        `normalize_with_clauses`);
      * los marcadores se comparan como palabras completas;
      * un terminador ("pero", "y", "con", ...) entre el marcador y la frase cierra la
        negación. Ante duda, NO se descarta la señal.
    """
    lo = max(0, start - window)
    brk = normalized_text.rfind(CLAUSE_BREAK, lo, start)
    if brk != -1:
        lo = brk + 1
    # Se busca sobre el texto completo (no sobre un recorte) para que el límite de palabra
    # al inicio de la ventana se evalúe contra el carácter real que la precede.
    cues = list(_NEGATION_PATTERN.finditer(normalized_text, lo, start))
    if not cues:
        return False
    last_cue_end = cues[-1].end()
    return _TERMINATOR_PATTERN.search(normalized_text, last_cue_end, start) is None
