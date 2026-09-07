# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Text normalization for speech synthesis.

The normalizer is intentionally conservative: each transformation can be
turned off through ``NormalizationOptions`` and the output remains plain text.
"""

from __future__ import annotations

import math
import re
from typing import Optional

try:
    import inflect

    INFLECT_ENGINE = inflect.engine()
except ImportError:
    INFLECT_ENGINE = None

from ..structures.schemas import NormalizationOptions

VALID_TLDS = [
    "com", "org", "net", "edu", "gov", "mil", "int", "biz", "info", "name",
    "pro", "coop", "museum", "travel", "jobs", "mobi", "tel", "asia", "cat",
    "xxx", "aero", "arpa", "bg", "br", "ca", "cn", "de", "es", "eu", "fr",
    "in", "it", "jp", "mx", "nl", "ru", "uk", "us", "io", "co", "ai", "app",
    "dev", "me", "nz", "au",
]

VALID_UNITS = {
    "m": "meter", "cm": "centimeter", "mm": "millimeter", "km": "kilometer",
    "in": "inch", "ft": "foot", "yd": "yard", "mi": "mile",
    "g": "gram", "kg": "kilogram", "mg": "milligram", "lb": "pound", "oz": "ounce",
    "s": "second", "ms": "millisecond", "min": "minute", "h": "hour",
    "l": "liter", "ml": "milliliter", "cl": "centiliter", "dl": "deciliter",
    "kph": "kilometer per hour", "mph": "mile per hour", "m/s": "meter per second",
    "km/h": "kilometer per hour",
    "°c": "degree celsius", "°f": "degree fahrenheit", "k": "kelvin",
    "hz": "hertz", "khz": "kilohertz", "mhz": "megahertz", "ghz": "gigahertz",
    "w": "watt", "kw": "kilowatt", "mw": "megawatt", "j": "joule", "kj": "kilojoule",
    "b": "bit", "kb": "kilobit", "mb": "megabit", "gb": "gigabit", "tb": "terabit",
    "kbps": "kilobit per second", "mbps": "megabit per second",
    "gbps": "gigabit per second", "px": "pixel",
}

SYMBOL_REPLACEMENTS = {
    "~": " ", "@": " at ", "#": " number ", "$": " dollar ", "%": " percent ",
    "^": " ", "&": " and ", "*": " ", "_": " ", "|": " ", "\\": " ",
    "/": " slash ", "=": " equals ", "+": " plus ",
}

MONEY_UNITS = {
    "$": ("dollar", "cent"),
    "£": ("pound", "pence"),
    "€": ("euro", "cent"),
    "¥": ("yen", "sen"),
}

EMAIL_PATTERN = re.compile(
    r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
)

# The previous expression used ``\\.`` inside a raw string, which means
# "literal backslash + any character" rather than a domain dot. As a result,
# normal domains such as example.com were never normalized.
_TLD_PATTERN = "|".join(re.escape(tld) for tld in VALID_TLDS)
URL_PATTERN = re.compile(
    r"(?:(?:https?://|www\.)?"
    r"(?:localhost|(?:[a-zA-Z0-9-]+\.)+(?:" + _TLD_PATTERN + r")|(?:\d{1,3}\.){3}\d{1,3})"
    r"(?::\d{1,5})?(?:[/?#][^\s]*)?)",
    re.IGNORECASE,
)

_UNIT_KEYS = "|".join(
    re.escape(unit) for unit in sorted(VALID_UNITS, key=len, reverse=True)
)
UNIT_PATTERN = re.compile(
    r"((?<!\w)([+-]?)(\d{1,3}(,\d{3})*|\d+)(\.\d+)?)\s*("
    + _UNIT_KEYS
    + r")(?=[^\w\d]|\b)",
    re.IGNORECASE,
)

TIME_PATTERN = re.compile(
    r"([0-9]{1,2} ?: ?[0-9]{2}( ?: ?[0-9]{2})?)( ?(pm|am)\b)?",
    re.IGNORECASE,
)

MONEY_PATTERN = re.compile(
    r"(-?)([" + re.escape("".join(MONEY_UNITS)) + r"])(\d+(?:\.\d+)?)"
    r"((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(
    r"(-?)(\d+(?:\.\d+)?)"
    r"((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
    re.IGNORECASE,
)

# --- Markdown structure patterns -------------------------------------------
# ATX headings: 1 to 6 leading '#', optional closing '#'-run. We keep the
# capture on the heading text so we can append a pause cue after it.
MARKDOWN_HEADING_PATTERN = re.compile(
    r"^[ \t]{0,3}(#{1,6})[ \t]+([^\n]+?)[ \t]*(?:#{1,6})?[ \t]*$",
    re.MULTILINE,
)
# Setext headings: a line of text followed by a line of '=' or '-'.
# Detected *before* other list/block handling so the underline is consumed.
MARKDOWN_SETEXT_HEADING_PATTERN = re.compile(
    r"^([^\n]+)\n[ \t]*([=]{2,}|[-]{2,})[ \t]*$", re.MULTILINE)
# Unordered list bullets: '-', '*', '+' followed by a space.
MARKDOWN_UNORDERED_LIST_PATTERN = re.compile(r"^( {0,3}[-*+])[ \t]+", re.MULTILINE)
# Ordered list items: '1.', '2)', '10.' ... followed by a space.
MARKDOWN_ORDERED_LIST_PATTERN = re.compile(r"^( {0,3}\d{1,9}[.)])[ \t]+", re.MULTILINE)
# Horizontal rules: '---', '***', '___' (3+ same chars, spaces allowed).
MARKDOWN_HR_PATTERN = re.compile(
    r"^[ \t]{0,3}([-*_])([ \t]*\1){2,}[ \t]*$", re.MULTILINE
)
# Blockquotes: leading '>' optionally nested.
MARKDOWN_BLOCKQUOTE_PATTERN = re.compile(r"^( {0,3}>+)[ \t]*", re.MULTILINE)
# Emphasis / strong markers we simply unwrap (the words stay).
MARKDOWN_EMPHASIS_PATTERN = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
# Inline code spans: `code` -> code (kept as plain text).
MARKDOWN_INLINE_CODE_PATTERN = re.compile(r"`([^`\n]+)`")
# Markdown links/images: [text](url) -> text ; ![alt](url) -> alt
MARKDOWN_LINK_PATTERN = re.compile(r"!\[([^\]]*)\]\([^)]*\)|\[([^\]]*)\]\([^)]*\)")
# Fenced code blocks: ``` ... ``` (and indented code blocks of 4+ spaces).
MARKDOWN_FENCE_PATTERN = re.compile(
    r"(^|\n)([ \t]{0,3})(`{3,}|~{3,})[^\n]*\n[\s\S]*?\n[ \t]{0,3}\3[ \t]*$",
    re.MULTILINE,
)


def _has_terminal_punctuation(text: str) -> bool:
    """Return True if ``text`` already ends with sentence/clause punctuation."""
    return bool(text) and text[-1] in ".!?;:,:-"


def _strip_inline_markdown(text: str) -> str:
    """Remove inline-only Markdown markers while keeping the readable words."""
    # Inline code first (so emphasis markers inside are not re-processed).
    text = MARKDOWN_INLINE_CODE_PATTERN.sub(r"\1", text)
    # Images and links: keep the visible text/alt text.
    text = MARKDOWN_LINK_PATTERN.sub(lambda m: (m.group(1) or m.group(2) or ""), text)
    # Bold/italic markers: unwrap, preserving the inner text.
    text = MARKDOWN_EMPHASIS_PATTERN.sub(r"\2", text)
    return text


def normalize_markdown(text: str) -> str:
    """Flatten Markdown structure into spoken-friendly text with pause cues.

    The Markdown is converted to plain text where structural breaks are
    translated into punctuation the TTS model renders as audible pauses:

    * Headings keep their text and gain a trailing period so the model takes
      a breath before the body that follows.
    * Paragraph breaks (one or more blank lines) become a period.
    * List items are stripped of their bullets/numbers and separated with
      commas so each item gets its own micro-pause instead of running
      together; the end of a list gets a period.
    * Blockquote markers, horizontal rules, fenced/indented code fences,
      emphasis and link syntax are removed, keeping the readable words.

    The output still contains newlines: the rest of :func:`normalize_text`
    collapses whitespace afterwards.
    """
    if not text:
        return text

    # 1. Drop fenced code blocks entirely (replace with a blank line so the
    #    surrounding paragraphs stay separated). Code is rarely meant to be
    #    spoken verbatim and its punctuation confuses the model.
    text = MARKDOWN_FENCE_PATTERN.sub(r"\1\2\n", text)

    # 2. Setext headings: 'Title\n===' or 'Title\n---'. Must run before the
    #    horizontal-rule and unordered-list handling, which would otherwise
    #    eat the underline as a rule or a bullet line.
    def _setext(match: re.Match[str]) -> str:
        return f"# {match.group(1).strip()}"

    text = MARKDOWN_SETEXT_HEADING_PATTERN.sub(_setext, text)

    # 3. Horizontal rules -> a sentence break. Consume the surrounding blank
    #    lines so we do not emit an empty "sentence" between two pauses.
    text = re.sub(
        r"(?:\n[ \t]*)?^[ \t]{0,3}([-*_])([ \t]*\1){2,}[ \t]*$(?:[ \t]*\n)?",
        ".\n\n",
        text,
        flags=re.MULTILINE,
    )

    # 4. Headings: strip the '#'-run, keep the text, ensure a trailing period.
    def _heading(match: re.Match[str]) -> str:
        heading_text = _strip_inline_markdown(match.group(2).strip())
        if not heading_text:
            return ""
        if not _has_terminal_punctuation(heading_text):
            heading_text += "."
        # Force a paragraph break after every heading so the body that follows
        # is synthesized as a separate chunk (real silence) rather than being
        # comma-joined to the heading by the list/prose run pass below.
        return f"{heading_text}\n\n"

    text = MARKDOWN_HEADING_PATTERN.sub(_heading, text)

    # 5. Lists. Each list item becomes its own block (its own line, preceded
    #    by a blank line) so the chunker synthesizes it as a separate chunk with
    #    a real pause before it.
    #      * Unordered items ('-', '*', '+') drop their bullet.
    #      * Ordered items ('1.', '2.', '1)', ...) speak their ordinal first:
    #        "one, first", "two, second", ...
    #    Indentation (nested lists) is preserved only in that nested items are
    #    still recognized as items; each is emitted as its own block regardless
    #    of depth. A trailing period is added if the item lacks terminal
    #    punctuation.
    _unordered_re = re.compile(r"^([ \t]*)([-*+])[ \t]+(.*)$")
    _ordered_re = re.compile(r"^([ \t]*)(\d{1,9})[.)][ \t]+(.*)$")

    raw_lines = text.split("\n")
    out: list[str] = []
    for line in raw_lines:
        om = _ordered_re.match(line)
        um = _unordered_re.match(line)
        if om:
            number = _number_to_words(om.group(2))
            item = f"{number}, {om.group(3).strip()}"
        elif um:
            item = um.group(3).strip()
        else:
            out.append(line)
            continue
        if item and not _has_terminal_punctuation(item):
            item += "."
        # Each list item is its own block: separate it from the previous line
        # with a blank line so the chunker gives it its own chunk (and a real
        # pause before it).
        if out and out[-1].strip():
            out.append("")
        out.append(item)
    text = "\n".join(out)

    # 6. Blockquote markers: remove the leading '>' runs.
    text = MARKDOWN_BLOCKQUOTE_PATTERN.sub("", text)

    # 7. Remaining inline markdown (links, emphasis, inline code) on each
    #    line that was not a structural element.
    text = _strip_inline_markdown(text)

    # 8. Paragraph / block separation. A blank line (possibly with whitespace)
    #    marks a structural break. We keep the blank line so that step 9 can
    #    tell paragraphs (separated by a blank line) from list-item runs (joined
    #    by a single newline) and so the downstream chunker can split paragraphs
    #    into separate chunks and insert actual silence between them. If the
    #    paragraph lacks terminal punctuation we add a period so it still reads
    #    as a finished sentence. Paragraphs that became only punctuation (e.g.
    #    a leftover horizontal-rule break) are dropped.
    def _paragraph_break(match: re.Match[str]) -> str:
        before = match.group(1).strip()
        if not before:
            return "\n\n"
        if _has_terminal_punctuation(before):
            return f"{before}\n\n"
        return f"{before}.\n\n"

    text = re.sub(r"([^\n]*?)\n[ \t]*\n+", _paragraph_break, text)
    # Drop lines that became only punctuation after the steps above.
    text = re.sub(r"\n[ \t]*[.!?:;,-][ \t]*(?=\n)", "\n", text)

    # 9. Per-line punctuation. After step 8, paragraph blocks are separated
    #    by blank lines and lists (handled in step 5) are already collapsed
    #    into single lines. Any remaining non-blank line is a standalone
    #    block (a heading, a list, or a prose line the author put on its own
    #    line). We make sure each such line ends with terminal punctuation so
    #    it reads as a finished sentence; the downstream chunker then gives
    #    every line its own chunk with real silence around it. We do NOT
    #    comma-join consecutive lines here: that used to merge a lead-in line
    #    into the following list/prose, swallowing the pause between them.
    lines = text.split("\n")
    out_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not _has_terminal_punctuation(stripped):
            out_lines.append(line.rstrip() + ".")
        else:
            out_lines.append(line)
    text = "\n".join(out_lines)

    return text


def _number_to_words(number) -> str:
    if INFLECT_ENGINE:
        return INFLECT_ENGINE.number_to_words(number)
    return str(number)


def _plural(word: str, count=2) -> str:
    if INFLECT_ENGINE:
        return INFLECT_ENGINE.plural(word, count)
    if abs(float(count)) == 1:
        return word
    irregular = {
        "foot": "feet", "inch": "inches", "pence": "pence",
        "hertz": "hertz", "sen": "sen",
    }
    if word in irregular:
        return irregular[word]
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def _no(word: str, count) -> str:
    if INFLECT_ENGINE:
        return INFLECT_ENGINE.no(word, count)
    try:
        numeric_count = float(str(count).replace(",", ""))
    except ValueError:
        numeric_count = 2
    return f"{count} {_plural(word, numeric_count)}"


def conditional_int(number: float, threshold: float = 0.00001) -> int | float:
    if abs(round(number) - number) < threshold:
        return int(round(number))
    return number


def translate_multiplier(multiplier: str) -> str:
    translation = {"k": "thousand", "m": "million", "b": "billion", "t": "trillion"}
    stripped = multiplier.strip()
    return translation.get(stripped.lower(), stripped)


def split_four_digit(number: float) -> str:
    digits = str(int(abs(number)))
    return f"{_number_to_words(digits[:2])} {_number_to_words(digits[2:])}"


def handle_units(match: re.Match[str]) -> str:
    unit_string = match.group(6).strip()
    unit_name = VALID_UNITS.get(unit_string.lower())
    if not unit_name:
        return f"{match.group(1)} {unit_string}"

    parts = unit_name.split(" ")
    # Lowercase b means bit; uppercase B means byte (KB, MB, GB, ...).
    if parts[0].endswith("bit") and unit_string[-1:] == "B":
        parts[0] = parts[0][:-3] + "byte"

    number = match.group(1).replace(",", "")
    parts[0] = _no(parts[0], number)
    return " ".join(parts)


def handle_numbers(match: re.Match[str]) -> str:
    try:
        number = float(match.group(2))
    except ValueError:
        return match.group()

    if match.group(1) == "-":
        number *= -1
    multiplier = translate_multiplier(match.group(3))
    number = conditional_int(number)

    if multiplier:
        return f"{_number_to_words(number)} {multiplier}"

    if (
        isinstance(number, int)
        and len(str(abs(number))) == 4
        and abs(number) > 1500
        and abs(number) % 1000 > 9
    ):
        spoken = split_four_digit(number)
        return f"minus {spoken}" if number < 0 else spoken

    return _number_to_words(number)


def handle_money(match: re.Match[str]) -> str:
    bill, coin = MONEY_UNITS[match.group(2)]
    try:
        value = float(match.group(3))
    except ValueError:
        return match.group()

    negative = match.group(1) == "-"
    value = abs(value)
    multiplier = translate_multiplier(match.group(4))

    if multiplier or value.is_integer():
        count = conditional_int(value)
        phrase = f"{_number_to_words(count)}"
        if multiplier:
            phrase += f" {multiplier}"
        phrase += f" {_plural(bill, count=count)}"
    else:
        whole = int(math.floor(value))
        cents = int(round((value - whole) * 100))
        if cents == 100:
            whole += 1
            cents = 0
        phrase = f"{_number_to_words(whole)} {_plural(bill, count=whole)}"
        if cents:
            phrase += f" and {_number_to_words(cents)} {_plural(coin, count=cents)}"

    return f"minus {phrase}" if negative else phrase


def handle_decimal(match: re.Match[str]) -> str:
    whole, fraction = match.group().split(".", 1)
    return f"{whole} point {' '.join(fraction)}"


def handle_email(match: re.Match[str]) -> str:
    user, domain = match.group(0).split("@", 1)
    return f"{user} at {domain.replace('.', ' dot ')}"


def handle_url(match: re.Match[str]) -> str:
    url = match.group(0).strip()
    url = re.sub(
        r"^https?://",
        lambda item: "https " if item.group().lower().startswith("https") else "http ",
        url,
        flags=re.IGNORECASE,
    )
    url = re.sub(r"^www\.", "www ", url, flags=re.IGNORECASE)
    url = re.sub(r":(\d+)(?=/|$)", lambda item: f" colon {item.group(1)}", url)

    domain, separator, path = url.partition("/")
    domain = domain.replace(".", " dot ")
    url = f"{domain} slash {path}" if separator else domain

    replacements = {
        "-": " dash ", "_": " underscore ", "?": " question mark ",
        "#": " hash ", "=": " equals ", "&": " and ", "%": " percent ",
        ":": " colon ", "/": " slash ",
    }
    for symbol, spoken in replacements.items():
        url = url.replace(symbol, spoken)
    return re.sub(r"\s+", " ", url).strip()


def handle_phone_number(match: re.Match[str]) -> str:
    groups = list(match.groups())
    parts = []
    if groups[0] is not None:
        parts.append(_number_to_words(groups[0].replace("+", "")))

    area = groups[2].replace("(", "").replace(")", "")
    for value in (area, groups[3], groups[4]):
        if INFLECT_ENGINE:
            parts.append(INFLECT_ENGINE.number_to_words(value, group=1, comma=""))
        else:
            parts.append(" ".join(value))
    return ", ".join(parts)


def handle_time(match: re.Match[str]) -> str:
    groups = match.groups()
    time_parts = groups[0].split(":")
    hour = time_parts[0].strip()
    minute = time_parts[1].strip()
    minute_value = int(minute)

    numbers = [_number_to_words(hour)]
    if minute_value == 0:
        if len(time_parts) == 2:
            numbers.append("o'clock")
    elif minute_value < 10:
        numbers.append(f"oh {_number_to_words(minute)}")
    else:
        numbers.append(_number_to_words(minute))

    if len(time_parts) > 2:
        seconds = int(time_parts[2].strip())
        numbers.append(f"and {_number_to_words(seconds)} {_plural('second', seconds)}")
    elif groups[2] is not None:
        numbers.append(groups[2].strip())

    return " ".join(numbers)


def normalize_text(text: str, options: Optional[NormalizationOptions] = None) -> str:
    """Normalize URLs, email, numbers, units, punctuation, and symbols for TTS."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    options = options or NormalizationOptions()
    if not options.normalize:
        return text

    if options.markdown_normalization:
        text = normalize_markdown(text)

    if options.email_normalization:
        text = EMAIL_PATTERN.sub(handle_email, text)
    if options.url_normalization:
        text = URL_PATTERN.sub(handle_url, text)

    # Currency must be normalized before units. Otherwise phrases such as
    # "$1.50 in fees" can be misread as "1.50 inches".
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = MONEY_PATTERN.sub(handle_money, text)

    if options.unit_normalization:
        text = UNIT_PATTERN.sub(handle_units, text)
    if options.optional_pluralization_normalization:
        text = re.sub(r"\(s\)", "s", text)
    if options.phone_normalization:
        text = re.sub(
            r"(\+?\d{1,2})?([ .-]?)(\(?\d{3}\)?)[\s.-](\d{3})[\s.-](\d{4})",
            handle_phone_number,
            text,
        )

    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("«", '"').replace("»", '"').replace("“", '"').replace("”", '"')
    for source, target in zip("、。！，：；？–", ",.!,:;?-"):
        text = text.replace(source, target + " ")

    text = TIME_PATTERN.sub(handle_time, text)
    text = re.sub(r"[^\S \n]", " ", text)
    # Newline handling. With markdown normalization on, paragraph breaks are
    # meaningful: collapse runs of blank lines into a single newline and KEEP
    # it as a hard separator so the downstream chunker splits paragraphs into
    # separate chunks and inserts real silence between them. With markdown
    # normalization off we preserve the legacy behavior of flattening all
    # newlines into spaces.
    text = text.replace("\r", " ")
    if options.markdown_normalization:
        text = re.sub(r"\n\s*\n+", "\n", text)
    else:
        text = text.replace("\n", " ")

    text = re.sub(r"\bD[Rr]\.(?= [A-Z])", "Doctor", text)
    text = re.sub(r"\b(?:Mr\.|MR\.(?= [A-Z]))", "Mister", text)
    text = re.sub(r"\b(?:Ms\.|MS\.(?= [A-Z]))", "Miss", text)
    text = re.sub(r"\b(?:Mrs\.|MRS\.(?= [A-Z]))", "Mrs", text)
    text = re.sub(r"\betc\.(?! [A-Z])", "etc", text)
    text = re.sub(r"(?i)\b(y)eah?\b", r"\1e'a", text)

    text = NUMBER_PATTERN.sub(handle_numbers, text)
    text = re.sub(r"\d*\.\d+", handle_decimal, text)

    if options.replace_remaining_symbols:
        for symbol, replacement in SYMBOL_REPLACEMENTS.items():
            text = text.replace(symbol, replacement)

    text = re.sub(r"(?<=\d)-(?=\d)", " to ", text)
    text = re.sub(r"(?<=\d)S", " S", text)
    text = re.sub(r"(?<=[BCDFGHJ-NP-TV-Z])'?s\b", "'S", text)
    text = re.sub(r"(?<=X')S\b", "s", text)
    text = re.sub(r"(?:[A-Za-z]\.){2,} [a-z]", lambda m: m.group().replace(".", "-"), text)
    text = re.sub(r"(?i)(?<=[A-Z])\.(?=[A-Z])", "-", text)
    return re.sub(r"\s{2,}", " ", text).strip()
