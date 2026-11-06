
"""Pure deterministic routing rules; intentionally independent of LangChain.

The rules are deliberately conservative: mentioning a flight or hotel is not
by itself an instruction to search. A specialist is selected only when the
message contains a reasonably explicit service request or a clear follow-up.
"""
from __future__ import annotations

import re


_SERVICE_TOPICS = {
    "flight": r"(?:flights?|airfare|air\s*tickets?|plane\s*tickets?)",
    "hotel": r"(?:hotels?|accommodations?|hostels?|resorts?|guesthouses?|lodging)",
}


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split()).strip()


_TERMINAL_COMMAND_PATTERNS = (
    # Python and package-management commands commonly pasted into the chat by mistake.
    r"^(?:python(?:[0-9]+(?:\.[0-9]+)*)?|py)\s+-m\s+(?:pip|venv|pytest|unittest)\b",
    r"^(?:python(?:[0-9]+(?:\.[0-9]+)*)?|py)\s+[\w./\\-]+\.py(?:\s+.*)?$",
    r"^(?:pip|pip[0-9]+)\s+(?:install|uninstall|show|list|freeze|check|download|cache)\b",
    r"^(?:conda|mamba|uv|poetry)\s+(?:install|add|remove|sync|lock|run|update)\b",
    r"^(?:npm|npx|yarn|pnpm)\s+(?:install|add|remove|run|exec|update)\b",
    r"^(?:git|docker|kubectl|winget|choco|scoop)\s+[a-z][a-z0-9_-]*\b",
    r"^(?:install|update|upgrade)\s+(?:the\s+)?(?:requirements(?:\.txt)?|dependencies|packages?)\b",
    r"^(?:pytest|unittest)\s+(?:-\w+\s+)*[\w./\\-]*test[\w./\\-]*",
)


def is_terminal_command_request(text: str) -> bool:
    """Recognize shell/developer commands that must not enter a travel specialist.

    This check runs before contextual follow-up routing. It deliberately targets
    command-shaped messages rather than ordinary questions about programming.
    """
    value = _normalize(text).strip().strip('`\"\'')
    if not value:
        return False

    # Strip common pasted shell prompts while preserving the command itself.
    value = re.sub(r"^(?:\(venv\)\s*)?(?:ps\s+[a-z]:[^>]*>|[a-z]:\\[^>]*>)\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^\$\s*", "", value)
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in _TERMINAL_COMMAND_PATTERNS)


_SQL_STATEMENT_PATTERNS = (
    # Require a complete-looking SQL SELECT rather than treating natural
    # language such as "select flights from COK to KUL" as a SQL statement.
    r"^select\s+(?:\*|distinct\b|top\s+\d+\b|"
    r"(?:count|sum|avg|min|max)\s*\(|"
    r"[a-z_][\w$]*(?:\s*,\s*[a-z_][\w$]*)*)\s+from\s+[a-z_][\w$]*"
    r"(?:\s+(?:where|join|left\s+join|right\s+join|inner\s+join|full\s+join|group\s+by|order\s+by|having|limit|offset|union)\b.*)?\s*;?$",
    r"^insert\s+into\s+[a-z_][\w$]*\s*(?:\([^)]*\)\s*)?values\b",
    r"^update\s+[a-z_][\w$]*\s+set\b",
    r"^delete\s+from\s+[a-z_][\w$]*\b",
    r"^create\s+(?:or\s+replace\s+)?(?:table|view|index|database)\b",
    r"^alter\s+table\s+[a-z_][\w$]*\b",
    r"^drop\s+(?:table|view|index|database)\b",
    r"^truncate\s+(?:table\s+)?[a-z_][\w$]*\b",
    r"^with\s+[a-z_][\w$]*\s+as\s*\(",
)


# Language aliases are intentionally broad. Short/ambiguous names (C, R, Go,
# D, Java) are only considered programming languages when programming intent
# is explicit, so destinations such as Java are not accidentally rejected.
_PROGRAMMING_LANGUAGES = (
    "python", "javascript", "typescript", "java", "kotlin", "swift",
    "objective-c", "objective c", "c++", "cpp", "c#", "csharp", "c sharp",
    "golang", "rust", "ruby", "php", "dart", "scala", "perl", "haskell",
    "lua", "matlab", "fortran", "cobol", "pascal", "delphi", "visual basic",
    "vb.net", "f#", "f sharp", "elixir", "erlang", "clojure", "lisp", "scheme",
    "prolog", "julia", "groovy", "solidity", "pl/sql", "tsql", "t-sql", "vba", "go",
    "sas", "ocaml", "fsharp", "crystal", "nim", "zig", "objective-c", "assembly",
    "asm", "gdscript", "apex", "abap", "scratch", "actionscript", "powershell",
    "bash", "shell scripting", "shell script", "sql", "html", "css", "scss",
    "brainfuck", "befunge", "intercal", "whitespace", "malbolge", "piet",
    "lolcode", "ook", "shakespeare", "chef", "odin", "mojo", "carbon",
    "pony", "hack", "q#", "lean", "idris", "coq", "agda", "elm",
    "purescript", "reason", "rescript", "forth", "modula-2", "pl/i",
)

_AMBIGUOUS_LANGUAGE_NAMES = {"java", "go", "c", "r", "d", "ada", "basic"}


def is_travel_destination_context(text: str) -> bool:
    """Return True when an ambiguous language name is clearly a travel place.

    Examples include "plan a trip to Java" and "Kochi to Java, one-way, alone".
    "What is Java?" remains a programming question.
    """
    value = _normalize(text)
    return bool(value) and any(
        re.search(pattern, value, re.IGNORECASE)
        for pattern in _TRAVEL_DESTINATION_OVERRIDE_PATTERNS
    )


_PROGRAMMING_INTENT_PATTERNS = (
    r"\b(?:programming|coding|software\s+development|source\s+code|computer\s+programming)\b",
    r"\b(?:teach|explain|help|guide)\s+me\s+(?:to\s+)?(?:code|program|debug|compile)\b",
    r"\b(?:write|create|generate|build|implement|debug|compile|run|execute|review|fix|refactor)\b.{0,100}\b(?:code|program|script|function|class|module|application|app|compiler|interpreter|syntax)\b",
    r"\b(?:show|give|provide)\s+(?:me\s+)?(?:an?\s+)?(?:code|source\s+code|program|script|function|class|syntax|compiler|interpreter|example)\b",
    r"\b(?:code|program|script|function|class|module|syntax|compiler|interpreter|framework|library|package|exception|traceback|runtime|variable|loop|array|pointer|recursion|algorithm|data\s+structure)\b.{0,80}\b(?:in|using|with|for)\b",
    r"\b(?:what\s+is|what\s+are|explain|define|teach\s+me|learn|study|compare|difference\s+between|how\s+does|how\s+do\s+i\s+use)\b.{0,80}\b(?:programming\s+language|coding|syntax|compiler|interpreter|code|programming)\b",
    r"\b(?:program|code|script|syntax|compiler|interpreter|function|class|variable|loop|array|pointer)\s+(?:in|using|with)\s+[a-z][a-z0-9+#. -]{0,30}\b",
    r"\b(?:in|using|with)\s+[a-z][a-z0-9+#. -]{0,30}\s+(?:programming\s+language|code|syntax|program|script)\b",
    r"\b(?:what\s+is|explain|teach\s+me|learn|study|compare|debug|write\s+code\s+in)\s+(?:the\s+)?(?:[a-z][a-z0-9+#. -]{0,25})\s+programming\s+language\b",
    r"\b(?:how\s+to\s+use|using|learn|study|programming\s+in)\s+(?:the\s+)?(?:r|c|go|d)\b.{0,60}\b(?:programming|code|script|data\s+analysis|statistics|syntax|function|package)\b",
    r"\b(?:pointer|closure|generics?|type\s+system|memory\s+management|inheritance|polymorphism|recursion|lambda|decorator|annotation)\b.{0,60}\b(?:example|examples|in|using|with|for)\s+[a-z][a-z0-9+#. -]{0,25}\b",
)


_TRAVEL_DESTINATION_OVERRIDE_PATTERNS = (
    # Explicit travel verbs or trip nouns before an ambiguous place name.
    r"\b(?:travel|trip|vacation|holiday|visit|fly|flying|flight|flights|hotel|hotels|itinerary|destination|plan)\b.{0,70}\b(?:to|in|around|for)\s+(?:java|goa|go|r|c|d|ada)\b",
    # Route-shaped travel statements such as "Kochi to Java". Require an
    # additional travel clue so programming statements containing "to Java"
    # are not automatically classified as travel.
    r"\b(?:from\s+)?[a-z][a-z .'-]{1,30}\s+to\s+java\b.{0,120}\b(?:one[- ]way|round[- ]trip|alone|adult|passenger|flight|travel|trip|date|today|tomorrow|after\s+\d+\s+days?)\b",
    r"\b(?:to|in|around)\s+(?:java|goa|go|r|c|d|ada)\s+(?:island|islands|city|airport|indonesia|country|region)\b",
    r"\b(?:java|go)\s+(?:island|islands|province|indonesia|itinerary|travel|tourism|holiday|vacation)\b",
)


def is_programming_related_request(text: str) -> bool:
    """Detect programming questions for mainstream, obscure and future languages.

    We combine language aliases with language-agnostic intent patterns. A
    language name alone is not enough for ambiguous words such as Java, Go,
    C, R or D unless the message is phrased as a programming/language question.
    """
    value = " ".join((text or "").lower().split()).strip()
    if not value:
        return False

    # Explicit travel-destination wording wins for names that overlap with
    # programming languages (Java, Go, etc.).
    if is_travel_destination_context(value):
        return False

    if any(re.search(pattern, value, re.IGNORECASE) for pattern in _PROGRAMMING_INTENT_PATTERNS):
        return True

    # A comparison between two recognized language names is a programming query.
    if re.search(r"\b(?:difference\s+between|compare|versus|vs\.?)\b", value):
        recognized = 0
        for alias in _PROGRAMMING_LANGUAGES:
            token = rf"(?<![a-z0-9_]){re.escape(alias)}(?![a-z0-9_])"
            if re.search(token, value):
                recognized += 1
            if recognized >= 2:
                return True

    for language in _PROGRAMMING_LANGUAGES:
        language_re = re.escape(language)
        # Use non-word boundaries so punctuated names such as C++, C#, F# and PL/I work.
        language_token = rf"(?<![a-z0-9_]){language_re}(?![a-z0-9_])"
        question_pattern = rf"\b(?:what\s+is|what\s+about|explain|define|teach\s+me|learn|study|compare)\s+(?:the\s+)?{language_token}"
        help_pattern = rf"\b(?:i\s+)?(?:need|want)\s+help\s+with\s+{language_token}"
        technical_pattern = rf"{language_token}.{{0,45}}\b(?:programming|language|code|coding|syntax|compiler|interpreter|tutorial|course|example|examples|function|class|variable|loop|array|pointer|closure|type\s+system|recursion|oop|object[- ]oriented|exception|inheritance|dictionary|tuple|list|package|module)\b"
        reverse_pattern = rf"\b(?:program|code|script|function|class|syntax|compiler|interpreter|tutorial|example|examples)\s+(?:in|using|with|for)\s+{language_token}"
        learning_pattern = rf"\b(?:learn|study|teach\s+me|write\s+code\s+in|program\s+in|code\s+in|learning|how\s+to\s+use|how\s+do\s+i\s+use)\s+{language_token}"
        topic_then_language = rf"\b(?:what\s+is|explain|show|teach|debug|write|create|implement|how\s+do\s+i|how\s+to)\b.{{0,100}}\b(?:in|using|with)\s+{language_token}"

        if re.search(question_pattern, value) or re.search(help_pattern, value) or re.search(technical_pattern, value) or re.search(reverse_pattern, value) or re.search(learning_pattern, value) or re.search(topic_then_language, value):
            return True

    # Single-word, unambiguous language names are likely language queries
    # when supplied as a short answer while a specialist is awaiting input.
    unambiguous_single_names = {
        "python", "javascript", "typescript", "kotlin", "swift", "rust", "ruby",
        "php", "dart", "scala", "perl", "haskell", "lua", "matlab", "fortran",
        "cobol", "pascal", "elixir", "erlang", "clojure", "lisp", "scheme",
        "prolog", "julia", "groovy", "solidity", "ocaml", "crystal", "nim", "zig",
        "golang", "assembly", "bash", "powershell", "sql", "html", "css",
        "brainfuck", "befunge", "intercal", "whitespace", "malbolge", "piet",
        "lolcode", "odin", "mojo", "carbon", "pony", "lean", "idris", "coq",
        "agda", "elm", "purescript", "reason", "rescript", "forth",
    }
    if value in unambiguous_single_names:
        return True

    return False


def is_code_or_query_snippet(text: str) -> bool:
    """Recognize pasted source code/SQL across languages using generic syntax cues."""
    value = (text or "").strip()
    if not value:
        return False

    # Any explicitly labelled fenced code block is code, including obscure or
    # future language identifiers. Plain text fences are not automatically code.
    fence = re.search(r"```\s*([A-Za-z][A-Za-z0-9_+#.\-]*)\s*\n", value)
    if fence and fence.group(1).lower() not in {"text", "txt", "plain", "plaintext", "markdown", "md"}:
        return True
    if re.search(r"```\s*(?:sql|python|py|javascript|js|typescript|ts|java|c|cpp|csharp|go|rust|bash|sh)\b", value, re.IGNORECASE):
        return True

    normalized = " ".join(value.lower().split()).strip('`"\' ')
    normalized = re.sub(r"^(?:please\s+)?(?:run|execute)\s*:\s*", "", normalized)

    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in _SQL_STATEMENT_PATTERNS):
        return True

    # Language-neutral / cross-ecosystem entry points and compiler signatures.
    code_prefixes = (
        r"^(?:#include\s*[<\"]|using\s+namespace\s+std\s*;|int\s+main\s*\(|void\s+main\s*\(|public\s+(?:static\s+)?class\s+|public\s+static\s+void\s+main\s*\()",
        r"^(?:package\s+main\b|func\s+main\s*\(|fn\s+(?:main|[a-z_]\w*)\s*\(|pub\s+fn\s+|fun\s+main\s*\()",
        r"^(?:(?:public|private|protected|internal|abstract|final|static)\s+)*(?:def|async\s+def|class|interface|enum|struct|namespace)\s+[a-z_]\w*\s*[(:{]",
        r"^(?:from\s+[a-z_][\w.]*\s+import|import\s+[a-z_][\w., ]*)\b",
        r"^(?:const|let|var)\s+[a-z_$][\w$]*\s*[:=]",
        r"^(?:function\s+[a-z_$][\w$]*\s*\(|export\s+(?:default\s+)?(?:function|class|const|let|var)\b)",
        r"^(?:console\.log|print|printf|println!?|echo|system\.out\.println|fmt\.println|writeln!?|cout\s*<<)\s*[!(]",
        r"^(?:<\?php|<!doctype\s+html|<html\b|<script\b|<style\b)",
        r"^(?:#!\s*/|#!.*\b(?:bash|sh|python|node|ruby|perl)\b)",
        r"^(?:insert\s+into\s+|update\s+\w+\s+set\s+|delete\s+from\s+)",
    )
    if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in code_prefixes):
        return True

    # Generic multi-line source: several lines with indentation plus operators,
    # braces, declarations or terminators. Avoid flagging ordinary prose.
    lines = [line for line in value.splitlines() if line.strip()]
    if len(lines) >= 2:
        indented = sum(1 for line in lines if line.startswith(("    ", "\t")))
        syntax_lines = sum(
            1 for line in lines
            if re.search(r"(?:[{};]|=>|:=|\breturn\b|\bthen\b|\bend\b|\bfunction\b|\bfunc\b|\bfn\b|\bdef\b|\bvar\b|\blet\b|\bconst\b)", line, re.IGNORECASE)
        )
        if indented >= 1 and syntax_lines >= 1:
            return True

    # Strong source-like single-line patterns, language independent.
    if re.search(r"\b(?:return|function|func|fn|def|var|let|const)\b.{0,100}(?:\(|=>|:=|=|\{)", normalized):
        return True
    if re.search(r"^[\w.$\[\]]+\s*(?:=|:=|\+=|-=|\*=|/=)\s*[^=].*[;{}()]?$", normalized) and not re.fullmatch(r"[a-z ]+", normalized):
        return True

    return False


def is_cancel_request(text: str) -> bool:
    value = _normalize(text).strip(".! ")
    exact = {
        "cancel", "cancel it", "cancel that", "stop", "stop it", "stop searching",
        "abort", "never mind", "nevermind", "forget it", "clear this", "don't search",
        "do not search", "no, stop", "no stop",
    }
    if value in exact:
        return True
    return bool(re.fullmatch(
        r"(?:please\s+)?(?:cancel|stop|abort|never mind|nevermind|forget|clear)"
        r"(?:\s+(?:(?:the|my|this|that)\s+)?(?:search|request|trip|flight(?:s)?(?:\s+search|\s+booking)?|hotel(?:s)?(?:\s+search|\s+reservation)?|booking|reservation|planning|it))?",
        value,
    ))


def _split_independent_clauses(value: str) -> list[str]:
    """Split clauses so a negative service does not cancel a positive one.

    Keep ``or`` inside a clause: ``I don't want flights or hotels`` negates
    both. Split commas/semicolons and contrast words such as ``but``.
    """
    return [part.strip() for part in re.split(r"\s+\b(?:but|however|instead)\b\s+|[,;]+", value) if part.strip()]


def _topic_is_negated_in_clause(clause: str, topic: str, other_topic: str) -> bool:
    # The service noun must immediately follow a service-selection/action
    # phrase. This intentionally does NOT match ordinary worries such as
    # "I don't want to miss my flight" or "I don't want my flight delayed".
    article = r"(?:(?:any|the|a|an)\s+)?"
    service_tail = r"(?=\s*(?:$|[.!?;]|,|today\b|tomorrow\b|right\s+now\b|at\s+all\b|anymore\b|any\s+more\b|(?:recommendations?|suggestions?|results?|options?|search(?:es)?|bookings?)\b))"
    want_need_prefix = (
        r"\b(?:i|we)\s+(?:do\s+not|don't|dont)\s+(?:want|need|prefer)\s+"
        r"(?:(?:to\s+)?(?:search(?:\s+for)?|look\s+for|find|book|compare|reserve|include|use)\s+)?"
    )
    direct_action_prefix = (
        r"\b(?:do\s+not|don't|dont)\s+"
        r"(?:search(?:\s+for)?|look\s+for|find|book|compare|reserve|include|use|recommend|suggest|show|check|get|give)\s+"
    )
    indirect_prefixes = (
        r"\b(?:not\s+interested\s+in|no\s+need\s+for|avoid|skip|without|stop|cancel)\s+",
        r"^\s*not\s+",
        r"^\s*no\s+(?:more\s+)?",
    )

    patterns = [
        want_need_prefix + article + topic + service_tail,
        direct_action_prefix + article + topic + service_tail,
        r"\bno\s+need\s+to\s+(?:search(?:\s+for)?|find|book|compare)\s+" + article + topic + service_tail,
        rf"\bno\s+{topic}\s+search(?:es)?\s+(?:is\s+)?(?:needed|required)\b",
    ]
    patterns.extend(prefix + article + topic + service_tail for prefix in indirect_prefixes)

    if any(re.search(pattern, clause, re.IGNORECASE) for pattern in patterns):
        return True

    # Coordinated exclusion: "I don't want flights or hotels" or the reverse.
    pair_pattern = (
        r"\b(?:i|we)\s+(?:do\s+not|don't|dont)\s+(?:want|need|prefer)\s+"
        r"(?:(?:to\s+)?(?:search(?:\s+for)?|look\s+for|find|book|compare|reserve)\s+)?"
        + article + r"(?:" + topic + r")\s+or\s+(?:any\s+|the\s+|a\s+|an\s+)?(?:" + other_topic + r")\b"
    )
    return bool(re.search(pair_pattern, clause, re.IGNORECASE))


def negated_services(text: str) -> list[str]:
    """Return services the user explicitly says they do not want searched.

    Result order is stable: flight first, hotel second.
    """
    value = _normalize(text)
    if not value:
        return []

    clauses = _split_independent_clauses(value)
    paired_exclusion_prefix = (
        r"\b(?:i|we)\s+(?:do\s+not|don't|dont)\s+(?:want|need|prefer)\s+"
        r"(?:(?:to\s+)?(?:search(?:\s+for)?|look\s+for|find|book|compare|reserve)\s+)?"
        r"(?:(?:any|the|a|an)\s+)?"
    )
    flight_topic = _SERVICE_TOPICS["flight"]
    hotel_topic = _SERVICE_TOPICS["hotel"]

    both_declined = any(
        re.search(
            paired_exclusion_prefix + rf"(?:{first})\s+or\s+(?:(?:any|the|a|an)\s+)?(?:{second})\b",
            clause,
            re.IGNORECASE,
        )
        for clause in clauses
        for first, second in ((flight_topic, hotel_topic), (hotel_topic, flight_topic))
    )
    if both_declined:
        return ["flight", "hotel"]

    result: list[str] = []
    for service, topic in _SERVICE_TOPICS.items():
        other = _SERVICE_TOPICS["hotel" if service == "flight" else "flight"]
        if any(_topic_is_negated_in_clause(clause, topic, other) for clause in clauses):
            result.append(service)
    return result


def service_request_is_negated(value: str, service: str) -> bool:
    if service not in _SERVICE_TOPICS:
        raise ValueError("service must be 'flight' or 'hotel'")
    return service in negated_services(value)


def has_negated_service_request(value: str) -> bool:
    return bool(negated_services(value))


def negative_service_route(text: str) -> dict | None:
    """Return a deterministic route when the user explicitly declines a service.

    If the same message asks for another service (e.g. "no flights, find
    hotels"), route only to that positive specialist. Otherwise return the
    fixed-response node so no LLM or search provider is called.
    """
    declined = negated_services(text)
    if not declined:
        return None

    positive_specialists = explicit_specialists(text)
    if positive_specialists:
        next_agent = positive_specialists[0]
    elif re.search(
        r"\b(?:itinerary|day[- ]by[- ]day|places to visit|things to do|"
        r"plan (?:a )?(?:trip|vacation|holiday)|trip plan)\b",
        _normalize(text),
    ):
        next_agent = "general_assistant"
    else:
        next_agent = "service_preference"

    return {
        "declined_services": declined,
        "required_agents": positive_specialists if positive_specialists else (
            ["general_assistant"] if next_agent == "general_assistant" else []
        ),
        "next_agent": next_agent,
    }


def _is_likely_air_route(value: str) -> bool:
    """Detect a city-to-city route without mistaking normal prose for a route.

    The old broad ``word to word`` expression interpreted phrases such as
    "planning to stay" and "want to travel" as airport routes whenever a date
    appeared later in the sentence. Accept an explicit ``from X to Y`` route,
    or a route-only message beginning with ``X to Y``; reject common verbs and
    pronouns as route endpoints.
    """
    patterns = (
        r"\bfrom\s+(?P<origin>[a-z][a-z .'-]{1,30}?)\s+to\s+(?P<destination>[a-z][a-z .'-]{1,40})\b",
        r"^(?P<origin>[a-z][a-z .'-]{1,30}?)\s+to\s+(?P<destination>[a-z][a-z .'-]{1,40})\b",
    )
    non_place_words = {
        "i", "iam", "im", "we", "you", "they", "he", "she", "it",
        "am", "is", "are", "was", "were", "planning", "plan", "want",
        "need", "going", "go", "travel", "travelling", "traveling",
        "visiting", "visit", "stay", "staying", "hope", "hoping",
        "intend", "intending", "looking", "trying", "would", "could",
        "should", "like", "love", "wish", "book", "booking", "search",
        "find", "the", "a", "an",
    }
    non_destination_verbs = {
        "stay", "staying", "travel", "travelling", "traveling", "visit",
        "visiting", "go", "going", "fly", "flying", "book", "booking",
        "search", "find", "plan", "planning", "want", "need",
    }
    for pattern in patterns:
        match = re.search(pattern, value)
        if not match:
            continue
        origin = match.group("origin").strip().split()
        destination = match.group("destination").strip().split()
        if not origin or not destination:
            continue
        if len(origin) > 4 or any(word in non_place_words for word in origin):
            continue
        if destination[0] in non_destination_verbs:
            continue
        return True
    return False


def explicit_specialists(text: str) -> list[str]:
    """Identify clearly requested services while respecting service negation.

    Avoid broad ``want ... flight`` matching: it incorrectly classified
    ``I don't want to miss my flight`` as a request to search flights.
    """
    value = _normalize(text)
    flight_topic = _SERVICE_TOPICS["flight"]
    hotel_topic = _SERVICE_TOPICS["hotel"]
    adjective = r"(?:(?:some|the|any|a|an|available|cheap(?:est)?|direct|nonstop|budget)\s+)*"

    # Positive search/action language must be close to the service noun.
    flight_action = (
        rf"\b(?:search(?:\s+for)?|find|look\s+for|compare|check|show|give|get|book|reserve|recommend|suggest)\s+(?:me\s+)?{adjective}{flight_topic}\b"
        rf"|\b(?:i\s+)?(?:want|need)\s+(?:(?:to\s+)?(?:search(?:\s+for)?|find|book|compare|look\s+for|reserve)\s+)?{adjective}{flight_topic}\b"
        rf"|\bhelp\s+(?:me\s+)?(?:to\s+)?(?:search(?:\s+for)?|find|compare)\s+(?:me\s+)?{adjective}{flight_topic}\b"
    )
    hotel_action = (
        rf"\b(?:search(?:\s+for)?|find|look\s+for|compare|check|show|give|get|book|reserve|recommend|suggest)\s+(?:me\s+)?{adjective}{hotel_topic}\b"
        rf"|\b(?:i\s+)?(?:want|need)\s+(?:(?:to\s+)?(?:search(?:\s+for)?|find|book|compare|look\s+for|reserve)\s+)?{adjective}{hotel_topic}\b"
        rf"|\bhelp\s+(?:me\s+)?(?:with|to\s+find|to\s+search(?:\s+for)?)\s+{adjective}{hotel_topic}\b"
    )

    wants_flight = bool(
        re.search(flight_action, value)
        or re.search(rf"\b{flight_topic}\s+(?:from|between|on|for|options?|prices?|fares?|search|details|schedule|availability)\b", value)
        or re.search(r"\bflights?\s+to\s+(?!be\b|cancelled\b|delayed\b|arrive\b|depart\b)[a-z]", value)
        or re.search(r"\b(?:cheapest|cheaper|lowest[- ]priced|best)\s+flights?\b", value)
        or re.search(r"\bwhich\s+flights?\b.{0,50}\b(?:choose|book|buy|select)\b", value)
        or re.search(r"\bwhat\s+about\b.{0,20}\bflights?\b", value)
        or re.search(r"\b(?:and|also)\s+flights?\b", value)
        or re.search(r"\b(?:fly|flying)\s+(?:from|to)\b", value)
        or re.search(r"\bby\s+(?:flight|air)\b", value)
        or re.search(rf"\b(?:just|only)\s+(?:(?:a|an|the)\s+)?{flight_topic}\b", value)
        or value in {"flight", "flights", "flight search", "find flights", "search flights"}
        or (
            _is_likely_air_route(value)
            and re.search(r"\b(today|tomorrow|tonight|weekend|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2}|date|dates|alone|adult|passenger)\b", value)
        )
    )

    wants_hotel = bool(
        re.search(hotel_action, value)
        or re.search(rf"\b{hotel_topic}\s+(?:in|near|at|for|options?|prices?|availability)\b", value)
        or re.search(r"\bwhat\s+about\b.{0,20}\b(?:hotels?|accommodations?)\b", value)
        or re.search(r"\b(?:and|also)\s+(?:hotels?|accommodations?)\b", value)
        or re.search(r"\bwhere\s+(?:can|should)\s+i\s+stay\b", value)
        or re.search(r"\b(?:a\s+)?place\s+to\s+stay\b", value)
        or re.search(r"\bstay\s+(?:in|at)\s+[a-z]", value)
        or re.search(r"\bstaying\s+(?:in|at)\s+[a-z]", value)
        # Treat the colloquial/incorrect "stay on <city> on Friday" as a
        # hotel request, but do not classify phrases like "stay on schedule".
        or (
            re.search(r"\bstay(?:ing)?\s+on\s+(?P<place>[a-z][a-z'-]*)\b", value)
            and re.search(r"\b(today|tomorrow|tonight|this\s+week|next\s+week|coming|monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})\b", value)
            and not re.search(r"\bstay(?:ing)?\s+on\s+(?:schedule|track|budget|course|task|topic|line|top)\b", value)
        )
        or re.search(rf"\b(?:just|only)\s+(?:(?:a|an|the)\s+)?{hotel_topic}\b", value)
        or re.search(r"\b(?:hotel|hotels|accommodation)\s+please\b", value)
        # Hotel-property requirements (e.g. 5-star, sea-view, hot tub/Jacuzzi)
        # are explicit hotel intent even when the user does not say "search".
        or re.search(r"\b(?:5[- ]?stars?|five[- ]stars?)\s+hotels?\b", value)
        or re.search(r"\bhotels?\b.{0,100}\b(?:sea[- ]view|sea[- ]facing|ocean[- ]view|hot\s+tubs?|jacuzzis?|whirlpools?|spa|pool)\b", value)
        or re.search(r"\b(?:sea[- ]view|sea[- ]facing|ocean[- ]view|hot\s+tubs?|jacuzzis?|whirlpools?)\b.{0,60}\b(?:hotels?|resorts?|properties|places\s+to\s+stay)\b", value)
        or value in {"hotel", "hotels", "accommodation", "hotel search", "find hotels", "search hotels"}
    )

    excluded = set(negated_services(value))
    if "flight" in excluded:
        wants_flight = False
    if "hotel" in excluded:
        wants_hotel = False

    if value in {"both", "both flights and hotels", "flights and hotels", "hotels and flights"} and not excluded:
        return ["flight_agent", "hotel_agent"]

    return (["flight_agent"] if wants_flight else []) + (["hotel_agent"] if wants_hotel else [])
