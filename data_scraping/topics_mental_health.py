"""Bulgarian keyword bank for mental-health passage retrieval.

Each topic maps to a list of lower-case Bulgarian keywords/phrases. A
passage is considered "matched" for a topic if at least one of its
keywords appears as a substring (case-insensitive, NFC-normalized).

Notes on scope:

* We focus on *peer-support / self-help* phrasing — emotions, daily
  coping, communication, habits, growth. We deliberately AVOID
  clinical / medication / diagnosis keywords; the resulting dataset
  is meant for a peer-support model, not a doctor.
* Keywords are intentionally permissive (root forms and common
  derivations). The dataset-build step filters again on length/quality.
* When a passage matches multiple topics, the highest-priority topic
  wins (see :data:`TOPIC_PRIORITY`).
"""

# Bulgarian keywords per topic. Lowercase, NFC-normalized.
# Add new topics by appending here — the pipeline picks them up automatically.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    # ---- Emotional health ------------------------------------------------
    "anxiety": (
        "тревожност", "тревога", "тревож", "безпокойство", "безпокоен",
        "паник", "паническа атака", "страх от", "напрежение",
    ),
    "depression": (
        "депресия", "депресивен", "тъга", "тъжен", "тъжно", "апатия",
        "безнадежд", "отчаяние", "загуба на смисъл", "празнота",
    ),
    "stress": (
        "стрес", "пренапрежение", "претоварване", "изтощение",
        "изгаряне", "бърнаут", "напрегнат",
    ),
    "anger": (
        "гняв", "раздразнение", "ярост", "недоволство", "агресия",
        "раздразнителен",
    ),
    "grief": (
        "загуба", "скръб", "оплакване", "болка от загубата", "траур",
        "тъга по загубен",
    ),

    # ---- Self ------------------------------------------------------------
    "self_esteem": (
        "самочувствие", "самооценка", "увереност", "самоувереност",
        "несигурност", "съмнение в себе си",
    ),
    "self_compassion": (
        "съчувствие към себе си", "приемане на себе си", "самосъстрадание",
        "грижа за себе си", "обич към себе си",
    ),
    "self_awareness": (
        "осъзнатост", "самопознание", "себепознание", "вътрешен глас",
        "вътрешен мир", "интроспекция", "рефлекс",
    ),
    "growth": (
        "личностно развитие", "личностен растеж", "себеусъвършенстване",
        "самоусъвършенстване", "промяна", "развитие на личността",
    ),

    # ---- Relations -------------------------------------------------------
    "relationships": (
        "взаимоотношения", "отношения с", "партньор", "приятел",
        "семейство", "родител", "близост", "доверие",
    ),
    "boundaries": (
        "лични граници", "граници в отношенията", "поставяне на граници",
        "лични граници", "да кажеш не",
    ),
    "communication": (
        "комуникация", "общуване", "диалог", "слушане", "изразяване",
        "конфликт", "разрешаване на конфликт",
    ),
    "loneliness": (
        "самота", "усещане за самота", "изолация", "социална изолация",
        "чувствам се сам",
    ),
    "parenting": (
        "родителство", "възпитание", "отглеждане", "родителска роля",
        "деца", "тийнейджъри",
    ),

    # ---- Coping & lifestyle ---------------------------------------------
    "mindfulness": (
        "осъзнато присъствие", "медитация", "осъзнатост", "присъствие в мига",
        "дишане", "дихателни упражнения",
    ),
    "resilience": (
        "устойчивост", "справяне", "справям се", "възстановяване",
        "психическа устойчивост", "вътрешна сила",
    ),
    "habits": (
        "навици", "ежедневие", "рутина", "малки промени", "формиране на навик",
    ),
    "sleep": (
        "сън", "безсъние", "лош сън", "качество на съня", "хигиена на съня",
    ),
    "addiction": (
        "зависимост", "пристрастяване", "натрапчиво поведение",
        "алкохол", "наркотици", "хазарт",
    ),
    "motivation": (
        "мотивация", "цел", "стремеж", "вдъхновение", "воля",
        "поставяне на цели",
    ),
    "happiness": (
        "щастие", "удовлетворение", "благополучие", "положителни емоции",
        "благодарност",
    ),
    "fear": (
        "страх", "страхове", "фобия", "тревожен", "несигурност",
    ),
    "trauma": (
        "травма", "травматичн", "ПТСР", "детска травма", "загуба от",
    ),
}

# When a passage hits multiple topics, the earlier topic in this list wins.
# Ordered roughly by "specific symptom → general theme".
TOPIC_PRIORITY: tuple[str, ...] = (
    "trauma", "addiction", "depression", "anxiety", "anger",
    "grief", "fear", "stress", "sleep",
    "loneliness", "boundaries", "communication", "relationships",
    "parenting",
    "self_esteem", "self_compassion", "self_awareness",
    "mindfulness", "resilience", "habits", "motivation",
    "happiness", "growth",
)


def all_topics() -> tuple[str, ...]:
    """Stable, priority-ordered tuple of every topic name."""
    seen: set[str] = set()
    order = []
    for t in TOPIC_PRIORITY:
        if t in TOPIC_KEYWORDS and t not in seen:
            order.append(t); seen.add(t)
    for t in TOPIC_KEYWORDS:
        if t not in seen:
            order.append(t); seen.add(t)
    return tuple(order)


def match_passage(text: str, topics: tuple[str, ...] | None = None) -> tuple[str, tuple[str, ...]] | None:
    """Return ``(top_topic, matched_keywords)`` for a passage, or None.

    Performs a case-insensitive substring scan. ``topics`` filters to a
    subset (priority order is preserved); pass None to scan everything.
    """
    if not text:
        return None
    lowered = text.lower()
    candidates = topics if topics else all_topics()
    hits: dict[str, list[str]] = {}
    for topic in candidates:
        for keyword in TOPIC_KEYWORDS.get(topic, ()):
            if keyword in lowered:
                hits.setdefault(topic, []).append(keyword)
    if not hits:
        return None
    # Pick the highest-priority topic among the hits.
    priority_index = {t: i for i, t in enumerate(all_topics())}
    top = min(hits, key=lambda t: priority_index.get(t, 10_000))
    return top, tuple(hits[top])


__all__ = ["TOPIC_KEYWORDS", "TOPIC_PRIORITY", "all_topics", "match_passage"]
