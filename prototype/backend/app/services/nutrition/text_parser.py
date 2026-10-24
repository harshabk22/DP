"""Natural language food logging parser for text and voice transcripts."""

import re
from typing import Optional
from pydantic import BaseModel


class ParsedTextItem(BaseModel):
    dish_label: str
    unit: str = "standard_bowl"
    unit_count: float = 1.0


# Number word mapping
WORD_TO_NUM: dict[str, float] = {
    "half": 0.5,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "a": 1.0,
    "an": 1.0,
}

# Unit alias mapping
UNIT_MAP: dict[str, str] = {
    "roti": "slice_piece",
    "rotis": "slice_piece",
    "chapati": "slice_piece",
    "chapatis": "slice_piece",
    "slice": "slice_piece",
    "slices": "slice_piece",
    "piece": "slice_piece",
    "pieces": "slice_piece",
    "biscuit": "slice_piece",
    "biscuits": "slice_piece",
    "egg": "slice_piece",
    "eggs": "slice_piece",
    "banana": "slice_piece",
    "apple": "slice_piece",
    "katori": "small_bowl",
    "katoris": "small_bowl",
    "bowl": "standard_bowl",
    "bowls": "standard_bowl",
    "cup": "cup_mug",
    "cups": "cup_mug",
    "glass": "cup_mug",
    "glasses": "cup_mug",
    "plate": "dinner_plate",
    "plates": "dinner_plate",
    "spoon": "tablespoon",
    "tablespoon": "tablespoon",
    "tbsp": "tablespoon",
}


class TextMealParser:
    """Parses natural language food descriptions into structured meal items."""

    @classmethod
    def parse(cls, text: str) -> list[ParsedTextItem]:
        """Parse natural language meal text into candidate items."""
        cleaned = text.lower().strip()
        # Remove common conversational prefixes
        cleaned = re.sub(
            r"^(i\s+had|had|i\s+ate|ate|eating|logged|log|having|drank|i\s+drank)\s+",
            "",
            cleaned,
        )

        # Split clauses by delimiters
        chunks = re.split(r",|\sand\s|\swith\s|\splas\s|\+\s*", cleaned)
        items: list[ParsedTextItem] = []

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            item = cls._parse_single_chunk(chunk)
            if item:
                items.append(item)

        if not items:
            # Fallback: treat entire string as single item
            items.append(ParsedTextItem(dish_label=cleaned, unit="standard_bowl", unit_count=1.0))

        return items

    @classmethod
    def _parse_single_chunk(cls, chunk: str) -> Optional[ParsedTextItem]:
        words = chunk.split()
        if not words:
            return None

        unit_count = 1.0
        unit = "standard_bowl"
        label_words: list[str] = []

        i = 0
        while i < len(words):
            word = words[i]

            # Check if numeric
            if re.match(r"^\d+(\.\d+)?$", word):
                unit_count = float(word)
                i += 1
                continue

            # Check word numbers (e.g. "two", "half", "one")
            if word in WORD_TO_NUM and i == 0:
                unit_count = WORD_TO_NUM[word]
                i += 1
                continue

            # Check unit keywords
            if word in UNIT_MAP:
                unit = UNIT_MAP[word]
                # If the unit is also the dish (e.g. "2 rotis", "1 apple", "2 eggs")
                if word in ["roti", "rotis", "chapati", "chapatis", "egg", "eggs", "apple", "banana", "biscuit", "biscuits"]:
                    # Keep word as dish label too
                    label_words.append(word)
                i += 1
                continue

            label_words.append(word)
            i += 1

        dish_label = " ".join(label_words).strip()
        if not dish_label:
            dish_label = chunk

        # Singularize common items for better database matching
        if dish_label == "rotis":
            dish_label = "roti"
        elif dish_label == "eggs":
            dish_label = "egg"
        elif dish_label == "biscuits":
            dish_label = "biscuit"

        return ParsedTextItem(
            dish_label=dish_label,
            unit=unit,
            unit_count=unit_count,
        )
