"""The fact set used by the correction benchmark.

Each entry is:
    q    - the question asked to the host
    gold - the answer that counts as correct
    key  - the memory key; it MUST appear (as a substring, after
           normalisation) in the question, otherwise the reflex will not fire
           and the fact is a "cold" fact.

The categories are deliberately mundane: a small model gets some of them
right and some of them wrong, which is what a correction benchmark needs.
If every fact were easy the fix rate would be meaningless; if every fact were
impossible the harm rate would be meaningless.
"""
from __future__ import annotations

from typing import Dict, List

FACTS: List[Dict[str, str]] = [
    # --- capitals -------------------------------------------------------
    {"q": "What is the capital of France?", "gold": "Paris", "key": "capital of france"},
    {"q": "What is the capital of Italy?", "gold": "Rome", "key": "capital of italy"},
    {"q": "What is the capital of Spain?", "gold": "Madrid", "key": "capital of spain"},
    {"q": "What is the capital of Japan?", "gold": "Tokyo", "key": "capital of japan"},
    {"q": "What is the capital of Brazil?", "gold": "Brasilia", "key": "capital of brazil"},
    {"q": "What is the capital of Canada?", "gold": "Ottawa", "key": "capital of canada"},
    {"q": "What is the capital of Germany?", "gold": "Berlin", "key": "capital of germany"},
    {"q": "What is the capital of Egypt?", "gold": "Cairo", "key": "capital of egypt"},
    {"q": "What is the capital of Kenya?", "gold": "Nairobi", "key": "capital of kenya"},
    {"q": "What is the capital of Peru?", "gold": "Lima", "key": "capital of peru"},
    {"q": "What is the capital of Norway?", "gold": "Oslo", "key": "capital of norway"},
    {"q": "What is the capital of Greece?", "gold": "Athens", "key": "capital of greece"},
    {"q": "What is the capital of Poland?", "gold": "Warsaw", "key": "capital of poland"},
    {"q": "What is the capital of Portugal?", "gold": "Lisbon", "key": "capital of portugal"},
    {"q": "What is the capital of Austria?", "gold": "Vienna", "key": "capital of austria"},
    {"q": "What is the capital of Switzerland?", "gold": "Bern", "key": "capital of switzerland"},
    {"q": "What is the capital of Turkey?", "gold": "Ankara", "key": "capital of turkey"},
    {"q": "What is the capital of Argentina?", "gold": "Buenos Aires", "key": "capital of argentina"},
    {"q": "What is the capital of Nigeria?", "gold": "Abuja", "key": "capital of nigeria"},
    {"q": "What is the capital of Vietnam?", "gold": "Hanoi", "key": "capital of vietnam"},
    {"q": "What is the capital of Morocco?", "gold": "Rabat", "key": "capital of morocco"},
    {"q": "What is the capital of Sweden?", "gold": "Stockholm", "key": "capital of sweden"},
    {"q": "What is the capital of Belgium?", "gold": "Brussels", "key": "capital of belgium"},
    {"q": "What is the capital of Ireland?", "gold": "Dublin", "key": "capital of ireland"},
    {"q": "What is the capital of Thailand?", "gold": "Bangkok", "key": "capital of thailand"},
    {"q": "What is the capital of Indonesia?", "gold": "Jakarta", "key": "capital of indonesia"},
    {"q": "What is the capital of Switzerland?", "gold": "Bern", "key": "capital of switzerland"},
    {"q": "What is the capital of Denmark?", "gold": "Copenhagen", "key": "capital of denmark"},
    {"q": "What is the capital of Finland?", "gold": "Helsinki", "key": "capital of finland"},
    {"q": "What is the capital of Chile?", "gold": "Santiago", "key": "capital of chile"},
    {"q": "What is the capital of Colombia?", "gold": "Bogota", "key": "capital of colombia"},
    {"q": "What is the capital of Cuba?", "gold": "Havana", "key": "capital of cuba"},
    {"q": "What is the capital of Iceland?", "gold": "Reykjavik", "key": "capital of iceland"},
    {"q": "What is the capital of Croatia?", "gold": "Zagreb", "key": "capital of croatia"},
    {"q": "What is the capital of Romania?", "gold": "Bucharest", "key": "capital of romania"},
    {"q": "What is the capital of Hungary?", "gold": "Budapest", "key": "capital of hungary"},
    {"q": "What is the capital of Czechia?", "gold": "Prague", "key": "capital of czechia"},

    # --- chemistry ------------------------------------------------------
    {"q": "What is the chemical symbol for gold?", "gold": "Au", "key": "chemical symbol for gold"},
    {"q": "What is the chemical symbol for silver?", "gold": "Ag", "key": "chemical symbol for silver"},
    {"q": "What is the chemical symbol for iron?", "gold": "Fe", "key": "chemical symbol for iron"},
    {"q": "What is the chemical symbol for oxygen?", "gold": "O", "key": "chemical symbol for oxygen"},
    {"q": "What is the chemical symbol for carbon?", "gold": "C", "key": "chemical symbol for carbon"},
    {"q": "What is the chemical symbol for sodium?", "gold": "Na", "key": "chemical symbol for sodium"},
    {"q": "What is the chemical symbol for potassium?", "gold": "K", "key": "chemical symbol for potassium"},
    {"q": "What is the chemical symbol for copper?", "gold": "Cu", "key": "chemical symbol for copper"},
    {"q": "What is the chemical symbol for zinc?", "gold": "Zn", "key": "chemical symbol for zinc"},
    {"q": "What is the chemical symbol for nitrogen?", "gold": "N", "key": "chemical symbol for nitrogen"},
    {"q": "What is the chemical symbol for hydrogen?", "gold": "H", "key": "chemical symbol for hydrogen"},
    {"q": "What is the chemical symbol for calcium?", "gold": "Ca", "key": "chemical symbol for calcium"},
    {"q": "What is the chemical symbol for lead?", "gold": "Pb", "key": "chemical symbol for lead"},
    {"q": "What is the chemical symbol for mercury?", "gold": "Hg", "key": "chemical symbol for mercury"},
    {"q": "What is the chemical symbol for tin?", "gold": "Sn", "key": "chemical symbol for tin"},

    # --- literature -----------------------------------------------------
    {"q": "Who wrote the play Hamlet?", "gold": "Shakespeare", "key": "wrote the play hamlet"},
    {"q": "Who wrote the novel Moby Dick?", "gold": "Melville", "key": "wrote the novel moby dick"},
    {"q": "Who wrote Pride and Prejudice?", "gold": "Austen", "key": "wrote pride and prejudice"},
    {"q": "Who wrote the play Macbeth?", "gold": "Shakespeare", "key": "wrote the play macbeth"},
    {"q": "Who wrote the novel Frankenstein?", "gold": "Shelley", "key": "wrote the novel frankenstein"},
    {"q": "Who wrote The Great Gatsby?", "gold": "Fitzgerald", "key": "wrote the great gatsby"},
    {"q": "Who wrote the novel Dracula?", "gold": "Stoker", "key": "wrote the novel dracula"},
    {"q": "Who wrote the Odyssey?", "gold": "Homer", "key": "wrote the odyssey"},
    {"q": "Who wrote the play Romeo and Juliet?", "gold": "Shakespeare", "key": "wrote the play romeo and juliet"},
    {"q": "Who wrote One Hundred Years of Solitude?", "gold": "Garcia Marquez", "key": "wrote one hundred years of solitude"},
    {"q": "Who wrote the novel Jane Eyre?", "gold": "Charlotte Bronte", "key": "wrote the novel jane eyre"},
    {"q": "Who wrote the play A Doll's House?", "gold": "Ibsen", "key": "wrote the play a doll s house"},
    {"q": "Who wrote War and Peace?", "gold": "Tolstoy", "key": "wrote war and peace"},
    {"q": "Who wrote the novel Emma?", "gold": "Jane Austen", "key": "wrote the novel emma"},
    {"q": "Who wrote the play Hamlet in English?", "gold": "Shakespeare", "key": "wrote the play hamlet in english"},

    # --- science --------------------------------------------------------
    {"q": "What is the chemical formula for water?", "gold": "H2O", "key": "chemical formula for water"},
    {"q": "What is the largest ocean on Earth?", "gold": "Pacific", "key": "largest ocean on earth"},
    {"q": "What is the closest planet to the Sun?", "gold": "Mercury", "key": "closest planet to the sun"},
    {"q": "What is the largest planet in the Solar System?", "gold": "Jupiter", "key": "largest planet in the solar system"},
    {"q": "What gas do plants absorb during photosynthesis?", "gold": "carbon dioxide", "key": "gas do plants absorb during photosynthesis"},
    {"q": "What is the powerhouse of the cell?", "gold": "mitochondria", "key": "powerhouse of the cell"},
    {"q": "How many bones are in the adult human body?", "gold": "206", "key": "bones are in the adult human body"},
    {"q": "What is the hardest natural substance?", "gold": "diamond", "key": "hardest natural substance"},
    {"q": "What is the speed of light in a vacuum?", "gold": "300000", "key": "speed of light in a vacuum"},
    {"q": "What is the most abundant gas in Earth's atmosphere?", "gold": "nitrogen", "key": "most abundant gas in earth s atmosphere"},
    {"q": "How many continents are there on Earth?", "gold": "7", "key": "continents are there on earth"},
    {"q": "What is the smallest prime number?", "gold": "2", "key": "smallest prime number"},
    {"q": "What organ pumps blood through the body?", "gold": "heart", "key": "organ pumps blood through the body"},
    {"q": "What is the freezing point of water in Celsius?", "gold": "0", "key": "freezing point of water in celsius"},
    {"q": "What is the most common element in the universe?", "gold": "hydrogen", "key": "most common element in the universe"},

    # --- geography / history -------------------------------------------
    {"q": "What is the longest river in the world?", "gold": "Nile", "key": "longest river in the world"},
    {"q": "What is the largest desert in the world?", "gold": "Sahara", "key": "largest desert in the world"},
    {"q": "What is the highest mountain in the world?", "gold": "Everest", "key": "highest mountain in the world"},
    {"q": "What is the smallest country in the world?", "gold": "Vatican City", "key": "smallest country in the world"},
    {"q": "Who was the first person to walk on the Moon?", "gold": "Armstrong", "key": "first person to walk on the moon"},
    {"q": "In which year did the Second World War end?", "gold": "1945", "key": "second world war end"},
    {"q": "Who was the first president of the United States?", "gold": "Washington", "key": "first president of the united states"},
    {"q": "What is the currency of Japan?", "gold": "yen", "key": "currency of japan"},
    {"q": "What is the currency of Switzerland?", "gold": "franc", "key": "currency of switzerland"},
    {"q": "How many players are on the field per side in football?", "gold": "11", "key": "players are on the field per side in football"},
    {"q": "How many continents does Europe border?", "gold": "Africa and Asia", "key": "continents does europe border"},
    {"q": "What is the largest country in South America?", "gold": "Brazil", "key": "largest country in south america"},
    {"q": "Which planet is known as the Red Planet?", "gold": "Mars", "key": "planet is known as the red planet"},
    {"q": "What is the longest bone in the human body?", "gold": "femur", "key": "longest bone in the human body"},
    {"q": "How many teeth does an adult human have?", "gold": "32", "key": "teeth does an adult human have"},
]


def facts() -> List[Dict[str, str]]:
    """A copy of the fact set."""
    return [dict(f) for f in FACTS]


def facts_for_model(model: str) -> List[Dict[str, str]]:
    """Facts that are worth asking a given model.

    A 124M model has no chance on the literature or chemistry subsets; asking
    anyway produces a benchmark where the baseline is ~0% and every number is
    noise. The tiers keep the baseline in a measurable range.
    """
    if "124m" in model.lower() or "pythia-70m" in model.lower():
        return [f for f in FACTS if f["key"].startswith("capital of")]
    return [dict(f) for f in FACTS]
