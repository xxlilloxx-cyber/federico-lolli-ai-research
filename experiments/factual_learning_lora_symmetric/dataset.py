"""Deterministic synthetic factual associations and linguistic forms."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import string


@dataclass(frozen=True)
class Fact:
    fact_id: str
    fact_set: str
    subject: str
    answer: str


@dataclass(frozen=True)
class FactualExample:
    fact_id: str
    fact_set: str
    prompt_split: str
    prompt: str
    answer: str

    @property
    def full_text(self) -> str:
        return self.prompt + self.answer


TRAIN_TEMPLATES = (
    "Record: the code assigned to {subject} is",
    "Fact: {subject} has identification code",
    "Question: What code belongs to {subject}? Answer:",
)
VALIDATION_TEMPLATES = (
    "The identification code for {subject} is",
    "Complete the fact: {subject}'s code is",
)
PARAPHRASE_TEMPLATES = (
    "Which code is associated with {subject}?",
    "Respond with the code belonging to {subject}:",
)


def _token(rng: random.Random, prefix: str, letters: int) -> str:
    return prefix + "".join(rng.choice(string.ascii_uppercase) for _ in range(letters))


def generate_facts(seed: int, facts_per_set: int,
                   fact_sets: tuple[str, ...] = ("A", "B")) -> list[Fact]:
    if facts_per_set < 1:
        raise ValueError("facts_per_set must be positive")
    rng = random.Random(seed)
    facts, subjects, answers = [], set(), set()
    if not fact_sets or len(set(fact_sets)) != len(fact_sets):
        raise ValueError("fact_sets must be non-empty and unique")
    for fact_set in fact_sets:
        for index in range(facts_per_set):
            subject, answer = _token(rng, "QF-", 7), _token(rng, " VX-", 6)
            while subject in subjects:
                subject = _token(rng, "QF-", 7)
            while answer in answers:
                answer = _token(rng, " VX-", 6)
            subjects.add(subject); answers.add(answer)
            facts.append(Fact(f"{fact_set}{index:03d}", fact_set, subject, answer))
    return facts


def examples_for(facts: list[Fact], prompt_split: str) -> list[FactualExample]:
    templates = {"train": TRAIN_TEMPLATES, "validation": VALIDATION_TEMPLATES,
                 "paraphrase": PARAPHRASE_TEMPLATES}.get(prompt_split)
    if templates is None:
        raise ValueError(f"unknown prompt split: {prompt_split}")
    return [FactualExample(f.fact_id, f.fact_set, prompt_split,
                           template.format(subject=f.subject), f.answer)
            for f in facts for template in templates]


def serializable_dataset(facts: list[Fact]) -> dict:
    return {"facts": [asdict(fact) for fact in facts],
            "templates": {"train": list(TRAIN_TEMPLATES),
                          "validation": list(VALIDATION_TEMPLATES),
                          "paraphrase": list(PARAPHRASE_TEMPLATES)}}
