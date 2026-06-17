"""
MMLU Dataset Loader
===================
Loads MMLU test questions. Two modes:

1. HuggingFace datasets (preferred): pip install datasets
   Loads the official cais/mmlu test split.

2. CSV fallback: if you have MMLU CSVs locally, point --data_dir at them.

Each question is normalized to:
    {id, subject, question, choices: [A,B,C,D], answer: 'A'|'B'|'C'|'D'}
"""

from __future__ import annotations
import os
import csv
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MMLUQuestion:
    id:       str
    subject:  str
    question: str
    choices:  List[str]   # length 4
    answer:   str         # 'A', 'B', 'C', or 'D'


_LETTERS = ["A", "B", "C", "D"]


def load_from_huggingface(n: int = 100, subjects: Optional[List[str]] = None,
                          seed: int = 42) -> List[MMLUQuestion]:
    """Load n questions from HuggingFace cais/mmlu (stratified across subjects)."""
    from datasets import load_dataset
    import random

    # 'all' config concatenates all 57 subjects
    ds = load_dataset("cais/mmlu", "all", split="test")

    rows = list(ds)
    if subjects:
        rows = [r for r in rows if r["subject"] in subjects]

    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    questions = []
    for i, r in enumerate(rows):
        # HF schema: question, choices (list of 4), answer (int 0-3), subject
        ans_idx = r["answer"]
        questions.append(MMLUQuestion(
            id=f"mmlu_{i:04d}",
            subject=r["subject"],
            question=r["question"],
            choices=r["choices"],
            answer=_LETTERS[ans_idx],
        ))
    return questions


def load_from_csv(data_dir: str, n: int = 100, seed: int = 42) -> List[MMLUQuestion]:
    """
    Load from MMLU CSV files (the original format from the MMLU repo).
    Each CSV row: question, A, B, C, D, answer_letter
    Files named like 'abstract_algebra_test.csv'.
    """
    import random, glob

    questions = []
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*_test.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No *_test.csv files found in {data_dir}. "
            "Download MMLU from https://people.eecs.berkeley.edu/~hendrycks/data.tar"
        )

    for path in csv_files:
        subject = os.path.basename(path).replace("_test.csv", "")
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for j, row in enumerate(reader):
                if len(row) < 6:
                    continue
                question, a, b, c, d, ans = row[0], row[1], row[2], row[3], row[4], row[5]
                questions.append(MMLUQuestion(
                    id=f"{subject}_{j:04d}",
                    subject=subject,
                    question=question,
                    choices=[a, b, c, d],
                    answer=ans.strip().upper(),
                ))

    rng = random.Random(seed)
    rng.shuffle(questions)
    return questions[:n]


def load_mmlu(n: int = 100, data_dir: Optional[str] = None,
              subjects: Optional[List[str]] = None, seed: int = 42) -> List[MMLUQuestion]:
    """Try HuggingFace first; fall back to CSV if data_dir provided."""
    if data_dir:
        return load_from_csv(data_dir, n=n, seed=seed)
    try:
        return load_from_huggingface(n=n, subjects=subjects, seed=seed)
    except ImportError:
        raise ImportError(
            "Install HuggingFace datasets (`pip install datasets`) "
            "OR pass --data_dir pointing at MMLU CSV files."
        )


def format_prompt(q: MMLUQuestion, dry_run: bool = False) -> str:
    """Format an MMLU question as a zero-shot multiple-choice prompt."""
    lines = [q.question, ""]
    for letter, choice in zip(_LETTERS, q.choices):
        lines.append(f"{letter}. {choice}")
    lines.append("")
    lines.append("Answer with only the single letter (A, B, C, or D) of the correct choice.")
    prompt = "\n".join(lines)
    if dry_run:
        # embed the answer so MockAdapter can simulate realistic accuracy
        prompt += f"\n[CORRECT:{q.answer}]"
    return prompt


SYSTEM_PROMPT = (
    "You are an expert exam-taker. Read each multiple-choice question carefully "
    "and respond with ONLY the single letter (A, B, C, or D) of the correct answer. "
    "Do not explain."
)
