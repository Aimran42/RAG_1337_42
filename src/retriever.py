import json
import math
import re
from collections import Counter
from pathlib import Path

from src.models import MinimalSource


class Retriever:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.chunks: list[dict[str, object]] = []
        self.documents: list[list[str]] = []
        self.frequencies: Counter[str] = Counter()
        self.average_length = 0.0
        self._load()

    def _load(self) -> None:
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            self.chunks = [row for row in data if isinstance(row, dict)]
        self.documents = [self._tokens(str(row.get("text", ""))) for row in self.chunks]
        self.frequencies = Counter(term for doc in self.documents for term in set(doc))
        self.average_length = sum(map(len, self.documents)) / max(1, len(self.documents))

    def search(self, query: str, k: int) -> list[MinimalSource]:
        if not query.strip() or k <= 0 or not self.chunks:
            return []
        terms = self._tokens(query)
        ranked = []
        for index, doc in enumerate(self.documents):
            counts = Counter(doc)
            score = 0.0
            for term in terms:
                if counts[term]:
                    inverse = math.log(1 + (len(self.documents) - self.frequencies[term] + 0.5)
                                       / (self.frequencies[term] + 0.5))
                    denom = counts[term] + 1.5 * (
                        0.25 + 0.75 * len(doc) / max(1, self.average_length))
                    score += inverse * counts[term] * 2.5 / denom
            if score > 0:
                ranked.append((score, index))
        ranked.sort(reverse=True)
        results = []
        for _, index in ranked[:k]:
            row = self.chunks[index]
            results.append(MinimalSource(
                file_path=str(row["file_path"]),
                first_character_index=int(row["first_character_index"]),
                last_character_index=int(row["last_character_index"]),
            ))
        return results

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+", text.lower())
