from pathlib import Path

from src.generator import AnswerGenerator
from src.models import MinimalSearchResults, StudentSearchResults
from src.retriever import Retriever


class RagPipeline:
    def __init__(self, index_path: Path = Path("data/processed/index.json"),
                 model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.retriever = Retriever(index_path)
        self.generator = AnswerGenerator(model_name)

    def search(self, question_id: str, question: str, k: int) -> MinimalSearchResults:
        sources = self.retriever.search(question, k)
        return MinimalSearchResults(question_id=question_id, question=question,
                                    retrieved_sources=sources)

    def search_dataset(self, questions: list[dict[str, object]], k: int) -> StudentSearchResults:
        results = [self.search(str(row.get("question_id", "")),
                               str(row.get("question", "")), k) for row in questions]
        return StudentSearchResults(search_results=results, k=max(0, k))

    def answer(self, result: MinimalSearchResults) -> dict[str, object]:
        return {**result.model_dump(), "answer": self.generator.answer(
            result.question, result.retrieved_sources)}
