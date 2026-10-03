import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.indexer import Indexer
from src.models import MinimalSearchResults, StudentSearchResultsAndAnswer
from src.pipeline import RagPipeline


class Commands:
    def index(self, max_chunk_size: int = 2000, data_path: str = "data/raw",
              output_path: str = "data/processed") -> str:
        try:
            count = Indexer(Path(data_path), Path(output_path), max_chunk_size).build()
            return f"Ingestion complete: {count} chunks saved under {output_path}"
        except (OSError, ValueError) as error:
            return f"Indexing failed: {error}"

    def search(self, query: str, k: int = 5,
               index_path: str = "data/processed/index.json") -> str:
        try:
            pipeline = RagPipeline(Path(index_path))
            result = pipeline.search("query", query, k)
            return json.dumps(result.model_dump(), indent=2)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return f"Search failed: {error}"

    def search_dataset(self, dataset_path: str, k: int = 10,
                       save_directory: str = "data/output/search_results",
                       index_path: str = "data/processed/index.json") -> str:
        try:
            rows = self._read_questions(Path(dataset_path))
            pipeline = RagPipeline(Path(index_path))
            results = []
            for row in tqdm(rows, desc="Searching"):
                results.append(pipeline.search(str(row.get("question_id", "")),
                                               str(row.get("question", "")), k))
            target = self._target(dataset_path, save_directory)
            target.write_text(json.dumps({"search_results": [r.model_dump() for r in results],
                                          "k": max(0, k)}, indent=2), encoding="utf-8")
            return f"Saved student_search_results to {target}"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return f"Dataset search failed: {error}"

    def answer(self, query: str, k: int = 5,
               index_path: str = "data/processed/index.json") -> str:
        try:
            pipeline = RagPipeline(Path(index_path))
            result = pipeline.search("query", query, k)
            return pipeline.generator.answer(query, result.retrieved_sources)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return f"Answer failed: {error}"

    def answer_dataset(self, student_search_results_path: str,
                       save_directory: str = "data/output/search_results_and_answer",
                       model_name: str = "Qwen/Qwen3-0.6B") -> str:
        try:
            data = json.loads(Path(student_search_results_path).read_text(encoding="utf-8"))
            pipeline = RagPipeline(model_name=model_name)
            answers = []
            for row in tqdm(data.get("search_results", []), desc="Generating answers"):
                result = MinimalSearchResults.model_validate(row)
                answers.append(pipeline.answer(result))
            output = StudentSearchResultsAndAnswer(search_results=answers,
                                                   k=int(data.get("k", 0)))
            target = self._target(student_search_results_path, save_directory)
            target.write_text(output.model_dump_json(indent=2), encoding="utf-8")
            return f"Saved student_search_results_and_answer to {target}"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return f"Answer dataset failed: {error}"

    def evaluate(self, student_search_results_path: str, dataset_path: str) -> str:
        try:
            student = json.loads(Path(student_search_results_path).read_text(encoding="utf-8"))
            ground = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
            references = {str(row.get("question_id")): row
                          for row in ground.get("rag_questions", [])}
            scores = []
            for row in student.get("search_results", []):
                reference = references.get(str(row.get("question_id")), {})
                expected = reference.get("sources", [])
                found = sum(any(self._overlap(source, candidate) >= 0.05
                                for candidate in row.get("retrieved_sources", []))
                            for source in expected)
                scores.append(found / len(expected) if expected else 1.0)
            mean = sum(scores) / len(scores) if scores else 0.0
            return f"Recall@k: {mean:.3f} ({len(scores)} questions)"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            return f"Evaluation failed: {error}"

    @staticmethod
    def _read_questions(path: Path) -> list[dict[str, Any]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("rag_questions", []) if isinstance(data, dict) else []
        return [row for row in rows if isinstance(row, dict)]

    @staticmethod
    def _target(source: str, directory: str) -> Path:
        target = Path(directory) / Path(source).name
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @staticmethod
    def _overlap(first: dict[str, Any], second: dict[str, Any]) -> float:
        if first.get("file_path") != second.get("file_path"):
            return 0.0
        start = max(int(first.get("first_character_index", 0)),
                    int(second.get("first_character_index", 0)))
        end = min(int(first.get("last_character_index", 0)),
                  int(second.get("last_character_index", 0)))
        intersection = max(0, end - start)
        union = max(int(first.get("last_character_index", 0)),
                    int(second.get("last_character_index", 0))) - min(
                        int(first.get("first_character_index", 0)),
                        int(second.get("first_character_index", 0)))
        return intersection / union if union > 0 else 0.0
