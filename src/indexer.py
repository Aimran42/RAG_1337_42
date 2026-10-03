import json
from pathlib import Path

from tqdm import tqdm

from src.chunker import Chunker


class Indexer:
    extensions = {".py", ".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml"}

    def __init__(self, root: Path, output: Path, max_chunk_size: int = 2000) -> None:
        self.root = root
        self.output = output
        self.chunker = Chunker(max_chunk_size)

    def build(self) -> int:
        files = [path for path in self.root.rglob("*")
                 if path.is_file() and path.suffix.lower() in self.extensions]
        chunks = []
        for path in tqdm(files, desc="Indexing"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                relative = path.as_posix()
                for start, end, content in self.chunker.chunk(text, path.suffix.lower()):
                    chunks.append({"file_path": relative, "first_character_index": start,
                                   "last_character_index": end, "text": content})
            except OSError:
                continue
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "index.json").write_text(
            json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        return len(chunks)
