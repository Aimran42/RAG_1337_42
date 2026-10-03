import re


class Chunker:
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max(1, min(max_chunk_size, 2000))

    def chunk(self, text: str, suffix: str) -> list[tuple[int, int, str]]:
        if not text:
            return []
        if suffix == ".py":
            return self._python(text)
        return self._text(text)

    def _python(self, text: str) -> list[tuple[int, int, str]]:
        starts = [0]
        starts.extend(match.start() for match in re.finditer(r"(?m)^(?=class |def |async def )", text))
        starts = sorted(set(starts))
        ranges = [(start, starts[i + 1] if i + 1 < len(starts) else len(text))
                  for i, start in enumerate(starts)]
        return self._split_ranges(text, ranges)

    def _text(self, text: str) -> list[tuple[int, int, str]]:
        ranges = []
        cursor = 0
        for match in re.finditer(r"\n\s*\n", text):
            if match.end() > cursor:
                ranges.append((cursor, match.end()))
                cursor = match.end()
        if cursor < len(text):
            ranges.append((cursor, len(text)))
        return self._split_ranges(text, ranges)

    def _split_ranges(self, text: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
        result = []
        for start, end in ranges:
            while end - start > self.max_chunk_size:
                limit = start + self.max_chunk_size
                split = text.rfind("\n", start, limit)
                if split <= start:
                    split = limit
                else:
                    split += 1
                result.append((start, split, text[start:split]))
                start = split
            if end > start and text[start:end].strip():
                result.append((start, end, text[start:end]))
        return result
