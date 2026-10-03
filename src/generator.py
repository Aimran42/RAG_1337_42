from src.models import MinimalSource


class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    def answer(self, question: str, sources: list[MinimalSource], root: str = ".") -> str:
        context = self._context(sources, root)
        if not context:
            return "I could not find relevant source material to answer this question."
        try:
            return self._generate(question, context)
        except Exception as error:
            return f"Answer generation failed: {error}"

    def _context(self, sources: list[MinimalSource], root: str) -> str:
        from pathlib import Path

        parts = []
        for source in sources:
            try:
                content = Path(root, source.file_path).read_text(encoding="utf-8", errors="ignore")
                snippet = content[source.first_character_index:source.last_character_index]
                parts.append(f"[{source.file_path}]\n{snippet}")
            except OSError:
                continue
        return "\n\n".join(parts)

    def _generate(self, question: str, context: str) -> str:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.model is None or self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        prompt = ("Answer using only the sources below. If they do not contain the answer, "
                  "say so.\n\nSources:\n" + context + "\n\nQuestion: " + question + "\nAnswer:")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6000)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=300, do_sample=False)
        answer = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
        return answer.strip()
