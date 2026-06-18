from contextlib import nullcontext

from configs.prompt_mode import PromptMode
from pipeline.generators.gemma3 import Gemma3Generator
from pipeline.run_baseline_norag import build_norag_prompt


class _FakeInputIds:
    shape = (1, 3)


class _FakeInputs(dict):
    def __init__(self):
        super().__init__({"input_ids": _FakeInputIds()})

    def to(self, _device):
        return self


class _FakeTokenizer:
    def __init__(self):
        self.last_prompt = None
        self.eos_token_id = 0

    def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
        self.last_prompt = prompt
        return _FakeInputs()

    def decode(self, _tokens, skip_special_tokens=True):
        return "decoded-answer"


class _FakeModel:
    device = "cpu"

    def generate(self, **kwargs):
        return [[11, 12, 13, 14]]


class _FakeTorch:
    @staticmethod
    def no_grad():
        return nullcontext()


def _make_generator(prompt_mode: PromptMode) -> tuple[Gemma3Generator, _FakeTokenizer]:
    tokenizer = _FakeTokenizer()
    generator = Gemma3Generator.__new__(Gemma3Generator)
    generator.prompt_mode = prompt_mode
    generator.tokenizer = tokenizer
    generator.processor = None
    generator.model = _FakeModel()
    generator.use_vision = False
    generator.device = "cpu"
    generator.decoding_params = {"temperature": 0.0, "max_new_tokens": 32, "do_sample": False}
    generator._torch = _FakeTorch()
    return generator, tokenizer


def test_gemma3_generate_uses_balanced_prompt_and_preserves_schema():
    generator, tokenizer = _make_generator(PromptMode.BALANCED)

    answer = generator.generate(
        query="Provide a ranked differential diagnosis.\n\nClinical Context: Pancytopenia and splenomegaly.",
        contexts=[{"doc_id": "PMC1", "text": "Example retrieved case text."}],
        query_images=[],
        context_images=[],
        use_rag_prompt=True,
    )

    prompt = tokenizer.last_prompt
    assert answer == "decoded-answer"
    assert "## INSTRUCTIONS (BALANCED RAG)" in prompt
    assert "REFERENCE CASES FROM LITERATURE" in prompt
    assert "Chosen Final Diagnosis for Scoring" in prompt
    assert "Evidence Source" in prompt
    assert "## SUPPORTING EVIDENCE" in prompt
    assert "## DIFFERENTIAL CONSIDERATIONS" in prompt
    assert "## EVIDENCE PRIORITY INSTRUCTION (AUGMENTATION MODE)" not in prompt


def test_gemma3_norag_prompt_contract_and_passthrough():
    generator, tokenizer = _make_generator(PromptMode.NO_CONTEXT)
    no_rag_prompt = build_norag_prompt(
        query_text="Provide a ranked differential diagnosis.\n\nClinical Context: Pancytopenia and splenomegaly."
    )

    answer = generator.generate(
        query=no_rag_prompt,
        contexts=[],
        query_images=[],
        context_images=[],
        use_rag_prompt=False,
    )

    prompt = tokenizer.last_prompt
    assert answer == "decoded-answer"
    assert prompt == no_rag_prompt
    assert "## INSTRUCTIONS (NO-RAG CONTROL)" in prompt
    assert "REFERENCE CASES FROM LITERATURE" not in prompt
    assert "Chosen Final Diagnosis for Scoring" in prompt
    assert "Evidence Source" in prompt
    assert "## SUPPORTING EVIDENCE" in prompt
    assert "## DIFFERENTIAL CONSIDERATIONS" in prompt
