"""Offline tests for the experimental evidence gate and unchanged default path."""

import pytest

from app.chatbot.evidence import (
    BINARY_POLICY,
    BinaryEvidenceGate,
    EvidenceOutputError,
    build_binary_evidence_messages,
    parse_binary_decision,
    parse_evidence_gate_enabled,
    parse_zero_one_setting,
)
from app.chatbot.exceptions import GenerationError
from app.chatbot.models import Chunk, RetrievedChunk
from app.chatbot.service import MedicalAssistance, NO_CONTEXT_RESPONSE


class FakeRetriever:
    def __init__(self, result=None):
        self.queries = []
        self.result = result if result is not None else [
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id="cbc_000", document_id="cbc", title="CBC",
                    source="Test", source_url="https://example.org/cbc",
                    section="Overview", content="A CBC measures blood cells.",
                ),
                score=0.8,
            )
        ]

    def retrieve(self, query):
        self.queries.append(query)
        return self.result


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


@pytest.mark.parametrize("text", ['{"status":"SUFFICIENT"}', '{"status":"INSUFFICIENT"}',
                                     '```json\n{"status":"SUFFICIENT"}\n```'])
def test_parser_accepts_expected_format(text):
    assert parse_binary_decision(text) in {"SUFFICIENT", "INSUFFICIENT"}


@pytest.mark.parametrize("text", [
    '{"status":"PARTIAL"}',
    '{"status":"SUFFICIENT","status":"INSUFFICIENT"}',
    '{"status":"SUFFICIENT","reason":"okay"}',
    'I think {"status":"SUFFICIENT"}',
    '[{"status":"SUFFICIENT"}]',
    '{"status":"SUFFICIENT"',
    '',
    'null',
])
def test_parser_rejects_invalid_or_extra_information(text):
    with pytest.raises(EvidenceOutputError):
        parse_binary_decision(text)


def test_prompt_matches_frozen_pilot_text_and_does_not_leak_scores():
    messages = build_binary_evidence_messages("¿Qué mide un hemograma?", FakeRetriever().result)
    assert len(messages) == 2
    assert messages[0] == {"role": "system", "content": BINARY_POLICY}
    assert messages[1]["content"] == (
        "<question>\n¿Qué mide un hemograma?\n</question>\n\n"
        "<passages>\n[Chunk ID: cbc_000]\nTitle: CBC\nSection: Overview\n"
        "Content: A CBC measures blood cells.\n</passages>"
    )
    assert "https://example.org" not in str(messages)
    assert "0.8" not in str(messages)


def test_default_mode_is_unchanged_and_does_not_call_evaluator():
    llm = FakeLLM(["ANSWER"])
    service = MedicalAssistance(llm, FakeRetriever())
    output = service.generate_response([{"role": "user", "content": "question"}])
    assert output.response == "ANSWER"
    assert len(output.sources) == 1
    assert len(llm.calls) == 1
    assert llm.calls[0][0]["role"] == "system"


def test_sufficient_calls_gate_then_generator_using_one_llm():
    llm = FakeLLM(['{"status":"SUFFICIENT"}', "SUPPORTED ANSWER"])
    gate = BinaryEvidenceGate(llm)
    service = MedicalAssistance(llm, FakeRetriever(), evidence_gate=gate)
    output = service.generate_response([{"role": "user", "content": "question"}])
    assert output.response == "SUPPORTED ANSWER"
    assert len(output.sources) == 1
    assert len(llm.calls) == 2
    assert llm.calls[0][0]["content"] == BINARY_POLICY
    assert "educational medical information assistant" in llm.calls[1][0]["content"]


def test_insufficient_abstains_without_generator_or_sources():
    llm = FakeLLM(['{"status":"INSUFFICIENT"}'])
    service = MedicalAssistance(llm, FakeRetriever(), evidence_gate=BinaryEvidenceGate(llm))
    output = service.generate_response([{"role": "user", "content": "question"}])
    assert output.response == NO_CONTEXT_RESPONSE
    assert output.sources == []
    assert len(llm.calls) == 1


def test_invalid_evaluator_abstains_and_logs_warning(caplog):
    llm = FakeLLM(['{"status":"PARTIAL"}'])
    service = MedicalAssistance(llm, FakeRetriever(), evidence_gate=BinaryEvidenceGate(llm))
    with caplog.at_level("WARNING"):
        output = service.generate_response([{"role": "user", "content": "question"}])
    assert output.response == NO_CONTEXT_RESPONSE
    assert output.sources == []
    assert "Invalid evidence-gate output" in caplog.text
    assert len(llm.calls) == 1


def test_empty_retrieval_skips_gate_and_generator():
    llm = FakeLLM([])
    service = MedicalAssistance(
        llm, FakeRetriever(result=[]), evidence_gate=BinaryEvidenceGate(llm)
    )
    output = service.generate_response([{"role": "user", "content": "question"}])
    assert output.response == NO_CONTEXT_RESPONSE
    assert output.sources == []
    assert llm.calls == []


def test_evaluator_runtime_failure_is_service_error_not_missing_evidence():
    llm = FakeLLM([RuntimeError("device failure")])
    service = MedicalAssistance(llm, FakeRetriever(), evidence_gate=BinaryEvidenceGate(llm))
    with pytest.raises(GenerationError, match="Evidence assessment failed"):
        service.generate_response([{"role": "user", "content": "question"}])
    assert len(llm.calls) == 1


@pytest.mark.parametrize("text,value", [("0", False), ("1", True)])
def test_explicit_environment_toggle(text, value):
    assert parse_evidence_gate_enabled(text) is value


@pytest.mark.parametrize("text", ["true", "false", "yes", "", "2", "ON"])
def test_invalid_environment_value_fails_early(text):
    with pytest.raises(ValueError, match="MEDICAL_EVIDENCE_GATE_ENABLED"):
        parse_evidence_gate_enabled(text)


def test_sampling_setting_can_be_pinned_for_matched_ab_comparison():
    assert parse_zero_one_setting("0", "MEDICAL_LLM_DO_SAMPLE") is False
    assert parse_zero_one_setting("1", "MEDICAL_LLM_DO_SAMPLE") is True
    with pytest.raises(ValueError, match="MEDICAL_LLM_DO_SAMPLE"):
        parse_zero_one_setting("maybe", "MEDICAL_LLM_DO_SAMPLE")
