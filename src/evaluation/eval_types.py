from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


@dataclass
class SamplerResponse:
    """
    Response from a sampler.
    """

    response_text: str
    actual_queried_message_list: MessageList
    finish_reason: str | None


@dataclass
class EvalResult:
    """
    Result of running an evaluation (usually consisting of many samples)
    """

    score: float | None  # top-line metric
    metrics: dict[str, float] | None  # other metrics
    convos: list[MessageList]  # sampled conversations
    metadata: dict[str, Any] | None  # Extra data such as rubric scores or sollen
    correct_answers: list[str] | None = None
    extracted_answers: list[str] | None = None
    finish_reasons: list[str] | None = None


@dataclass
class SingleEvalResult:
    """
    Result of evaluating a single sample
    """

    score: float | None
    metrics: dict[str, float] = field(default_factory=dict)
    convo: MessageList | None = None  # sampled conversation
    correct_answer: str | None = None
    extracted_answer: str | None = None
    example_level_metadata: dict[str, Any] | None = (
        None  # Extra data such as rubric scores or sollen
    )
    finish_reason: str | None = None
