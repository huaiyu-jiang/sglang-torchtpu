"""Request features TpuSpecWorker cannot serve, decided per request at admission.

Mirrors TpuSpecWorker._check_supported, which sees the same features on a whole
batch and can only fail it: a request rejected here never reaches that check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.sampling.custom_logit_processor import UOCR_PAGE_GUARD_PROCESSOR
from sglang.srt.sampling.sampling_params import TOP_K_ALL

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req


def unsupported_request_features(
    req: Req,
    *,
    supports_grammar: bool,
    supports_logprobs: bool,
    supports_additive_penalties: bool,
    bsearch_filtering: bool,
    custom_logit_processor_enabled: bool,
) -> list[str]:
    """Names of the request's features the worker cannot serve; empty when it can."""
    params = req.sampling_params
    features = []
    if not supports_grammar and (
        params.json_schema is not None
        or params.regex is not None
        or params.ebnf is not None
        or params.structural_tag is not None
    ):
        features.append("structured output (json_schema, regex, ebnf, structural_tag)")
    if req.return_logprob and not supports_logprobs:
        features.append("logprobs")
    if params.logit_bias is not None:
        features.append("logit_bias")
    # Scaling penalties have no snapshot path; additive ones only with the opt-in.
    if params.repetition_penalty != 1.0:
        features.append("repetition_penalty")
    if not supports_additive_penalties:
        if params.frequency_penalty != 0.0:
            features.append("frequency_penalty")
        if params.presence_penalty != 0.0:
            features.append("presence_penalty")
        if params.min_new_tokens > 0:
            features.append("min_new_tokens")
    if req.custom_logit_processor and (
        custom_logit_processor_enabled
        or req.custom_logit_processor == UOCR_PAGE_GUARD_PROCESSOR
    ):
        features.append("custom_logit_processor")
    # A sampled row (top_k > 1) with a finite top_k or top_p < 1 needs the
    # sort-free filter; min_p is already rejected for every speculative worker.
    sampled = params.top_k > 1
    if (
        not bsearch_filtering
        and sampled
        and (params.top_k < TOP_K_ALL or params.top_p < 1.0)
    ):
        features.append(
            "top_k/top_p sampling without SGLANG_TPU_ENABLE_BSEARCH_SAMPLING=1"
        )
    return features
