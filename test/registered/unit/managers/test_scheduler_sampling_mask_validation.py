"""Requests the scheduler must turn away with a 400 before admission."""

import unittest
from array import array
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _handle_pd_request(mode, req):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = mode
    scheduler.disagg_metadata_buffers = SimpleNamespace(enable_sampling_mask=False)
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
    scheduler._record_kv_event_lora_name = MagicMock()
    scheduler.spec_algorithm = SimpleNamespace(
        is_dflash_family=lambda: False,
        is_uno=lambda: False,
    )
    scheduler._add_request_to_queue = MagicMock()
    scheduler.output_streamer = MagicMock()
    scheduler.tree_cache = MagicMock()
    scheduler.beam_coordinator = MagicMock()
    scheduler.hisparse_coordinator = None
    scheduler.uocr_runtime = None
    scheduler.send_abort_output = MagicMock()
    scheduler.init_request_retirer()
    recv_req = MagicMock(
        session_params=None,
        session_id=None,
        input_embeds=None,
        bootstrap_port=1,
        bootstrap_room=9,
    )
    req.exit_via = req.send_via = req.kv.retraction_backup = None
    req.kv.holds_kv = req.kv.holds_mamba = False
    with (
        patch(
            "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
            return_value=1,
        ),
        patch("sglang.srt.managers.scheduler.Req", return_value=req),
        patch("sglang.srt.managers.scheduler.prepare_abort") as abort,
    ):
        scheduler.handle_generate_request(recv_req)
    return abort, scheduler._add_request_to_queue


class _RejectingSpecWorker(BaseSpecWorker):
    reason = "repetition_penalty is not supported by this speculative worker"
    seen = []

    @classmethod
    def validate_request(cls, req):
        cls.seen.append(req)
        return cls.reason


def _handle_spec_request(spec_worker_cls, *, draft_worker):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.enable_session_radix_cache = False
    scheduler.model_config = SimpleNamespace(hf_eos_token_id={1}, vocab_size=128)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
    scheduler.tokenizer = None
    scheduler.dllm_config = None
    scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
    scheduler._record_kv_event_lora_name = MagicMock()
    scheduler.spec_algorithm = SpeculativeAlgorithm.EAGLE
    scheduler.spec_worker_cls = spec_worker_cls
    scheduler.draft_worker = draft_worker
    scheduler._add_request_to_queue = MagicMock()
    scheduler.output_streamer = MagicMock()
    scheduler.tree_cache = MagicMock()
    scheduler.beam_coordinator = MagicMock()
    scheduler.hisparse_coordinator = None
    scheduler.uocr_runtime = None
    scheduler.grammar_manager = MagicMock(
        process_req_with_grammar=MagicMock(return_value=False)
    )
    scheduler.invariant_checker = MagicMock()
    scheduler._bind_pp_arrival_prefetch = MagicMock()
    scheduler.send_abort_output = MagicMock()
    scheduler.init_req_max_new_tokens = MagicMock()
    scheduler.max_req_input_len = 64
    scheduler._embedding_preserve_last_token_id = None
    scheduler.init_request_retirer()
    recv_req = MagicMock(
        session_params=None,
        session_id=None,
        input_embeds=None,
        mm_inputs=None,
        return_logprob=False,
        logprob_start_len=-1,
        token_ids_logprob_rows=None,
        return_routed_experts=False,
    )
    req = MagicMock(
        return_sampling_mask=False,
        return_logprob=False,
        is_prefill_only=False,
        origin_input_ids=[2, 3, 4],
        sampling_params=SamplingParams(max_new_tokens=8, repetition_penalty=1.05),
    )
    req.exit_via = req.send_via = req.kv.retraction_backup = None
    req.kv.holds_kv = req.kv.holds_mamba = False
    with (
        patch(
            "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
            return_value=1,
        ),
        patch("sglang.srt.managers.scheduler.Req", return_value=req),
        patch("sglang.srt.managers.scheduler.prepare_abort") as abort,
        get_context().override_server_args(),
    ):
        scheduler.handle_generate_request(recv_req)
    return req, abort, scheduler._add_request_to_queue


class TestSchedulerSamplingMaskValidation(CustomTestCase):
    def test_disabled_pd_masks_return_bad_request_before_admission(self):
        """Neither PD role may queue mask requests without metadata buffers."""
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            with self.subTest(mode=mode):
                abort, add_to_queue = _handle_pd_request(
                    mode, MagicMock(return_sampling_mask=True, return_logprob=False)
                )
                abort.assert_called_once()
                self.assertEqual(
                    abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST
                )
                add_to_queue.assert_not_called()

    def test_pd_top_logprobs_beyond_capacity_return_bad_request_before_admission(
        self,
    ):
        capacity = 8
        for mode in (DisaggregationMode.PREFILL, DisaggregationMode.DECODE):
            for top_logprobs_num in (capacity, capacity + 1):
                with (
                    self.subTest(mode=mode, top_logprobs_num=top_logprobs_num),
                    get_context().override_server_args(
                        disaggregation_max_top_logprobs_num=capacity
                    ),
                ):
                    req = MagicMock(return_sampling_mask=True, return_logprob=True)
                    req.logprob.top_logprobs_num = top_logprobs_num
                    abort, add_to_queue = _handle_pd_request(mode, req)
                    abort.assert_called_once()
                    self.assertIn(
                        (
                            "top_logprobs_num"
                            if top_logprobs_num > capacity
                            else "return_sampling_mask"
                        ),
                        abort.call_args.args[1],
                    )
                    self.assertEqual(
                        abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST
                    )
                    add_to_queue.assert_not_called()

    def test_rejected_streaming_turn_leaves_session_idle(self):
        """A turn rejected after `create_req` must not wedge its streaming session."""
        for name in ("sampling_mask", "dllm"):
            with self.subTest(rejection=name):
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.enable_session_radix_cache = False
                scheduler.model_config = SimpleNamespace(
                    hf_eos_token_id={1}, vocab_size=128
                )
                scheduler.disaggregation_mode = DisaggregationMode.NULL
                scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
                scheduler.tokenizer = None
                scheduler.dllm_config = None
                scheduler.validate_dllm_request = lambda req: (
                    "dllm rejection" if name == "dllm" else None
                )
                scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
                scheduler._record_kv_event_lora_name = MagicMock()
                scheduler.spec_algorithm = SimpleNamespace(
                    is_dflash_family=lambda: False,
                    is_uno=lambda: False,
                    is_none=lambda: True,
                )
                scheduler.send_abort_output = MagicMock()
                scheduler.output_streamer = MagicMock()
                scheduler.tree_cache = MagicMock()
                scheduler.beam_coordinator = MagicMock()
                scheduler.hisparse_coordinator = None
                scheduler.uocr_runtime = None
                scheduler.init_request_retirer()
                scheduler.server_args = SimpleNamespace(sampling_mask_max_tokens=64)
                scheduler._create_generation_req = lambda recv_req, **kw: Req(**kw)
                scheduler.session_controller = SessionController(MagicMock())
                scheduler.session_controller.open(
                    SimpleNamespace(
                        session_id="s",
                        capacity_of_str_len=0,
                        streaming=True,
                        timeout=None,
                    )
                )
                session = scheduler.session_controller.get("s")
                recv_req = MagicMock(
                    rid="r",
                    input_ids=array("q", [5, 6, 7]),
                    input_text="",
                    mm_inputs=None,
                    session_id=None,
                    session_params=SimpleNamespace(
                        id="s",
                        rid=None,
                        offset=None,
                        replace=False,
                        drop_previous_output=False,
                    ),
                    # The default top_k is invalid together with a sampling mask.
                    sampling_params=SamplingParams(max_new_tokens=8),
                    return_sampling_mask=name == "sampling_mask",
                    return_logprob=False,
                    return_hidden_states=False,
                    pp_prefetch_ticketed=False,
                    bootstrap_port=1,
                )
                with (
                    patch(
                        "sglang.srt.managers.scheduler.BeamCoordinator.request_beam_width",
                        return_value=1,
                    ),
                    patch(
                        "sglang.srt.managers.scheduler_components.request_retirer.make_abort_req"
                    ),
                ):
                    scheduler.handle_generate_request(recv_req)
                self.assertFalse(session._inflight)

    def test_spec_worker_rejection_returns_bad_request_before_admission(self):
        """A spec worker that cannot serve a request answers 400 instead of failing its batch."""
        _RejectingSpecWorker.seen.clear()
        req, abort, add_to_queue = _handle_spec_request(
            _RejectingSpecWorker, draft_worker=MagicMock()
        )
        self.assertEqual(_RejectingSpecWorker.seen, [req])
        abort.assert_called_once()
        self.assertEqual(abort.call_args.args[1], _RejectingSpecWorker.reason)
        self.assertEqual(abort.call_args.kwargs["status_code"], HTTPStatus.BAD_REQUEST)
        add_to_queue.assert_not_called()

    def test_spec_rejection_does_not_depend_on_a_local_draft_worker(self):
        """A PP stage without a draft must reject what the last stage rejects."""
        _, abort, add_to_queue = _handle_spec_request(
            _RejectingSpecWorker, draft_worker=None
        )
        abort.assert_called_once()
        add_to_queue.assert_not_called()

    def test_spec_worker_without_validate_request_admits_the_request(self):
        """A worker that does not override validate_request leaves admission as before."""

        class _DefaultSpecWorker(BaseSpecWorker):
            pass

        req, abort, add_to_queue = _handle_spec_request(
            _DefaultSpecWorker, draft_worker=MagicMock()
        )
        abort.assert_not_called()
        req.set_finish_with_abort.assert_not_called()
        add_to_queue.assert_called_once()
        self.assertIs(add_to_queue.call_args.args[0], req)


if __name__ == "__main__":
    unittest.main()
