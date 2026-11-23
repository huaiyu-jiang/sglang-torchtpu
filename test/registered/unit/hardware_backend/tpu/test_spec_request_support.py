"""TpuSpecWorker turns away, at admission, every request feature its verify step cannot run."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.request_support import (
    unsupported_request_features,
)


def _req(return_logprob=False, custom_logit_processor=None, **params):
    return SimpleNamespace(
        sampling_params=SamplingParams(max_new_tokens=8, **params),
        return_logprob=return_logprob,
        custom_logit_processor=custom_logit_processor,
    )


def _features(req, **flags):
    kwargs = dict(
        supports_grammar=False,
        supports_logprobs=False,
        supports_additive_penalties=False,
        bsearch_filtering=False,
        custom_logit_processor_enabled=True,
    )
    kwargs.update(flags)
    return unsupported_request_features(req, **kwargs)


class TestSpecRequestSupport(CustomTestCase):
    def test_each_unsupported_feature_is_named(self):
        """Every feature _check_supported fails a batch on is rejected per request."""
        cases = {
            "repetition_penalty": _req(repetition_penalty=1.05),
            "frequency_penalty": _req(frequency_penalty=0.5),
            "presence_penalty": _req(presence_penalty=0.5),
            "min_new_tokens": _req(min_new_tokens=4),
            "logit_bias": _req(logit_bias={"7": 2.0}),
            "structured output": _req(json_schema='{"type": "object"}'),
            "logprobs": _req(return_logprob=True),
            "custom_logit_processor": _req(custom_logit_processor="serialized"),
        }
        for name, req in cases.items():
            with self.subTest(feature=name):
                features = _features(req)
                self.assertEqual(len(features), 1, features)
                self.assertTrue(features[0].startswith(name), features)

    def test_plain_requests_are_accepted(self):
        for req in (_req(temperature=0.0), _req(temperature=0.7)):
            with self.subTest(params=req.sampling_params):
                self.assertEqual(_features(req), [])

    def test_additive_opt_in_still_rejects_scaling_penalty(self):
        """The additive snapshot opt-in covers additive penalties, never repetition."""
        additive = _req(frequency_penalty=0.5, presence_penalty=0.5, min_new_tokens=4)
        self.assertEqual(_features(additive, supports_additive_penalties=True), [])
        self.assertEqual(
            _features(_req(repetition_penalty=1.05), supports_additive_penalties=True),
            ["repetition_penalty"],
        )

    def test_filtered_sampling_needs_bsearch_only_for_sampled_rows(self):
        filtered = _req(temperature=0.7, top_p=0.9, top_k=50)
        self.assertEqual(len(_features(filtered)), 1)
        self.assertEqual(_features(filtered, bsearch_filtering=True), [])
        # Greedy rows (top_k normalized to 1) take the greedy rule, never the filter.
        self.assertEqual(_features(_req(temperature=0.0, top_p=0.9)), [])

    def test_worker_rejects_instead_of_failing_the_batch(self):
        """A penalty request used to crash the scheduler inside verify; it gets a reason now."""
        exec_ctx = SimpleNamespace(
            features=SimpleNamespace(enable_custom_logit_processor=False)
        )
        with patch(
            "sglang_tpu.experimental.speculative.base_spec_worker.get_exec",
            return_value=exec_ctx,
        ):
            reason = TpuSpecWorker.validate_request(_req(frequency_penalty=0.5))
            self.assertIn("frequency_penalty", reason)
            self.assertIsNone(TpuSpecWorker.validate_request(_req(temperature=0.0)))


if __name__ == "__main__":
    unittest.main()
