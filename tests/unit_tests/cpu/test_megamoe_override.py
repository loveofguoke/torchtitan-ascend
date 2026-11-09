# SPDX-License-Identifier: Apache-2.0
"""Check the optional MegaMoE adapter's configuration and weight/gradient seam."""

import builtins
import copy
import sys
import unittest
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from torchtitan.config import derive
from torchtitan.config.override import apply_overrides
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.common.moe.token_dispatcher import LocalTokenDispatcher
from torchtitan_ascend.config.override_utils import add_override
from torchtitan_ascend.models.deepseek_v4_1 import DeepSeekV41Model
from torchtitan_ascend.overrides.hp_components.megamoe import (
    _ExecutionGroup,
    hyper_megamoe,
    MegaMoeRoutedExperts,
)
from torchtitan_ascend.recipes.deepseek_v4_1 import deepseek_v4_1_debugmodel_npu


class ReferenceExecutor(torch.nn.Module):
    """Differentiable external-parameter executor; no native/NPU claim."""

    def forward(self, x, topk_ids, topk_weights, *, expert_weights):
        gate_up, down = expert_weights
        projected = torch.einsum("td,tkfd->tkf", x, gate_up[topk_ids])
        gate, up = projected.chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        routed = torch.einsum("tkf,tkdf->tkd", hidden, down[topk_ids])
        return (routed.float() * topk_weights.unsqueeze(-1)).sum(1).to(x.dtype)

    def close(self):
        self.closed = True


class TransformedLinear(GroupedLinear):
    """Represent a projection whose forward math cannot be bypassed."""

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedLinear.Config):
        pass


def source_config():
    return RoutedExperts.Config(
        w13=GroupedLinear.Config(
            group_size=4, in_features=8, out_features=8, num_linears=2
        ),
        w2=GroupedLinear.Config(group_size=4, in_features=8, out_features=8),
        token_dispatcher=LocalTokenDispatcher.Config(num_experts=4, top_k=2),
    )


class TestMegaMoeOverride(unittest.TestCase):
    def test_workspace_sharing_rejects_unconfigured_layers(self):
        with patch("torchtitan_ascend.overrides.hp_components.megamoe.require_device"):
            module = hyper_megamoe(source_config()).build()
        group = _ExecutionGroup.register(module, SimpleNamespace())
        with self.assertRaisesRegex(RuntimeError, "Configure all MegaMoE"):
            group.prepare()
        self.assertFalse(group.prepared)

    def test_recipe_claims_routed_experts_without_activation_conflicts(self):
        with patch(
            "torch.npu.init", side_effect=AssertionError("config initialized NPU")
        ):
            config = deepseek_v4_1_debugmodel_npu(ep=8)
            add_override(config, hyper_megamoe)
            replacements = apply_overrides(config.override, config)
        self.assertIs(type(config.model)._owner, DeepSeekV41Model)
        experts = [layer.feed_forward.routed_experts for layer in config.model.layers]
        self.assertEqual(len(experts), 40)
        self.assertTrue(
            all(isinstance(c, MegaMoeRoutedExperts.Config) for c in experts)
        )
        self.assertEqual(sum("hyper_megamoe" in line for line in replacements), 40)
        baseline = deepseek_v4_1_debugmodel_npu(ep=8)
        apply_overrides(baseline.override, baseline)
        self.assertIs(type(baseline.model)._owner, DeepSeekV41Model)
        self.assertTrue(
            all(
                type(layer.feed_forward.routed_experts) is RoutedExperts.Config
                for layer in baseline.model.layers
            )
        )

    def test_execution_group_collects_layers_without_model_hooks(self):
        with patch("torchtitan_ascend.overrides.hp_components.megamoe.require_device"):
            modules = [hyper_megamoe(source_config()).build() for _ in range(2)]
        context = SimpleNamespace()
        for module in modules:
            module._execution_group = _ExecutionGroup.register(module, context)
        group = modules[0]._execution_group
        self.assertIs(group, modules[1]._execution_group)
        self.assertEqual(set(group.members), set(modules))
        for module in modules:
            module._executor = ReferenceExecutor()
        native = Mock()
        with patch.dict(
            sys.modules,
            {"hyper_parallel.core.multicore": SimpleNamespace(MegaMoeExperts=native)},
        ):
            group.prepare()
            group.prepare()
        native.share_execution_resources.assert_called_once()
        self.assertEqual(
            set(native.share_execution_resources.call_args.args[0]),
            {module._executor for module in modules},
        )
        self.assertTrue(group.prepared)
        with self.assertRaisesRegex(RuntimeError, "before first forward"):
            _ExecutionGroup.register(modules[0], context)
        other = _ExecutionGroup.register(modules[0], SimpleNamespace())
        self.assertIsNot(other, group)
        for module in modules:
            module.close()
        self.assertEqual(len(group.members), 0)

    def test_config_generation_does_not_require_hyper(self):
        original_import = builtins.__import__

        def without_hyper(name, *args, **kwargs):
            if name == "hyper_parallel" or name.startswith("hyper_parallel."):
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_hyper):
            baseline = deepseek_v4_1_debugmodel_npu()
            mega = deepseek_v4_1_debugmodel_npu(ep=8)
            add_override(mega, hyper_megamoe)
            apply_overrides(mega.override, mega)
        self.assertIs(type(baseline.model)._owner, DeepSeekV41Model)
        self.assertIs(type(mega.model)._owner, DeepSeekV41Model)

    def test_close_is_idempotent_and_preserves_parameter_identity(self):
        with patch("torchtitan_ascend.overrides.hp_components.megamoe.require_device"):
            module = hyper_megamoe(source_config()).build()
        parameters = dict(module.named_parameters())
        executor = ReferenceExecutor()
        module._executor = executor
        module.close()
        module.close()
        self.assertTrue(executor.closed)
        self.assertEqual(set(parameters), set(dict(module.named_parameters())))
        for name, parameter in module.named_parameters():
            self.assertIs(parameter, parameters[name])
        with self.assertRaisesRegex(RuntimeError, "before forward"):
            module(None, None, None, None)

    def test_projection_transforms_are_not_silently_discarded(self):
        config = source_config()
        projection = derive(config.w13, TransformedLinear.Config)
        with self.assertRaisesRegex(ValueError, "unmodified GroupedLinear"):
            hyper_megamoe(replace(config, w13=projection))

    def test_weight_layout_and_gradients_keep_original_parameters(self):
        torch.manual_seed(7)
        with patch("torchtitan_ascend.overrides.hp_components.megamoe.require_device"):
            module = hyper_megamoe(source_config()).build()
        module._executor = ReferenceExecutor()
        for parameter in module.parameters():
            torch.nn.init.normal_(parameter, std=0.1)
        reference = copy.deepcopy(module)
        x = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
        scores = torch.rand(5, 2, requires_grad=True)
        ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
        expected_x = x.detach().clone().requires_grad_()
        expected_scores = scores.detach().clone().requires_grad_()
        pieces = []
        for token in range(5):
            terms = []
            for choice in range(2):
                expert = ids[token, choice]
                weight = reference.w13.weight[expert].bfloat16()
                gate = F.linear(expected_x[token], weight[0])
                up = F.linear(expected_x[token], weight[1])
                hidden = F.silu(gate) * up
                routed = F.linear(hidden, reference.w2.weight[expert].bfloat16())
                terms.append(routed.float() * expected_scores[token, choice])
            pieces.append(sum(terms))
        expected = torch.stack(pieces).bfloat16()
        actual = module(x, scores, ids, torch.bincount(ids.flatten(), minlength=4))
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.001)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        torch.testing.assert_close(x.grad, expected_x.grad, rtol=0.03, atol=0.001)
        torch.testing.assert_close(
            scores.grad, expected_scores.grad, rtol=0.03, atol=0.001
        )
        self.assertEqual(set(module.state_dict()), {"w13.weight", "w2.weight"})
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                dict(reference.named_parameters())[name].grad,
                rtol=0.03,
                atol=0.001,
            )

    def test_unsupported_math_and_parallelism_fail_explicitly(self):
        config = source_config()
        with self.assertRaisesRegex(ValueError, "plain SwiGLU"):
            hyper_megamoe(replace(config, activation_fn=Sigmoid.Config()))
        with self.assertRaisesRegex(ValueError, "postprocessing"):
            hyper_megamoe(replace(config, output_postprocess=config.w2))
        with patch("torchtitan_ascend.overrides.hp_components.megamoe.require_device"):
            module = hyper_megamoe(config).build()
        for tp, cp, pp, ep in [(2, 1, 1, 8), (1, 2, 1, 8), (1, 1, 2, 8), (1, 1, 1, 1)]:
            with self.subTest(tp=tp, cp=cp, pp=pp, ep=ep):
                with self.assertRaisesRegex(ValueError, "TP=CP=PP=1 and EP>1"):
                    module._parallelize(SimpleNamespace(tp=tp, cp=cp, pp=pp, ep=ep))


if __name__ == "__main__":
    unittest.main()
