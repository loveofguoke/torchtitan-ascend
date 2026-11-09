# SPDX-License-Identifier: Apache-2.0
"""Check deterministic NPU selection and its compiled MLA integration."""

from types import SimpleNamespace

import pytest
import torch


def test_deterministic_npu_selection_uses_pytorch_policy(monkeypatch):
    from torch._inductor import config
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    from torchtitan_ascend.patches.inductor_selection import _restore_deterministic_selection

    def vendor_selector(*args, **kwargs):
        raise AssertionError("deterministic selection must not enter timed autotuning")

    vendor_selector.__module__ = "torch_npu._inductor.select_algorithm"
    monkeypatch.setattr(AlgorithmSelectorCache, "__call__", vendor_selector)
    _restore_deterministic_selection()
    selected = object()
    choices = [SimpleNamespace(output_node=lambda: selected), SimpleNamespace()]
    cache = AlgorithmSelectorCache()
    with config.patch(deterministic=True):
        node, choice = cache("flex_attention", choices, [], SimpleNamespace(device=torch.device("npu")))
    assert node is selected
    assert choice is cache.pick_deterministic_choice(choices)
    installed = AlgorithmSelectorCache.__call__
    _restore_deterministic_selection()
    assert AlgorithmSelectorCache.__call__ is installed


@pytest.mark.parametrize("device,deterministic", [("cpu", True), ("cuda", True), ("npu", False)])
def test_selection_preserves_other_backend_and_nondeterministic_policy(monkeypatch, device, deterministic):
    from torch._inductor import config
    from torch._inductor.select_algorithm import AlgorithmSelectorCache

    from torchtitan_ascend.patches.inductor_selection import _restore_deterministic_selection

    expected = object()

    def vendor_selector(*args, **kwargs):
        return expected

    vendor_selector.__module__ = "torch_npu._inductor.select_algorithm"
    monkeypatch.setattr(AlgorithmSelectorCache, "__call__", vendor_selector)
    _restore_deterministic_selection()
    with config.patch(deterministic=deterministic):
        result = AlgorithmSelectorCache()(
            "attention", [object(), object()], [], SimpleNamespace(device=torch.device(device))
        )
        assert result is expected


@pytest.mark.parametrize("defer_epilogue", [False, True])
def test_single_template_does_not_time_under_deferred_epilogue(monkeypatch, defer_epilogue):
    from torch._inductor import config
    from torch._inductor.select_algorithm import AlgorithmSelectorCache, TritonTemplateCaller

    from torchtitan_ascend.patches.inductor_selection import _restore_deterministic_selection

    node = object()
    choice = object.__new__(TritonTemplateCaller)
    monkeypatch.setattr(TritonTemplateCaller, "output_node", lambda self: node)

    def vendor_selector(self, name, choices, input_nodes, layout, *args, **kwargs):
        if len(choices) == 1 and not kwargs.get("defer_epilogue_compile_only", False):
            return choices[0].output_node(), choices[0]
        raise AssertionError("the NPU deferred-epilogue path would enter timed autotuning")

    vendor_selector.__module__ = "torch_npu._inductor.select_algorithm"
    monkeypatch.setattr(AlgorithmSelectorCache, "__call__", vendor_selector)
    _restore_deterministic_selection()
    with config.patch(deterministic=True):
        result = AlgorithmSelectorCache()(
            "flex_attention",
            [choice],
            [],
            SimpleNamespace(device=torch.device("npu")),
            defer_epilogue_compile_only=defer_epilogue,
        )
    assert result == (node, choice)


@pytest.mark.parametrize("fail", [False, True])
def test_compile_only_flags_are_restored_for_a_single_template(monkeypatch, fail):
    from torch._inductor import config
    from torch._inductor.select_algorithm import AlgorithmSelectorCache, TritonTemplateCaller

    from torchtitan_ascend.patches.inductor_selection import _restore_deterministic_selection

    node = object()
    choice = object.__new__(TritonTemplateCaller)
    monkeypatch.setattr(TritonTemplateCaller, "output_node", lambda self: node)
    calls = []

    def vendor_selector(self, name, choices, *args, **kwargs):
        assert all(candidate._nobench_select_first_compilable for candidate in choices)
        calls.append(choices[:])
        if fail:
            raise RuntimeError("no compilable template")
        return choices[0].output_node(), choices[0]

    vendor_selector.__module__ = "torch_npu._inductor.select_algorithm"
    monkeypatch.setattr(AlgorithmSelectorCache, "__call__", vendor_selector)
    _restore_deterministic_selection()
    with config.patch(deterministic=True):
        def invoke():
            return AlgorithmSelectorCache()(
                "flex_attention", [choice], [], SimpleNamespace(device=torch.device("npu")),
                defer_epilogue_compile_only=True,
            )
        if fail:
            with pytest.raises(RuntimeError, match="no compilable template"):
                invoke()
        else:
            assert invoke() == (node, choice)
    assert calls == [[choice]]
    assert not hasattr(choice, "_nobench_select_first_compilable")


@pytest.mark.npu
def test_deterministic_compiled_flex_preserves_strided_mla_gradients():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")
    from torchtitan.models.common.attention import FlexInnerAttention

    torch.npu.set_device(0)
    torch.manual_seed(42)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.autograd.set_multithreading_enabled(False):
            q = torch.randn(128, 1, 32, device="npu", dtype=torch.bfloat16, requires_grad=True)
            k = torch.randn_like(q).requires_grad_()
            # MLA splits packed KV; V keeps the parent's wider token stride.
            v = torch.randn_like(q)[..., :16].detach().requires_grad_()
            config = FlexInnerAttention.Config()
            metadata = config.build_attention_metadata(torch.arange(128, device="npu"))
            output = config.build()(q, k, v, attention_metadata=metadata)
            gradients = torch.autograd.grad(output.float().square().mean(), (q, k, v))
            reference_inputs = [x.detach().cpu().float().requires_grad_() for x in (q, k, v)]
            reference = (
                torch.nn.functional.scaled_dot_product_attention(
                    *(x.transpose(0, 1).unsqueeze(0) for x in reference_inputs), is_causal=True
                )
                .squeeze(0)
                .transpose(0, 1)
            )
            reference_gradients = torch.autograd.grad(reference.square().mean(), reference_inputs)
            for actual, expected in zip((output, *gradients), (reference, *reference_gradients), strict=True):
                relative_error = (actual.detach().cpu().float() - expected).norm() / expected.norm()
                assert relative_error < 0.01
            torch.npu.synchronize()
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
