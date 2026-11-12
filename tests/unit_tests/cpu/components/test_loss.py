# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

# Load Torch and its device extensions before SPMD registration.
import torch  # usort: skip
import spmd_types as spmd  # usort: skip
import torch.nn as nn
import torch.nn.functional as F
from spmd_types._checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import loss_parallel
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.components.loss import (
    _LossParallelCrossEntropy,
    BaseLoss,
    compute_logprobs,
    cross_entropy_loss,
    GradAccumulator,
    IGNORE_INDEX,
)
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.deepseek_v3.mtp import (
    get_mtp_token_counts,
    MTPDecoder,
    MTPLoss,
    roll_mtp_sequence,
)


class TestLoss(unittest.TestCase):
    def test_compute_logprobs_token_major(self):
        torch.manual_seed(42)
        logits = torch.randn(8, 16)
        labels = torch.randint(0, 16, (8,))

        logprobs, entropy = compute_logprobs(
            logits,
            labels,
            vocab_parallel_group=None,
            return_entropy=True,
            global_vocab_size=logits.shape[-1],
        )
        expected_logprobs = -F.cross_entropy(logits, labels, reduction="none")
        expected_entropy = torch.logsumexp(logits, dim=-1) - (
            torch.softmax(logits, dim=-1) * logits
        ).sum(dim=-1)

        torch.testing.assert_close(logprobs, expected_logprobs)
        torch.testing.assert_close(entropy, expected_entropy)

    def test_compute_logprobs_entropy_with_masked_logits(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                logits = torch.tensor(
                    [[0.0, 1.0, -torch.inf], [0.0, -torch.inf, -torch.inf]],
                    dtype=dtype,
                    requires_grad=True,
                )
                labels = torch.tensor([0, IGNORE_INDEX])
                logprobs, entropy = compute_logprobs(
                    logits, labels, vocab_parallel_group=None, return_entropy=True
                )
                expected_entropy = torch.distributions.Categorical(
                    logits=logits.detach().double()
                ).entropy()

                torch.testing.assert_close(
                    entropy.double(), expected_entropy, rtol=1e-6, atol=1e-7
                )
                torch.testing.assert_close(
                    logprobs, -F.cross_entropy(logits.float(), labels, reduction="none")
                )
                self.assertFalse(entropy.requires_grad)

    def test_compute_logprobs_entropy_preserves_invalid_distributions(self):
        logits = torch.tensor(
            [[-torch.inf, -torch.inf], [torch.inf, 0.0], [torch.nan, 0.0]]
        )
        _, entropy = compute_logprobs(
            logits,
            torch.zeros(3, dtype=torch.long),
            vocab_parallel_group=None,
            return_entropy=True,
        )

        self.assertTrue(torch.isnan(entropy).all())

    def test_roll_mtp_sequence_respects_packed_document_boundaries(self):
        tokens = torch.tensor([10, 11, 12, 20, 21, 22, 23, 24])
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])

        torch.testing.assert_close(
            roll_mtp_sequence(tokens, shift=1, positions=positions, fill_value=0),
            torch.tensor([11, 12, 0, 21, 22, 23, 24, 0]),
        )
        torch.testing.assert_close(
            roll_mtp_sequence(tokens, shift=2, positions=positions, fill_value=0),
            torch.tensor([12, 0, 0, 22, 23, 24, 0, 0]),
        )

    def test_roll_mtp_sequence_requires_current_and_shifted_tokens_non_padding(self):
        tokens = torch.tensor([10, 11, 12, 13])
        positions = torch.arange(4)
        padding_mask = torch.tensor([True, False, False, False])

        shifted, valid_mask = roll_mtp_sequence(
            tokens,
            shift=1,
            positions=positions,
            padding_mask=padding_mask,
            fill_value=0,
            return_valid_mask=True,
        )

        torch.testing.assert_close(shifted, torch.tensor([0, 12, 13, 0]))
        torch.testing.assert_close(valid_mask, torch.tensor([False, True, True, False]))

    def test_mtp_target_and_routing_masks_have_distinct_semantics(self):
        labels = torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 12, 13])
        positions = torch.arange(4)

        shifted_labels, routing_valid_mask = roll_mtp_sequence(
            labels,
            shift=1,
            positions=positions,
            fill_value=IGNORE_INDEX,
            return_valid_mask=True,
        )

        torch.testing.assert_close(
            shifted_labels,
            torch.tensor([IGNORE_INDEX, 12, 13, IGNORE_INDEX]),
        )
        torch.testing.assert_close(
            routing_valid_mask, torch.tensor([True, True, True, False])
        )
        self.assertEqual(int((shifted_labels != IGNORE_INDEX).sum()), 2)
        self.assertEqual(int(routing_valid_mask.sum()), 3)

    def test_mtp_preprocess_aligns_tokens_and_labels(self):
        # Two packed documents: [A0, A1, A2, B0, B1, B2, B3, B4].
        # TorchTitan positions reset at each document boundary.
        tokens = torch.tensor([10, 11, 12, 20, 21, 22, 23, 24])
        positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
        labels = torch.arange(8)
        padding_mask = torch.tensor(
            [False, False, False, False, False, False, True, True]
        )
        model = _FakeMTPDecoder(skip_lm_head=True, num_mtp_layers=2)
        routing_token_counts = torch.tensor([6, 4, 2])  # one per depth
        with patch(
            "torchtitan.models.deepseek_v3.mtp.annotate_input_spmd_types",
            side_effect=lambda _parallelism_context, batch, _input_sharding: batch,
        ):
            input_tokens, loss_labels, extra_kwargs, _ = model.preprocess_inputs(
                {
                    "input": tokens,
                    "labels": labels,
                    "positions": positions,
                    "padding_mask": padding_mask,
                },
                parallelism_context=SimpleNamespace(cp_enabled=False),
                parallelism=SimpleNamespace(),
                max_num_documents=2,
                max_context_length=8,
                global_routing_token_counts=routing_token_counts,
            )

        assert extra_kwargs["aux_loss_denominators"] is routing_token_counts
        assert isinstance(input_tokens, tuple)
        assert isinstance(loss_labels, tuple)
        torch.testing.assert_close(input_tokens[0], tokens)
        torch.testing.assert_close(
            input_tokens[1], torch.tensor([11, 12, 0, 21, 22, 0, 0, 0])
        )
        torch.testing.assert_close(
            input_tokens[2], torch.tensor([12, 0, 0, 22, 0, 0, 0, 0])
        )
        torch.testing.assert_close(loss_labels[0], labels)
        torch.testing.assert_close(
            loss_labels[1],
            torch.tensor(
                [1, 2, IGNORE_INDEX, 4, 5, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX]
            ),
        )
        torch.testing.assert_close(
            loss_labels[2],
            torch.tensor(
                [
                    2,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    5,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                ]
            ),
        )
        self.assertEqual(len(extra_kwargs["mtp_input_valid_masks"]), 2)
        torch.testing.assert_close(
            extra_kwargs["mtp_input_valid_masks"][0],
            torch.tensor([True, True, False, True, True, False, False, False]),
        )
        torch.testing.assert_close(
            extra_kwargs["mtp_input_valid_masks"][1],
            torch.tensor([True, False, False, True, False, False, False, False]),
        )
        torch.testing.assert_close(extra_kwargs["padding_mask"], padding_mask)
        loss_token_counts, routing_token_counts = get_mtp_token_counts(
            target_mask=labels != IGNORE_INDEX,
            positions=positions,
            padding_mask=padding_mask,
            num_mtp_layers=2,
        )
        torch.testing.assert_close(loss_token_counts, torch.tensor([8, 4, 2]))
        torch.testing.assert_close(routing_token_counts, torch.tensor([6, 4, 2]))

    def test_mtp_counts_chat_targets_separately_from_routing(self):
        labels = torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 12, 13])
        positions = torch.arange(4)
        padding_mask = torch.zeros(4, dtype=torch.bool)
        loss_token_counts, routing_token_counts = get_mtp_token_counts(
            target_mask=labels != IGNORE_INDEX,
            positions=positions,
            padding_mask=padding_mask,
            num_mtp_layers=1,
        )
        torch.testing.assert_close(loss_token_counts, torch.tensor([2, 2]))
        torch.testing.assert_close(routing_token_counts, torch.tensor([4, 3]))

    def test_mtp_loss_rejects_plain_tensor(self):
        loss_fn = MTPLoss(MTPLoss.Config(global_vocab_size=16))
        pred = torch.zeros(1, 4, 16)
        labels = torch.zeros(1, 4, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "expects prediction and labels tuples"):
            loss_fn(pred, labels)

    def test_mtp_loss_uses_each_depth_target_count(self):
        loss_fn = MTPLoss(MTPLoss.Config(mtp_scale=0.3))
        logits = tuple(torch.zeros(4, 4, requires_grad=True) for _ in range(3))
        labels = (
            torch.zeros(4, dtype=torch.long),
            torch.tensor([0, 0, 0, IGNORE_INDEX]),
            torch.tensor([0, 0, IGNORE_INDEX, IGNORE_INDEX]),
        )
        loss_token_counts = torch.tensor([4, 3, 2])

        loss, _ = loss_fn(logits, labels, loss_token_counts)

        torch.testing.assert_close(loss, torch.log(torch.tensor(4.0)) * 1.3)
        loss.backward()
        torch.testing.assert_close(logits[0].grad[0, 0], torch.tensor(-0.75 / 4))
        torch.testing.assert_close(logits[1].grad[0, 0], torch.tensor(-0.15 * 0.75 / 3))
        torch.testing.assert_close(logits[2].grad[0, 0], torch.tensor(-0.15 * 0.75 / 2))

    def test_mtp_loss_without_counts_returns_unnormalized_sum(self):
        loss_fn = MTPLoss(MTPLoss.Config(mtp_scale=0.3))
        logits = tuple(torch.zeros(4, 4) for _ in range(2))
        labels = tuple(torch.zeros(4, dtype=torch.long) for _ in range(2))

        loss, _ = loss_fn(logits, labels)

        torch.testing.assert_close(loss, torch.log(torch.tensor(4.0)) * 4 * 1.3)

    def test_ignore_index_equal_per_token_contribution(self):
        """Test that each valid token contributes equally to the loss.

        This test verifies that:
        1. Tokens marked with IGNORE_INDEX don't contribute to the loss
        2. Each valid token contributes the same amount to the total loss
        3. The sum-based loss calculation is correct for token normalization
        """
        torch.manual_seed(42)
        num_tokens = 32
        vocab_size = 100

        # Create predictions (logits) - same for all test cases
        predictions = torch.randn(num_tokens, vocab_size)

        # Create base labels with some tokens as IGNORE_INDEX, others valid
        # This ensures we test on the same subset of valid tokens
        labels = torch.randint(0, vocab_size, (num_tokens,))
        # Mark specific positions as IGNORE_INDEX
        labels[1] = IGNORE_INDEX
        labels[11] = IGNORE_INDEX
        labels[21] = IGNORE_INDEX
        labels[31] = IGNORE_INDEX

        # Test case 1: Compute loss on this label set
        loss1 = cross_entropy_loss(predictions, labels)
        num_loss_tokens1 = (labels != IGNORE_INDEX).sum().item()

        # Test case 2: Use the exact same predictions and labels in multiple microbatches
        # Simulating gradient accumulation with identical data
        loss2 = cross_entropy_loss(predictions, labels) + cross_entropy_loss(
            predictions, labels
        )
        num_loss_tokens2 = num_loss_tokens1 * 2

        # Per-token loss should be identical
        per_token_loss1 = loss1 / num_loss_tokens1
        per_token_loss2 = loss2 / num_loss_tokens2

        self.assertAlmostEqual(
            per_token_loss1.item(),
            per_token_loss2.item(),
            places=6,
            msg="Per-token loss should be the same across gradient accumulation steps",
        )

        # Test case 3: Verify loss scaling with token replication
        # Concatenate the sequence to create 2x the data
        predictions_doubled = torch.cat([predictions, predictions], dim=0)
        labels_doubled = torch.cat([labels, labels], dim=0)

        loss_doubled = cross_entropy_loss(predictions_doubled, labels_doubled)
        num_loss_tokens_doubled = (labels_doubled != IGNORE_INDEX).sum().item()

        per_token_loss_doubled = loss_doubled / num_loss_tokens_doubled

        self.assertAlmostEqual(
            per_token_loss1.item(),
            per_token_loss_doubled.item(),
            places=6,
            msg="Per-token loss should remain constant when scaling token count",
        )

        # Verify that total loss scales linearly with number of valid tokens
        expected_ratio = num_loss_tokens_doubled / num_loss_tokens1
        actual_ratio = loss_doubled / loss1

        self.assertAlmostEqual(
            expected_ratio,
            actual_ratio.item(),
            places=6,
            msg=f"Loss should scale linearly with valid token count. "
            f"Expected ratio: {expected_ratio}, got: {actual_ratio.item()}",
        )

    def test_ignore_index_gradient_accumulation_consistency(self):
        """Test that loss is consistent across gradient accumulation steps.

        This simulates the scenario where we have:
        - Multiple microbatches with different numbers of valid tokens
        - Total loss should equal sum of individual losses
        - Per-token loss should be consistent
        """
        torch.manual_seed(123)
        vocab_size = 100

        # Microbatch 1: eight valid tokens
        pred1 = torch.randn(8, vocab_size)
        labels1 = torch.randint(0, vocab_size, (8,))
        loss1 = cross_entropy_loss(pred1, labels1)
        tokens1 = (labels1 != IGNORE_INDEX).sum()

        # Microbatch 2: half valid tokens
        pred2 = torch.randn(8, vocab_size)
        labels2 = torch.randint(0, vocab_size, (8,))
        labels2[::2] = IGNORE_INDEX  # Mask every other token
        loss2 = cross_entropy_loss(pred2, labels2)
        tokens2 = (labels2 != IGNORE_INDEX).sum()

        # Microbatch 3: two valid tokens
        pred3 = torch.randn(8, vocab_size)
        labels3 = torch.randint(0, vocab_size, (8,))
        labels3[2:] = IGNORE_INDEX
        loss3 = cross_entropy_loss(pred3, labels3)
        tokens3 = (labels3 != IGNORE_INDEX).sum()

        # Simulate gradient accumulation: sum losses, sum tokens, then normalize
        total_loss = loss1 + loss2 + loss3
        total_tokens = tokens1 + tokens2 + tokens3
        global_avg_loss = total_loss / total_tokens

        # Verify this equals the average of individual per-token losses weighted by token count
        weighted_avg = (
            (loss1 / tokens1) * tokens1
            + (loss2 / tokens2) * tokens2
            + (loss3 / tokens3) * tokens3
        ) / total_tokens

        self.assertAlmostEqual(
            global_avg_loss.item(),
            weighted_avg.item(),
            places=5,
            msg="Global averaged loss should equal weighted average of per-token losses",
        )


class TestGradAccumulator(unittest.TestCase):
    def test_accumulate_matches_cat(self):
        """Verify GradAccumulator produces the same result as torch.cat."""
        torch.manual_seed(42)
        T, D = 16, 16
        num_chunks = 4
        reference = torch.randn(T, D)
        chunks = torch.chunk(reference, num_chunks, dim=0)

        acc = GradAccumulator(
            reference,
            num_chunks=num_chunks,
            dtype=reference.dtype,
        )
        for chunk in chunks:
            acc.add(chunk)

        result = acc.buffer
        torch.testing.assert_close(result, reference)

    def test_accumulate_with_dtype_conversion(self):
        """Verify fp32 accumulation from bf16 chunks."""
        torch.manual_seed(42)
        T, D = 16, 16
        num_chunks = 4
        reference = torch.randn(T, D)
        bf16_chunks = [c.bfloat16() for c in torch.chunk(reference, num_chunks, dim=0)]

        acc = GradAccumulator(reference, num_chunks=num_chunks, dtype=torch.float32)
        for chunk in bf16_chunks:
            acc.add(chunk)

        result = acc.buffer
        self.assertEqual(result.dtype, torch.float32)
        # Verify values match (allowing for bf16 precision loss)
        expected = torch.cat([c.float() for c in bf16_chunks], dim=0)
        torch.testing.assert_close(result, expected)

    def test_too_many_adds_raises(self):
        """Verify error when adding more chunks than expected."""
        acc = GradAccumulator(torch.randn(8, 16), num_chunks=2, dtype=torch.float32)
        acc.add(torch.randn(4, 16))
        acc.add(torch.randn(4, 16))
        with self.assertRaises(ValueError):
            acc.add(torch.randn(4, 16))


class TestLossParallelCrossEntropy(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_loss_parallel_cross_entropy_parity(self):
        """
        Tests loss-parallel cross-entropy loss bitwise parity, with torch.distributed.tensor.parallel.loss_parallel().
        Tests even/uneven vocab sharding, TP, DP+TP, and IGNORE_INDEX labels.

        Runs _LossParallelCrossEntropy under typing checking: logits S(1)@TP, labels I@TP -> I@TP.
        """
        # Ensure the determinism.
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        T = 128
        mesh_configs = (
            ((4,), ("tp",), (Shard(1),), (Replicate(),)),
            ((2, 2), ("dp", "tp"), (Shard(0), Shard(1)), (Shard(0), Replicate())),
        )
        cases = ((32000, 109, False), (32003, 211, False), (32000, 307, True))

        for mesh_shape, axis_names, logits_placements, label_placements in mesh_configs:
            mesh = init_device_mesh(
                self.device_type, mesh_shape, mesh_dim_names=axis_names
            )
            tp_group = mesh.get_group("tp")
            for vocab_size, seed, ignore in cases:
                with self.subTest(
                    mesh_shape=mesh_shape, vocab_size=vocab_size, ignore=ignore
                ):
                    # Build global test data once, then let DTensor derive the
                    # exact local shards for TP-only and DP+TP placements.
                    # Sample on CPU and move to the test device: not every
                    # accelerator registers a device-side torch.Generator.
                    generator = torch.Generator().manual_seed(seed)
                    global_logits = torch.randn(T, vocab_size, generator=generator).to(
                        self.device_type
                    )
                    global_labels = torch.randint(
                        0, vocab_size, (T,), dtype=torch.long, generator=generator
                    ).to(self.device_type)
                    if ignore:
                        mask = torch.rand(T, generator=generator).to(self.device_type)
                        global_labels[mask < 0.3] = IGNORE_INDEX
                    # distribute as DTensor; loss_parallel wrapper takes DTensors.
                    logits_dtensor = distribute_tensor(
                        global_logits,
                        mesh,
                        logits_placements,
                    ).detach()
                    logits_dtensor.requires_grad_(True)
                    labels_dtensor = distribute_tensor(
                        global_labels,
                        mesh,
                        label_placements,
                    )

                    # pytorch loss_parallel() as ground truth.
                    with loss_parallel():
                        wrapper_loss = F.cross_entropy(
                            logits_dtensor.float(),
                            labels_dtensor,
                            reduction="sum",
                            ignore_index=IGNORE_INDEX,
                        )

                    # typecheck S(1)@TP, I@TP -> I@TP
                    logits_type = {tp_group: spmd.S(1)}
                    labels_type = {tp_group: spmd.I}
                    if "dp" in axis_names:
                        dp_group = mesh.get_group("dp")
                        logits_type[dp_group] = spmd.S(0)
                        labels_type[dp_group] = spmd.S(0)

                    # run custom autograd
                    local_logits = (
                        logits_dtensor.to_local().detach().requires_grad_(True)
                    )
                    local_labels = labels_dtensor.to_local()
                    spmd.assert_type(local_logits, logits_type)
                    spmd.assert_type(local_labels, labels_type)
                    with typecheck(strict_mode="strict"):
                        local_loss = _LossParallelCrossEntropy.apply(
                            local_logits,
                            local_labels,
                            tp_group,
                            vocab_size,
                            "sum",
                        )
                    self.assertIs(
                        spmd.get_axis_local_type(local_loss, tp_group), spmd.I
                    )

                    # Loss and local-shard gradients must be bitwise-equivalent
                    # to torch.distributed.tensor.parallel.loss_parallel().
                    self.assertTrue(torch.equal(local_loss, wrapper_loss.to_local()))

                    # compare grad_logits; easier to wrap DTensor -> full_tensor for check.
                    with loss_parallel():
                        wrapper_loss.backward()
                    local_loss.backward()
                    local_grad_dtensor = DTensor.from_local(
                        local_logits.grad,
                        mesh,
                        logits_placements,
                        shape=torch.Size((T, vocab_size)),
                        stride=(vocab_size, 1),
                    )
                    self.assertTrue(
                        torch.equal(
                            local_grad_dtensor.full_tensor(),
                            logits_dtensor.grad.full_tensor(),
                        )
                    )

    @with_comms
    def test_vocab_parallel_policy_stats_parity(self):
        """Check no-gather logprobs, entropy, gradients, and SPMD types."""
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        T = 32
        mesh_configs = (
            ((4,), ("tp",), (Shard(1),), (Replicate(),)),
            ((2, 2), ("dp", "tp"), (Shard(0), Shard(1)), (Shard(0), Replicate())),
        )
        cases = (
            (128, torch.float32, False),
            (131, torch.bfloat16, True),
        )

        for mesh_shape, axis_names, logits_placements, label_placements in mesh_configs:
            mesh = init_device_mesh(
                self.device_type, mesh_shape, mesh_dim_names=axis_names
            )
            tp_group = mesh.get_group("tp")
            for vocab_size, dtype, ignore in cases:
                with self.subTest(
                    mesh_shape=mesh_shape,
                    vocab_size=vocab_size,
                    dtype=dtype,
                    ignore=ignore,
                ):
                    # Sample on CPU and move to the test device: not every
                    # accelerator registers a device-side torch.Generator.
                    generator = torch.Generator().manual_seed(1729 + vocab_size)
                    global_logits = torch.randn(
                        T, vocab_size, dtype=dtype, generator=generator
                    ).to(self.device_type)
                    global_labels = torch.randint(
                        0, vocab_size, (T,), generator=generator
                    ).to(self.device_type)
                    if ignore:
                        global_labels[[1, 17]] = IGNORE_INDEX
                    # Exercise masked logits without allowing 0 * -inf to
                    # contaminate otherwise finite entropy values.
                    global_logits[3, vocab_size // 2 :] = -torch.inf
                    global_labels[3] = 0
                    global_weights = torch.linspace(
                        -1.25, 1.75, T, device=self.device_type
                    )

                    reference_logits = (
                        global_logits.detach().clone().requires_grad_(True)
                    )
                    reference_logits_fp32 = reference_logits.float()
                    reference_logprobs = -F.cross_entropy(
                        reference_logits_fp32,
                        global_labels,
                        reduction="none",
                        ignore_index=IGNORE_INDEX,
                    )
                    with torch.no_grad():
                        reference_probs = torch.softmax(reference_logits_fp32, dim=-1)
                        reference_entropy = -torch.special.xlogy(
                            reference_probs, reference_probs
                        ).sum(dim=-1)
                    (reference_logprobs * global_weights).sum().backward()

                    local_logits = (
                        distribute_tensor(global_logits, mesh, logits_placements)
                        .to_local()
                        .detach()
                        .requires_grad_(True)
                    )
                    local_labels = distribute_tensor(
                        global_labels, mesh, label_placements
                    ).to_local()
                    local_weights = distribute_tensor(
                        global_weights, mesh, label_placements
                    ).to_local()
                    expected_logprobs = distribute_tensor(
                        reference_logprobs.detach(), mesh, label_placements
                    ).to_local()
                    expected_entropy = distribute_tensor(
                        reference_entropy, mesh, label_placements
                    ).to_local()
                    expected_grad = distribute_tensor(
                        reference_logits.grad, mesh, logits_placements
                    ).to_local()

                    logits_type = {tp_group: spmd.S(1)}
                    labels_type = {tp_group: spmd.I}
                    if "dp" in axis_names:
                        dp_group = mesh.get_group("dp")
                        logits_type[dp_group] = spmd.S(0)
                        labels_type[dp_group] = spmd.S(0)
                    spmd.assert_type(local_logits, logits_type)
                    spmd.assert_type(local_labels, labels_type)

                    with set_current_spmd_mesh(mesh):
                        with typecheck(strict_mode="strict"):
                            logprobs, entropy = compute_logprobs(
                                local_logits,
                                local_labels,
                                vocab_parallel_group=tp_group,
                                return_entropy=True,
                                global_vocab_size=vocab_size,
                            )

                    self.assertIs(spmd.get_axis_local_type(logprobs, tp_group), spmd.I)
                    self.assertIs(spmd.get_axis_local_type(entropy, tp_group), spmd.I)
                    self.assertFalse(entropy.requires_grad)
                    torch.testing.assert_close(logprobs, expected_logprobs)
                    torch.testing.assert_close(entropy, expected_entropy)

                    (logprobs * local_weights).sum().backward()
                    torch.testing.assert_close(local_logits.grad, expected_grad)


class _FakeDecoder(nn.Module):
    """Minimal Decoder-like model for testing ChunkedLossWrapper."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.output = nn.Linear(dim, vocab_size, bias=False)
        # Make it look like a Decoder to ChunkedLossWrapper
        self.layers = nn.ModuleDict()
        self.tok_embeddings = None
        self.norm = None

    def forward(self, tokens, skip_lm_head=False):
        if skip_lm_head:
            return tokens  # return hidden states directly
        return self.output(tokens)


class _IdentityAttention:
    inner_attention = None


class _IdentityDecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _IdentityAttention()

    def forward(
        self,
        hidden,
        attention_metadata,
        positions,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        del attention_metadata, positions, padding_mask, aux_loss_denominator
        return hidden


class _AddMTPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _IdentityAttention()

    def forward(
        self,
        mtp_input_embed,
        prev_embed,
        mtp_input_valid_mask,
        attention_metadata,
        positions,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        del attention_metadata, positions, padding_mask, aux_loss_denominator
        return mtp_input_embed + prev_embed * mtp_input_valid_mask.unsqueeze(-1)


class _FakeMTPDecoder(MTPDecoder):
    def __init__(self, *, skip_lm_head: bool, num_mtp_layers: int = 1):
        nn.Module.__init__(self)
        self._skip_lm_head = skip_lm_head
        self.tok_embeddings = nn.Embedding(16, 4)
        self.layers = nn.ModuleDict({"0": _IdentityDecoderBlock()})
        self.norm = nn.Identity()
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.mtp_layers = nn.ModuleList(_AddMTPBlock() for _ in range(num_mtp_layers))

    def _get_attention_metadata(self, positions, **kwargs):
        del positions, kwargs
        return {}


class _WeightedTwoOutputLoss(BaseLoss):
    """Two-output objective used to exercise generic chunked-loss plumbing."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        pass

    auxiliary_weight = 0.25

    def __init__(self, config: Config):
        del config
        self.fn = cross_entropy_loss

    def __call__(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, ...],
        labels: torch.Tensor | tuple[torch.Tensor, ...],
        global_loss_token_counts: torch.Tensor | None = None,
        **loss_inputs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del loss_inputs
        if not isinstance(pred, tuple) or not isinstance(labels, tuple):
            raise ValueError("Expected prediction and labels tuples.")
        loss = self.fn(pred[0], labels[0]) + self.auxiliary_weight * self.fn(
            pred[1], labels[1]
        )
        if global_loss_token_counts is not None:
            loss = loss / global_loss_token_counts
        return loss, {}


class _RecordingFSDPLinear(nn.Linear):
    def __init__(self, *args, events: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.events = events

    def set_reshard_after_forward(self, enabled):
        self.events.append(f"set_reshard_after_forward({enabled})")

    def set_reshard_after_backward(self, enabled):
        self.events.append(f"set_reshard_after_backward({enabled})")

    def set_requires_gradient_sync(self, enabled, *, recurse):
        self.events.append(f"gradient_sync({enabled})")

    def unshard(self):
        self.events.append("unshard")

    def reshard(self):
        self.events.append("reshard")

    def forward(self, input):
        self.events.append("forward")
        return super().forward(input)
