# SPDX-License-Identifier: Apache-2.0
"""Bound captured mask inputs in NPU backward sparse-mask materialization.

Remove when torch_npu bounds the compact backward template's mask indices
before evaluating captured tensor reads, as its forward template already does.
"""

import hashlib
from functools import wraps

from torchtitan_ascend.patches import patch


@patch(target="torch_npu._inductor.select_algorithm:NPUTritonTemplate.generate", deferred=True)
def bounded_backward_mask(original):
    @wraps(original)
    def generate(self, *args, **kwargs):
        if self.name == "flex_attention_bwd_mask_compact" and not getattr(
            self, "_torchtitan_ascend_bounded_mask", False
        ):
            from torch_npu._inductor.kernel.flexattention_template import compute_bwd_sparse_mask_kernel_compact

            source = compute_bwd_sparse_mask_kernel_compact
            bounded_source = source.replace(
                "m = offs_m[:, None]", "m = tl.minimum(offs_m[:, None], q_len - 1)"
            ).replace("n = offs_n[None, :]", "n = tl.minimum(offs_n[None, :], kv_len - 1)")
            if bounded_source != source:
                # The output mask alone cannot protect captured reads such as
                # document_id[m]. Bound user-mask inputs before those reads;
                # the template still masks stores with the original offsets.
                self.template = self._template_from_string(bounded_source)
                self._hash = hashlib.sha256(bounded_source.encode("utf-8")).hexdigest()
                self._generated_code_cache.cache_clear()
            self._torchtitan_ascend_bounded_mask = True
        return original(self, *args, **kwargs)

    return generate
