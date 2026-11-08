"""Construct an isolated JIT variant; never modify the pinned image's files."""

from triton.runtime.jit import JITFunction


TARGET = "acc += matrix_x * matrix_w"
REPLACEMENT = "acc += matrix_x.to(tl.float32) * matrix_w.to(tl.float32)"


def fp32_product_kernel(original: JITFunction) -> JITFunction:
    """Use a fresh specialization cache for the one-line precision control."""
    assert original.src.count(TARGET) == 1
    modified = JITFunction(
        original.fn, do_not_specialize_on_alignment=["num_cache_lines"]
    )
    modified._unsafe_update_src(original.src.replace(TARGET, REPLACEMENT))
    return modified
