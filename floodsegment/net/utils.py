def compute_padding(kernel_size: int, dilation: int) -> int:
    return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
