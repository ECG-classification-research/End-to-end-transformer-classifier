"""Architecture MAC estimates; torch is imported only during host profiling."""


def model_complexity(model, length, quantized=False):
    import torch
    from torch import nn
    if length not in (300, 400):
        raise ValueError('Expected input length 300 or 400.')
    conv_macs = linear_macs = weight_count = 0
    handles = []
    def count(module, inputs, output):
        nonlocal conv_macs, linear_macs
        if isinstance(module, nn.Conv1d):
            conv_macs += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        else:
            linear_macs += output.numel() * module.in_features
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
        if isinstance(module, nn.Linear):
            weight_count += module.weight.numel()
    training = model.training
    device = next(model.parameters()).device
    model.eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 1, length, device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    tokens = (length + 3) // 4
    attention_macs = sum(2 * tokens * tokens * block.attention.embed_dim
                         for block in model.transformer_encoder.blocks)
    total = conv_macs + linear_macs + attention_macs
    int8 = 2 * linear_macs if quantized else 0
    return dict(parameters_total=sum(p.numel() for p in model.parameters()),
                parameters_quantized=weight_count if quantized else 0,
                macs_per_sample=total, flops_equivalent_per_sample=2*total,
                fp32_ops_equivalent=2*total-int8, int8_ops_equivalent=int8,
                input_shape=[1, 1, length],
                convention='2 operations/MAC; Conv, Linear, QK and AV only; arithmetic equivalent, not hardware instructions',
                excluded_ops=['bias', 'normalization', 'activation', 'softmax', 'pooling', 'residual', 'Q/DQ'])
