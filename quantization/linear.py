"""Identical static quantization boundaries for PTQ calibration and QAT."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

QUANTIZATION_CONFIG = dict(
    format='QDQ', activation_dtype='uint8', activation_range=[0, 255],
    activation_scheme='per_tensor_affine', weight_dtype='int8', weight_range=[-127, 127],
    weight_scheme='per_output_channel_symmetric', observer='running_minmax',
    boundaries=['linear_input', 'linear_weight', 'linear_output'],
    shared_qkv_input=True, bias='float32', scope='Linear only')


class WeightFakeQuant(torch.autograd.Function):
    """Export symmetric restricted-range weights; exporter folds their Q to INT8."""
    @staticmethod
    def forward(ctx, x, scale, zero):
        ctx.save_for_backward(x, scale)
        return torch.fake_quantize_per_channel_affine(x, scale, zero, 0, -127, 127)

    @staticmethod
    def backward(ctx, grad):
        x, scale = ctx.saved_tensors
        mask = (x >= -127 * scale[:, None]) & (x <= 127 * scale[:, None])
        return grad * mask, None, None

    @staticmethod
    def symbolic(g, x, scale, zero):
        zero = g.op('Cast', zero, to_i=3)  # ONNX INT8
        quantized = g.op('QuantizeLinear', x, scale, zero, axis_i=0)
        return g.op('DequantizeLinear', quantized, scale, zero, axis_i=0)


class StaticQuantizer(nn.Module):
    def __init__(self, channels=None):
        super().__init__()
        self.channels = channels
        if channels is None:
            self.observer = MinMaxObserver(dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                                          quant_min=0, quant_max=255)
        else:
            self.observer = PerChannelMinMaxObserver(dtype=torch.qint8, qscheme=torch.per_channel_symmetric,
                                                    ch_axis=0, quant_min=-127, quant_max=127)
        self.register_buffer('scale', torch.ones(1 if channels is None else channels))
        self.register_buffer('zero_point', torch.zeros(1 if channels is None else channels, dtype=torch.int32))
        self.register_buffer('initialized', torch.tensor(False))
        self.observe = True
        self.fake = False

    def forward(self, x):
        if self.observe:
            with torch.no_grad():
                self.observer(x.detach())
                scale, zero = self.observer.calculate_qparams()
                self.scale.copy_(scale)
                self.zero_point.copy_(zero)
                self.initialized.fill_(True)
        if not self.fake:
            return x
        if not torch.onnx.is_in_onnx_export() and not bool(self.initialized):
            raise ValueError('Quantizer is uninitialized; calibrate on train before inference.')
        if self.channels is not None:
            return WeightFakeQuant.apply(x, self.scale, self.zero_point)
        return torch.fake_quantize_per_tensor_affine(x, self.scale, self.zero_point, 0, 255)


class QuantLinear(nn.Linear):
    def __init__(self, linear):
        super().__init__(linear.in_features, linear.out_features, linear.bias is not None,
                         device=linear.weight.device, dtype=linear.weight.dtype)
        self.weight = nn.Parameter(linear.weight.detach().clone())
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        self.input_quant = StaticQuantizer()
        self.weight_quant = StaticQuantizer(self.out_features)
        self.output_quant = StaticQuantizer()
        self.to(linear.weight.device)

    def forward(self, x):
        return self.output_quant(F.linear(self.input_quant(x), self.weight_quant(self.weight), self.bias))


def quantizers(model):
    return [m for m in model.modules() if isinstance(m, StaticQuantizer)]


def prepare_qat(model):
    model = copy.deepcopy(model)
    def replace(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                setattr(parent, name, QuantLinear(child))
            else:
                replace(child)
    replace(model)
    for block in model.transformer_encoder.blocks:
        attention = block.attention
        attention.k_proj.input_quant = attention.q_proj.input_quant
        attention.v_proj.input_quant = attention.q_proj.input_quant
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)
    return model


def freeze_observers(model):
    for q in quantizers(model):
        if not bool(q.initialized):
            raise ValueError('Cannot freeze an uninitialized observer.')
        q.observe = False
        q.fake = True


@torch.no_grad()
def calibrate(model, batches):
    model.eval()
    for q in quantizers(model):
        q.observe, q.fake = True, False
    count = 0
    device = next(model.parameters()).device
    for signals in batches:
        if isinstance(signals, (tuple, list)):
            signals = signals[0]
        if not torch.isfinite(signals).all():
            raise ValueError('Non-finite calibration input.')
        model(signals.to(device))
        count += len(signals)
    if not count:
        raise ValueError('Calibration dataset is empty.')
    freeze_observers(model)
    return count
