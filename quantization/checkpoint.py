"""Source checkpoint validation and portable provenance metadata."""
import platform
import torch
from model import ECGClassifier
from utils import load_checkpoint
from evaluate_pi.split_io import load_split, validate_metadata, sha256
from evaluate_pi.complexity import model_complexity
from .attention import explicit_model


def load_source(checkpoint_path, split_path):
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint['config']
    split = load_split(split_path)
    validate_metadata(split, split_path, config)
    model = ECGClassifier(**config['model_args'])
    model.load_state_dict(checkpoint['model_state'], strict=True)
    return explicit_model(model).eval(), config, split


def build_metadata(model, config, checkpoint_path, split_path, method):
    import onnx
    import onnxruntime
    from .linear import QUANTIZATION_CONFIG, quantizers
    length = 300 if config['mode'] == 'fixed' else 400
    states = [(q, q.observe, q.fake) for q in quantizers(model)]
    for q, _, _ in states:
        q.observe = q.fake = False
    try:
        complexity = model_complexity(model, length, method != 'fp32')
    finally:
        for q, observe, fake in states:
            q.observe, q.fake = observe, fake
    return dict(schema_version=1, method=method, source_checkpoint_sha256=sha256(checkpoint_path),
                split_sha256=sha256(split_path), model_args=config['model_args'], mode=config['mode'],
                class_names=config['class_names'], input_name='signals', output_name='logits',
                input_length=length, complexity=complexity,
                quantization=None if method == 'fp32' else QUANTIZATION_CONFIG,
                versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                              onnx=onnx.__version__, onnxruntime=onnxruntime.__version__))
