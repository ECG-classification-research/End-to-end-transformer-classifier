"""Common logits interface; ONNX deployment does not import torch."""
from abc import ABC, abstractmethod
import json
from pathlib import Path
import zipfile
import numpy as np

METADATA_KEY = 'ecg_metadata'


class ModelAdapter(ABC):
    metadata: dict

    def _input(self, x):
        if (not isinstance(x, np.ndarray) or x.dtype != np.float32 or x.ndim != 3
                or x.shape[1:] != (1, self.metadata['input_length']) or len(x) == 0):
            raise ValueError('Expected float32 signals [B,1,L] matching the model.')
        return x

    @abstractmethod
    def predict(self, signals):
        """Return logits [B,5], synchronously, on CPU."""


class TorchAdapter(ModelAdapter):
    def __init__(self, path, threads=1, **kwargs):
        import torch
        from model import ECGClassifier
        from utils import load_checkpoint
        from quantization.attention import explicit_model
        from .complexity import model_complexity
        torch.set_num_threads(threads)
        payload = load_checkpoint(path)
        config = payload['config']
        if config.get('mode') not in ('fixed', 'variable'):
            raise ValueError('Invalid checkpoint mode.')
        self.model = ECGClassifier(**config['model_args']).eval().requires_grad_(False)
        self.model.load_state_dict(payload['model_state'], strict=True)
        length = 300 if config['mode'] == 'fixed' else 400
        self.metadata = dict(config, schema_version=1, method='fp32', input_length=length,
                             input_name='signals', output_name='logits',
                             complexity=model_complexity(explicit_model(self.model), length))
        self.runtime = dict(backend='torch', version=str(torch.__version__), device='cpu',
                            threads=threads, interop_threads=torch.get_num_interop_threads(),
                            artifact_scope='training checkpoint: may include optimizer and config')

    def predict(self, signals):
        import torch
        self._input(signals)
        with torch.inference_mode():
            return self.model(torch.from_numpy(signals)).numpy()


class ONNXAdapter(ModelAdapter):
    def __init__(self, path, threads=1, optimization='all', profile_dir=None):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimization == 'all'
                                            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL)
        if profile_dir is not None:
            options.enable_profiling = True
            options.profile_file_prefix = str(Path(profile_dir) / 'ort_profile')
        self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
        raw = self.session.get_modelmeta().custom_metadata_map.get(METADATA_KEY)
        if raw is None:
            raise ValueError('ONNX has no ecg_metadata; export it with this repository first.')
        self.metadata = json.loads(raw)
        if (self.metadata.get('schema_version') != 1 or self.metadata.get('method') not in ('fp32', 'ptq', 'qat')
                or self.metadata.get('input_length') not in (300, 400) or not self.metadata.get('complexity')):
            raise ValueError('Invalid ECG ONNX metadata schema.')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].name != 'signals' or outputs[0].name != 'logits'
                or inputs[0].type != 'tensor(float)' or outputs[0].type != 'tensor(float)'
                or len(inputs[0].shape) != 3 or inputs[0].shape[1:] != [1, self.metadata['input_length']]
                or len(outputs[0].shape) != 2 or outputs[0].shape[1] != 5):
            raise ValueError('Expected ONNX signals float32 [B,1,L] -> logits float32 [B,5].')
        self.runtime = dict(backend='onnx', version=ort.__version__, providers=self.session.get_providers(),
                            threads=threads, interop_threads=1, optimization=optimization,
                            artifact_scope='inference-only ONNX', profiling=profile_dir is not None)

    def predict(self, signals):
        self._input(signals)
        return self.session.run(['logits'], {'signals': signals})[0]

    def end_profile(self):
        return self.session.end_profiling() if self.runtime['profiling'] else None


def create_adapter(path, backend='auto', threads=1, optimization='all', profile_dir=None):
    path = Path(path)
    if threads <= 0 or backend not in ('auto', 'torch', 'onnx') or optimization not in ('all', 'disabled'):
        raise ValueError('Invalid backend, threads or optimization.')
    if not path.is_file():
        raise FileNotFoundError(path)
    if backend == 'auto':
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
            if not any(n.endswith('/data.pkl') for n in names):
                raise ValueError('ZIP is not a PyTorch checkpoint; pass the model, not the NPZ split.')
            backend = 'torch'
        else:
            backend = 'onnx'
    if backend == 'torch':
        if profile_dir is not None:
            raise ValueError('--profile is supported by ONNX adapter only.')
        return TorchAdapter(path, threads)
    return ONNXAdapter(path, threads, optimization, profile_dir)
