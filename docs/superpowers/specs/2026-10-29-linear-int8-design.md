# Thiết kế PTQ/QAT Linear INT8 và đánh giá Raspberry Pi 4

Trạng thái: đã được người dùng duyệt; triển khai trực tiếp trong repo theo xác nhận của người dùng. Tiến độ và các lựa chọn kỹ thuật được ghi trong `../plans/2026-10-29-linear-int8.md`.

## Mục tiêu và phạm vi

So sánh FP32, static PTQ INT8 và static QAT INT8 trên Raspberry Pi 4 chạy Linux 64-bit, dùng ONNX Runtime CPU. Fine-tune và export thực hiện trên máy huấn luyện. Bộ đánh giá có adapter cho checkpoint PyTorch `.pt` gốc và model `.onnx` do pipeline này tạo; mọi lệnh nhận đường dẫn tường minh đến `beat_split.npz` đúng cấu trúc repo hiện tại.

Quantize các phép Linear: Q/K/V/output projection trong attention, hai Linear FFN mỗi block, ba Dense của head, residual projection và output classifier. CNN, BatchNorm, LayerNorm, positional encoding, softmax, attention score/value matmul, pooling và các phép residual giữ FP32. Bias giữ theo biểu diễn phù hợp backend; không yêu cầu mọi tensor trong Linear đều là INT8.

Không huấn luyện lại từ đầu, chia lại tập hoặc đọc lại WFDB. Không dùng test để calibration, observer update, lựa chọn epoch hoặc điều chỉnh quantization. Không triển khai dynamic PTQ trong giai đoạn này.

## Các phương án đã thảo luận

1. Static PTQ và static QAT: cùng phạm vi Linear và cùng quy ước quantization; thích hợp để so sánh ảnh hưởng của fine-tune QAT. Cần calibration cho PTQ và xử lý export QAT.
2. Dynamic PTQ và static QAT: PTQ đơn giản hơn, nhưng khác cách quantize activation khi inference nên không cô lập được tác động của QAT.

Người dùng đồng ý phương án static PTQ và static QAT, yêu cầu xem thiết kế các file trước khi triển khai. Đích chạy đã xác nhận: Pi 4, hệ điều hành 64-bit.

## Các file sẽ thêm

Các thư mục Python mới có `__init__.py` tối giản, không import sớm torch hay ONNX Runtime.

| File | Trách nhiệm |
| --- | --- |
| `quantization/attention.py` | Attention tương đương MultiheadAttention hiện tại nhưng dùng Q/K/V/out Linear tường minh. Chuyển trọng số gộp từ checkpoint; giữ thứ tự head, scaling và toán học attention. |
| `quantization/linear.py` | Linear có observer/fake quant cho QAT; cấu hình quantization chung PTQ/QAT; đóng băng observer và export thành QDQ. |
| `quantization/checkpoint.py` | Load an toàn checkpoint FP32 theo format_version=1, dựng ECGClassifier từ model_args, chuyển attention trên bản sao, kiểm tra split hash/mode/class order; lưu và load checkpoint QAT với format riêng. |
| `quantization/export.py` | Export FP32 và QAT ONNX; opset 17, batch động, chiều tín hiệu cố định 300 hoặc 400 theo run. Kiểm tra ONNX, chạy đối chiếu output và ghi metadata. Liệt kê Linear node, xác thực phạm vi quantization và weight INT8. |
| `PTQ/quantize.py` | CLI FP32 checkpoint → export baseline ONNX → calibration từ train → static Linear-only INT8 QDQ ONNX. Không cập nhật trọng số. |
| `QAT/finetune.py` | CLI khởi tạo từ FP32, bootstrap observer bằng train, fine-tune fake quant, freeze observer cuối lịch, lưu lịch sử/checkpoint và xuất QAT ONNX. |
| `evaluate_pi/adapters.py` | Interface ModelAdapter và hai implementation TorchAdapter/ONNXAdapter; factory chọn backend. Input NumPy float32 `[B,1,L]`, output logits float32 `[B,5]`; metadata cùng schema. Import backend khi cần. |
| `evaluate_pi/split_io.py` | Load NPZ với allow_pickle=False; kiểm tra shape/dtype/nhãn, partition train-test và sampled indices; kiểm tra hash/mode/class order từ metadata. Không phụ thuộc torch, WFDB hay SciPy. |
| `evaluate_pi/benchmark.py` | Đo latency batch 1 sau warm-up trên input đã nằm trong RAM; ghi từng lượt, mean/median/p95/std, số lượt, thread count và môi trường chạy. |
| `evaluate_pi/complexity.py` | Tính parameter của kiến trúc gốc và số phép tính Conv/Linear/attention theo input shape; tách phần FP32/INT8. Ghi công thức/quy ước và phạm vi bỏ qua trong metadata export. Pi đọc metadata, không cần dựng model torch để đếm. |
| `evaluate_pi/metrics.py` | Metric phân loại và xuất kết quả, tái sử dụng logic được tách từ utils.py; NumPy stable softmax và cross-entropy từ logits. Không phụ thuộc torch. |
| `evaluate_pi/evaluate.py` | CLI đánh giá chung: chọn adapter, load split, đánh giá toàn bộ test_indices theo batch, chạy benchmark riêng và xuất báo cáo. |
| `requirements-quantization.txt` | Dependency máy train/export: requirements gốc, ONNX và ONNX Runtime; pin dải phiên bản đã kiểm tra thực tế khi triển khai. |
| `evaluate_pi/requirements.txt` | Dependency đánh giá ONNX trên Pi 64-bit: NumPy, ONNX Runtime, scikit-learn, matplotlib, tqdm. PyTorch là tùy chọn nếu cần adapter .pt; không bắt cài torch cho ONNX. |
| `tests/test_quantization.py` | Kiểm tra chuyển attention tương đương, quantization coverage, train-only calibration/QAT, export, metadata và CLI trên fixture nhỏ. |
| `tests/test_evaluate_pi.py` | Kiểm tra adapter parity, split validation, metric parity, định nghĩa latency và count, đường chạy ONNX không import torch. |
| `evaluate_pi/README.md` | Hướng dẫn tạo artifact trên máy train, chuyển sang Pi, cài dependency và chạy cùng điều kiện để so sánh. |

Không thêm script riêng cho từng metric hoặc từng loại model; runner và adapter chia sẻ toàn bộ logic đánh giá.

## File hiện tại cần sửa

- `utils.py`: chuyển implementation `compute_metrics` và `save_results` sang `evaluate_pi/metrics.py`, import lại ở utils để các caller hiện tại vẫn dùng nguyên API. Không đổi định nghĩa metric cũ.
- `README.md`: bổ sung đường dẫn hướng dẫn PTQ/QAT/evaluate, lệnh mẫu và ý nghĩa FLOPs khi có INT8.
- Không đổi public API hoặc trọng số của model gốc; train.py và test.py tiếp tục hoạt động như hiện tại. Bản attention tường minh chỉ được dùng trên bản sao cho export/quantization.

## Hợp đồng artifact và adapter

TorchAdapter hỗ trợ checkpoint FP32 hiện có trong repo. Checkpoint QAT chứa fake quant là artifact phục vụ tiếp tục fine-tune/export, không được quảng bá là model INT8 triển khai và không benchmark như INT8. Artifact triển khai INT8 của cả PTQ/QAT là ONNX.

ONNX cần một input `signals` float32 `[B,1,L]` và một output `logits` float32 `[B,5]`. ONNXAdapter phục vụ FP32/PTQ/QAT qua cùng interface và dùng CPUExecutionProvider. Không giả định hỗ trợ tùy ý mọi ONNX ngoài pipeline: thiếu metadata hoặc sai hợp đồng phải báo lỗi rõ.

Factory `--backend auto` nhận dạng theo nội dung hợp lệ, không chỉ suffix: NPZ/PyTorch đều là ZIP container nhưng có thành phần nội bộ khác nhau; `.zip` checkpoint hiện có phải nhận ra được. Trường hợp không xác định được phải báo lỗi và hướng dẫn truyền backend tường minh, không thử unpickle tùy ý.

Metadata nhúng trong ONNX là nguồn chính, đồng thời xuất bản sao `metadata.json`: schema_version, method, source checkpoint SHA-256, split SHA-256, model_args, mode, class_names, input/output contract, quantization config, danh sách Linear dự kiến/đã quantize, parameter counts, complexity và phiên bản thư viện. Checkpoint không có metadata bắt buộc hoặc mismatch không tự suy đoán cấu hình.

`metadata.json` là manifest theo từng artifact; `fp32.onnx` và `ptq_int8.onnx` có metadata nhúng riêng, với method/coverage/complexity dtype đúng từng file. Không dùng coverage INT8 của PTQ làm thông tin FP32.

NPZ và checkpoint hiện trong outputs mang đuôi `.zip` nhưng có cấu trúc tương ứng NPZ/PyTorch. Loader dựa trên format; không tự giải nén, đổi tên hay sửa artifact người dùng.

## Quy tắc PTQ/QAT

Đề xuất kỹ thuật của người triển khai: weight signed INT8 symmetric per-output-channel; activation unsigned INT8 asymmetric per-tensor; QDQ cho ONNX. PTQ/QAT dùng cùng dtype, range và granularity. Cấu hình cuối được kiểm chứng bằng runtime smoke test; nếu backend Pi không hỗ trợ thì báo lại thay vì âm thầm đổi một nhánh.

Hợp đồng quantization: QDQ ở input, weight và output của từng Linear mục tiêu; phần nối giữa các Linear vẫn tuân theo các phép FP32 đã nêu. Q/K/V dùng chung quantization của input attention để tránh ba thang độc lập trên cùng tensor. Activation range [0,255], weight symmetric [-127,127]; MinMax cho PTQ, observer running MinMax cho QAT, thống kê weight per-output-channel. Ghi toàn bộ boundary/observer/range vào config; cả hai export phải kiểm tra đúng boundary thực tế. QAT fake-quant cả activation và weight, không chỉ weight.

PTQ calibration lấy mẫu có seed từ `original_train_indices`, ưu tiên đại diện các lớp, không oversample test; mặc định tối đa 1024 beat. CLI cho đổi số mẫu; metadata ghi số lượng thực và phân bố nhãn. Linear node selection dựa trên export mapping; không chọn tất cả MatMul vì sẽ quantize cả attention score/value ngoài phạm vi.

QAT dùng `train_indices` đã sampling để fine-tune; đề xuất mặc định epochs=10, lr=1e-5, batch_size=32, seed=42, augmentation tắt. CNN, BN statistics và các parameter ngoài Linear được freeze để cô lập ảnh hưởng fine-tune Linear. Dropout vẫn hoạt động trong train; evaluation/export dùng eval. Observer bootstrap bằng calibration train; cập nhật trong phần đầu fine-tune, freeze ở 20% epoch cuối (ít nhất epoch cuối); fake quant giữ bật. Không tạo validation mới; checkpoint chọn epoch cuối, lịch/seed/config được lưu. Đây là lựa chọn thiết kế có thể chỉnh khi duyệt.

QAT có thể dùng CPU/CUDA trên máy train; Pi chỉ inference. Khi export, kiểm tra output giữa fake-quant eval và ONNX Runtime, weight INT8 và coverage Linear. Nếu không đạt kiểm tra phải dừng và không gắn nhãn export thành công. Graph QDQ cần chứng minh chạy được; việc dùng kernel INT8 trên Pi được kiểm tra thêm qua profiling runtime trên Pi, không suy ra tốc độ chỉ từ sự có mặt của QDQ.

## Metric và cách đo

### Chất lượng phân loại

Giữ metric test.py: test loss, accuracy, precision/recall/F1 macro và weighted, per-class precision/recall/F1/support, one-vs-rest accuracy, TP/TN/FP/FN, ROC-AUC OvR từng lớp và macro/weighted, Cohen kappa, confusion matrix. Undefined AUC/kappa ghi null như hiện tại. Đánh giá toàn bộ test_indices theo đúng thứ tự, không augmentation.

### Inference

Latency mặc định: CPU, 1 thread, batch=1, 50 warm-up, 1000 lượt đo. Dùng perf_counter_ns; input được lấy từ test split và chuẩn bị sẵn ngoài đoạn đo; khi số test ít hơn số lượt thì chạy vòng lại. Đo adapter.predict bao gồm chi phí chuyển NumPy/tensor và gọi runtime; không gồm load model, load NPZ, preprocessing, metric hay lưu file. Ghi rõ đây là host-visible inference latency để so sánh có ý nghĩa; thêm benchmark theo batch có thể mở rộng sau.

Ghi mean_ms_per_sample (yêu cầu chính), median/p95/std và samples_per_second quy đổi từ cùng phép đo. Lưu raw latency, CPU/OS/architecture/runtime version, provider, thread settings và seed. Latency trên máy phát triển không đại diện cho Pi; kết quả Pi phải đo trên Pi.

### Parameter, dung lượng, phép tính

- `parameters_total`: số phần tử parameter học được của kiến trúc FP32, không cộng buffer, optimizer, observer hoặc QDQ scale.
- `parameters_quantized`: số phần tử weight Linear chuyển INT8; parameter count không giảm chỉ vì quantize.
- `model_size_bytes`: kích thước artifact triển khai. Checkpoint .pt có optimizer/config phải ghi rõ khác phạm vi ONNX inference-only; so sánh dung lượng FP32/PTQ/QAT chủ yếu dùng ba ONNX.
- `macs_per_sample`: Conv + Linear + hai attention matmul, batch=1; tính theo L=300/400 thực tế.
- `flops_equivalent_per_sample = 2 * macs_per_sample`: quy ước multiply và add là hai phép tính; đây là arithmetic complexity tương đương FP32, không gọi phép INT8 là floating-point execution.
- Tách `fp32_ops_equivalent` và `int8_ops_equivalent` theo Linear coverage. Báo phạm vi chưa đếm: normalization, activation, softmax, pooling, residual, bias và Q/DQ overhead; không mô tả là tổng FLOPs chính xác của mọi operator hoặc số lệnh CPU.

## CLI dự kiến

Chạy từ root repo bằng Python module:

```text
python -m PTQ.quantize --checkpoint_path outputs/run/model.pt --split_path outputs/run/beat_split.npz --output_path outputs/ptq
python -m QAT.finetune --checkpoint_path outputs/run/model.pt --split_path outputs/run/beat_split.npz --output_path outputs/qat --epochs 10 --lr 1e-5 --device cuda

python -m evaluate_pi.evaluate --model_path outputs/run/model.pt --backend torch --split_path outputs/run/beat_split.npz --output_path outputs/eval_fp32_pt
python -m evaluate_pi.evaluate --model_path outputs/ptq/fp32.onnx --split_path outputs/run/beat_split.npz --output_path outputs/eval_fp32
python -m evaluate_pi.evaluate --model_path outputs/ptq/ptq_int8.onnx --split_path outputs/run/beat_split.npz --output_path outputs/eval_ptq
python -m evaluate_pi.evaluate --model_path outputs/qat/qat_int8.onnx --split_path outputs/run/beat_split.npz --output_path outputs/eval_qat
```

Evaluate có `--backend auto|torch|onnx`, `--batch_size` cho metric (mặc định 64), `--threads` (1), `--warmup` (50), `--benchmark_runs` (1000). Benchmark luôn batch=1; batch_size không thay đổi định nghĩa latency. Folder output phải mới để bảo vệ kết quả cũ.

## Output dự kiến

PTQ: `fp32.onnx`, `ptq_int8.onnx`, `metadata.json`, `quantization_report.json`.

QAT: `qat_checkpoint.pt`, `qat_int8.onnx`, `history.csv`, `config.json`, `metadata.json`, `quantization_report.json`. Checkpoint QAT gồm model/observer/optimizer state và epoch, format riêng, không nạp nhầm bằng loader checkpoint FP32 cũ.

Evaluate: các artifact hiện tại (`metrics.json`, `summary_metrics.csv`, `per_label_metrics.csv`, `predictions.csv`, `probabilities.npy`, `predictions.npy`, `confusion_matrix.csv/png`) và `benchmark.json`, `latencies.csv`, `model_info.json`. Summary thêm latency mean/p95, params, size và complexity để dễ đặt ba run cạnh nhau. Không gộp các run thành báo cáo tổng hợp trong giai đoạn này.

## Điều kiện nghiệm thu

1. Attention sau chuyển trọng số khớp FP32 gốc ở L=300/400, batch=1 và batch>1 trong tolerance float32 được ghi rõ; gradient đi qua các projection.
2. FP32 ONNX khớp model PyTorch; PTQ/QAT chạy inference ONNX Runtime CPU hữu hạn, shape đúng.
3. Mọi Linear dự kiến được quantize, weight INT8; Conv và attention score/value không bị quantize. Graph report liệt kê coverage thực, không chỉ dựa trên filename.
4. Calibration/QAT không đọc test rows để cập nhật thống kê/trọng số; malformed split/hash/mode/class order bị từ chối.
5. Metric khớp logic test.py trên cùng logits, kể cả thiếu lớp; adapter không softmax hai lần.
6. Benchmark đúng số lượt/batch=1 và không bao gồm I/O; complexity dùng cùng quy ước cho các phương pháp.
7. Evaluate ONNX chạy trong môi trường không cài torch/WFDB; tests hiện tại tiếp tục pass.
8. Smoke test thật trên Pi 4 là bước kiểm tra đích triển khai; nếu không có quyền truy cập Pi, báo rõ phần đã kiểm tra trên host và phần còn cần chạy trên Pi.

## Cần người dùng duyệt

Duyệt cấu trúc file và luồng trên; các mặc định do người triển khai đề xuất gồm U8 activation/S8 weight, freeze phần ngoài Linear khi QAT, chọn epoch cuối, batch=1/1 thread và định nghĩa complexity có phạm vi rõ. Chưa triển khai và chưa chạy fine-tune toàn bộ checkpoint thật.
