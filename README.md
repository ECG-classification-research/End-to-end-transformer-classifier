# ECG CNN-Transformer end-to-end

Project độc lập: CNN Encoder nối trực tiếp Transformer Encoder và head phân loại
5 lớp AAMI. Toàn bộ mô hình được khởi tạo và huấn luyện cùng lúc bằng một
`CrossEntropyLoss`. Source trong folder này không import các module của pipeline cũ.

## Cài đặt

Python 3.10 trở lên. Chạy từ folder `ecg_end_to_end`:

```powershell
cd ecg_end_to_end
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Chọn bản PyTorch CPU/CUDA phù hợp máy. `--device auto` dùng CUDA nếu có.
Ở máy dùng Python Launcher, có thể thay `python` bằng `py`.

## Cấu trúc

```text
data/preprocessing.py          Đọc WFDB, lọc, chuẩn hóa, cắt nhịp
data/augmentation.py           Gaussian noise, scale, time shift
data/dataset.py                BeatDataset, trả signal [1,L] và label
data/pipeline.py               AAMI mapping, split, sampling, lưu/đọc split
model/cnn_encoder.py           CNN Encoder, tùy chọn more_cnn
model/transformer_encoder.py   Positional encoding và Transformer Encoder
model/classifier.py            Head và ECGClassifier hoàn chỉnh
utils.py                       Seed, device, checkpoint, hash, metrics và export
train.py                       Chuẩn bị dữ liệu và huấn luyện toàn mạng
test.py                        Đánh giá checkpoint trên segment test đã lưu
tests/test_system.py           Kiểm tra dữ liệu, gradient, checkpoint và CLI
```

## Dữ liệu và preprocessing

`--data_path` trỏ trực tiếp đến thư mục chứa `100.hea`, `100.dat`, `100.atr`, ...
Không cần export CSV/JSON. Khi bỏ `--records`, loader tìm các `.hea` ngay trong
thư mục này và kiểm tra đủ bộ ba file cho từng record.

Giữ hành vi của pipeline tham khảo:

1. Đọc kênh đầu tiên bằng WFDB; dùng `record.fs` làm tần số lấy mẫu.
2. Lọc toàn record bằng Butterworth bậc 4, 0.5–40 Hz, `sosfiltfilt`.
3. Fit min–max riêng trên toàn record, chuyển về `[-1,1]`. Tín hiệu hằng trả về -1.
4. Cắt quanh vị trí annotation; bỏ cửa sổ vượt biên record.
5. Ánh xạ AAMI, bỏ annotation không thuộc mapping.
6. Chia stratified 80% train / 20% test theo beat, seed mặc định 42.
7. Sampling chỉ train, sau split; augmentation chỉ train khi lấy mẫu.

| Mode | Cửa sổ | Tensor segment | Token CNN |
|---|---|---|---|
| fixed | `signal[R-200:R+100]` | `[1,300]` | `[64,75]` |
| variable | `signal[R-175:R+175+delta]` | `[1,400]` | `[64,100]` |

`delta` nguyên từ -50 đến +50. Variable chép 300–400 mẫu vào đầu mảng 400 số 0;
random một lần lúc load, trước split. Test đọc lại đúng segment đã lưu.

| ID | Lớp | Annotation |
|---|---|---|
| 0 | Normal | N, L, R, e, j |
| 1 | SVEB | A, a, J, S |
| 2 | VEB | V, E |
| 3 | Fusion | F |
| 4 | Unknown | /, f, Q |

Sampling `none` giữ nguyên; `over` bằng số mẫu lớp lớn nhất; `under` bằng lớp nhỏ
nhất; `hybrid` lấy số mẫu lớp lớn nhất ngoài Normal làm mốc để tăng/giảm các lớp
hiện có. Nếu chỉ có Normal thì hybrid giữ nguyên. Oversampling lặp indices.

Augmentation chạy theo thứ tự: Gaussian noise std=0.005 → scale `[0.95,1.05]` →
time shift nguyên `[-5,5]` và zero-fill. Không clip lại biên độ; augmentation cũng
tác động padding. Các mẫu lặp do oversampling được augmentation độc lập khi lấy mẫu.

Split vẫn theo beat; cùng record có thể góp nhịp cho cả train và test. Scaler fit
trên toàn record trước split. Kết quả phản ánh đánh giá beat-wise, chưa phải
đánh giá trên bệnh nhân mới. Variable giữ cách xử lý hiện tại, không có padding mask.

## Kiến trúc và loss

```text
ECG [B,1,L]
 → SAME padding → Conv1d 1→128, k=5, s=2 → ReLU → BN → Dropout
 → SAME padding → Conv1d 128→64, k=5, s=2 → ReLU → BN → Dropout
 → [more_cnn: 2 × (Conv1d 64→64, k=3, s=1, p=1 → ReLU → BN → Dropout)]
 → CNN features [B,64,T]
 → transpose [B,T,64] → sinusoidal positional encoding
 → N Transformer Encoder blocks
 → average pooling [B,64]
 → Dense 64→512→256→128 với activation/norm/dropout và residual projection
 → Linear 128→5
 → logits [B,5]
```

`L=300,T=75` hoặc `L=400,T=100`. Transformer dùng 4 head, FFN `64→256→64`,
LayerNorm eps=1e-6; mỗi block giữ residual của attention, FFN và residual ngoài
như source tham khảo. Head cộng projection `64→128` của pooled vector vào
đầu ra Dense 128 trước LayerNorm cuối. Dropout=0.2, BN eps=1e-3/momentum=0.01;
attention dropout nội bộ=0.0 như source gốc.

`ECGClassifier.forward()` trả logits. Loss là `CrossEntropyLoss(logits, labels)`;
Adam cập nhật đồng thời CNN, Transformer và head. Softmax chỉ dùng lúc tính xác
suất. Mô hình dùng nhánh Encoder CNN trước Dense của Autoencoder tham khảo.

## Train

```powershell
python train.py `
  --data_path "D:\path\to\mit-dataset" `
  --output_path "outputs\fixed_run01" `
  --mode fixed `
  --epochs 30 `
  --batch_size 32 `
  --lr 0.0001 `
  --seed 42 `
  --device auto
```

Variable, hybrid, augmentation và kiến trúc sâu hơn:

```powershell
python train.py `
  --data_path "D:\path\to\mit-dataset" `
  --output_path "outputs\variable_run01" `
  --mode variable `
  --sampling hybrid `
  --augment `
  --more_cnn `
  --transformer_blocks 4 `
  --epochs 50 `
  --batch_size 32 `
  --lr 0.0001 `
  --seed 42 `
  --device auto
```

| Argument | Mặc định | Ý nghĩa |
|---|---|---|
| `--data_path` | Bắt buộc | Thư mục WFDB |
| `--output_path` | Bắt buộc | Folder run mới, chưa tồn tại |
| `--records` | Tất cả `.hea` | Ví dụ `--records 100 101 103` |
| `--mode` | fixed | fixed / variable |
| `--sampling` | none | none / over / under / hybrid |
| `--augment` | Tắt | Bật augmentation online cho train |
| `--more_cnn` | Tắt | Thêm hai lớp CNN, giữ 64 kênh và độ dài |
| `--transformer_blocks` | 2 | Số Transformer Encoder block |
| `--epochs` | 30 | Số epoch |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.0001 | Learning rate Adam |
| `--seed` | 42 | Seed khởi tạo, split, segmentation, sampling, augmentation |
| `--device` | auto | auto / cpu / cuda |

Train chỉ dùng train batches để tối ưu. Checkpoint là epoch cuối; hiện chưa có
validation, early stopping, chọn best checkpoint hoặc resume. Metric train là
trung bình ghi nhận trong các batch khi model ở chế độ train, có dropout và
augmentation nếu bật. Đặt seed không đảm bảo bitwise reproducibility trên mọi CUDA.

## Test

```powershell
python test.py `
  --checkpoint_path "outputs\fixed_run01\model.pt" `
  --batch_size 64 `
  --device auto
```

| Argument | Mặc định | Ý nghĩa |
|---|---|---|
| `--checkpoint_path` | Bắt buộc | model.pt của run |
| `--split_path` | beat_split.npz cạnh checkpoint | Đường dẫn khác đến đúng bản sao split |
| `--output_path` | test_results cạnh checkpoint | Folder kết quả mới |
| `--batch_size` | 64 | Batch size suy luận |
| `--device` | auto | auto / cpu / cuda |

Test kiểm tra SHA-256 toàn file split so với checkpoint trước khi đánh giá.
Có thể di chuyển cả folder run mà không đổi CLI mặc định; khi tách split sang
nơi khác, dùng `--split_path` trỏ đến đúng file đã lưu. Không xử lý lại raw data,
random lại cửa sổ variable, sampling hoặc augmentation trong test.

## Output

```text
outputs/fixed_run01/
├── config.json
├── beat_split.npz
├── history.csv
├── train_metrics.json
├── model.pt
└── test_results/
    ├── metrics.json
    ├── summary_metrics.csv
    ├── per_label_metrics.csv
    ├── predictions.csv
    ├── probabilities.npy
    ├── predictions.npy
    ├── confusion_matrix.csv
    └── confusion_matrix.png
```

`beat_split.npz` lưu segments, labels, raw_labels, original_train_indices,
train_indices sau sampling, test_indices, kept_indices, records và class_names.
Segment chỉ lưu một lần; augmentation online không được ghi vào file. Các indices
trỏ tới hàng segment, không phải vị trí R-peak trên raw record. Chưa lưu record ID
và R-peak riêng cho từng beat.

`model.pt` có format_version, model_state, optimizer_state, epoch, config và
train_metrics. Config lưu model_args, preprocessing, class order, sampling counts
và SHA-256 split. Giữ checkpoint cùng split để có thể đánh giá lại.

Metric gồm test CrossEntropy, accuracy, precision/recall/F1 theo lớp và macro/weighted,
ROC-AUC OvR, Cohen's kappa. P/R/F1 tổng hợp trong summary là weighted; F1 macro
được ghi riêng. Accuracy từng lớp là `(TP+TN)/N` theo one-vs-rest. AUC không xác
định được ghi `null` khi thiếu positive hoặc negative của lớp đó.

Folder output đã tồn tại sẽ bị từ chối để giữ thí nghiệm trước. Lần chạy bị ngắt
có thể để lại output chưa hoàn chỉnh; chọn folder mới cho lần chạy lại.

## Kiểm tra

```powershell
python train.py --help
python test.py --help
python tests/test_system.py
```

Tests dùng tín hiệu WFDB tổng hợp để kiểm tra cơ chế và luồng CLI, gồm cả fixed
và variable. Các metric từ dữ liệu tổng hợp chỉ phục vụ kiểm tra phần mềm.
