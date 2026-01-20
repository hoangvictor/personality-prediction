# Hướng dẫn Fine-tuning với Freeze Layers (Partial Fine-tuning)

Tài liệu này hướng dẫn cách sử dụng script `finetune_models/finetune_partial.py` để fine-tune mô hình BERT cho bài toán dự đoán tính cách, đồng thời đóng băng (freeze) các lớp đầu tiên để tối ưu hóa hiệu suất và tài nguyên.

## 1. Tại sao nên Freeze 8-9 lớp đầu? (Partial Fine-tuning)

Mô hình BERT Base có 12 lớp encoder (layers). Khi áp dụng vào bài toán cụ thể (downstream task) với dữ liệu hạn chế, việc fine-tune toàn bộ 12 lớp (Full Fine-tuning) có thể gặp các vấn đề:
- **Overfitting**: Dữ liệu ít mà mô hình quá lớn.
- **Catastrophic Forgetting**: Quên kiến thức ngôn ngữ tổng quát đã học từ trước.
- **Tốn tài nguyên**: Cần nhiều GPU memory và thời gian tính toán.

**Giải pháp**: Chỉ fine-tune các lớp cuối (Higher layers) và đóng băng các lớp đầu (Lower layers).
- **Lower layers (0-8)**: Học các đặc trưng ngôn ngữ cơ bản (cú pháp, từ loại, cấu trúc câu). Các đặc trưng này mang tính tổng quát (general) và có thể tái sử dụng tốt.
- **Higher layers (9-11)**: Học các đặc trưng ngữ nghĩa trừu tượng và đặc thù cho bài toán (semantic, task-specific).

-> Việc freeze 8-9 lớp đầu và chỉ train 3-4 lớp cuối (cùng với MLP head) là một chiến lược cân bằng giữa hiệu suất (accuracy) và chi phí tính toán (efficiency).

## 2. Cách sử dụng `finetune_partial.py`

Script này hỗ trợ:
- Tùy chỉnh số lượng lớp cần freeze (`-n_freeze`).
- Chọn kiểu mô hình: Multi-Head (dự đoán chung) hoặc Single-Head (dự đoán riêng lẻ từng trait).
- Tự động lưu biểu đồ huấn luyện (Train/Val Loss, Accuracy).

### Các tham số chính:
- `-dataset`: Tên dataset (ví dụ: `essays`, `kaggle`).
- `-embed`: Tên mô hình (ví dụ: `bert-base`, `roberta-base`).
- `-n_freeze`: Số lượng lớp encoder muốn đóng băng (Default: 8).
- `-head_type`:
    - `multi`: Train 1 mô hình chung, output ra N traits (nhanh, chia sẻ kiến thức giữa các traits).
    - `single`: Train N mô hình riêng biệt cho từng trait (có thể chính xác hơn nhưng lâu hơn).
- `-epochs`, `-batch_size`, `-lr`: Tham số huấn luyện thông thường.

### Ví dụ chạy lệnh:

**Chạy với BERT-Base, freeze 8 lớp đầu, chế độ Multi-Head (khuyên dùng):**
```bash
python finetune_models/finetune_partial.py \
    -dataset essays \
    -embed bert-base \
    -n_freeze 8 \
    -head_type multi \
    -epochs 10 \
    -batch_size 16
```

**Chạy với RoBERTa-Base, freeze 9 lớp đầu, chế độ Single-Head:**
```bash
python finetune_models/finetune_partial.py \
    -dataset essays \
    -embed roberta-base \
    -n_freeze 9 \
    -head_type single \
    -epochs 10
```

## 3. Theo dõi Training Logs và Visualization

Script sẽ tự động:
1.  **In log ra màn hình**: Hiện Loss và Accuracy sau mỗi Epoch.
2.  **Lưu biểu đồ**:
    - Với `head_type multi`: Lưu file `log_foldX_multihead.png` (Biểu đồ Loss và Avg Accuracy).
    - Với `head_type single`: Lưu file `log_{TRAIT}_foldX_single.png` cho từng trait.
    
    *Bạn có thể xem các file ảnh .png này để đánh giá liệu mô hình có đang học tốt (Loss giảm, Acc tăng) hay bị Overfitting (Val Loss tăng).*

3.  **Lưu kết quả cuối cùng**:
    - File CSV: `results_freeze{N}_{TYPE}.csv` chứ độ chính xác từng fold cho từng trait.

### Code mẫu visualization (đã tích hợp trong script)
Phần visualize được thực hiện bởi hàm `plot_history` sử dụng `matplotlib`:

```python
def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    # ... code plotting ...
    plt.savefig(save_path)
```
