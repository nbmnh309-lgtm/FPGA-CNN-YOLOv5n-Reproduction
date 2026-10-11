# FPGA-CNN-YOLOv5n-Reproduction

Dự án học và tái hiện từng bước bài báo **An Energy-Efficient FPGA-Based CNN Accelerator with Dual-Multiply Packing and Ping-Pong Buffering for Real-Time Object Detection**, Wenrui Wang và cộng sự, Electronics 2026, 15, 2442. DOI: https://doi.org/10.3390/electronics15112442

## Trạng thái: Giai đoạn 1 — Khởi tạo

Repository hiện có bộ khung và tài liệu; chưa có mô hình YOLOv5n, thuật toán lượng tử, RTL hoặc testbench hoạt động. Chưa tải dataset/trọng số, cài thư viện lớn, huấn luyện hay chạy mô phỏng. Các số liệu trong bài báo là mục tiêu tham khảo, chưa phải kết quả tái hiện.

## Kiến trúc cần nghiên cứu

- **YOLOv5n**: mô hình nhận diện đối tượng; Python làm mô hình tham chiếu trước khi triển khai RTL.
- **INT8 Quantization**: ánh xạ giá trị thực sang số nguyên 8 bit; cần xác định scale, zero-point, quy tắc làm tròn, bão hòa và độ rộng tích lũy.
- **Dual-Multiply Packing (DMP)**: đóng gói hai phép nhân INT8 vào DSP48E1; phải kiểm chứng dấu, tách kết quả và hiệu chỉnh số học.
- **Ping-Pong Buffer**: hai vùng nhớ luân phiên để chồng lấp nạp dữ liệu và tính toán; cần kiểm chứng đổi bank và tránh đọc/ghi xung đột.
- **MQDA (Multi-Quantization Domain Alignment)**: căn chỉnh các miền lượng tử khác nhau trước khi hợp nhất tại Add/Cat, rồi lượng tử lại đầu ra.

Bài báo dùng Zynq-7100/XC7Z100, DSP48E1 và tần số 200 MHz; hậu xử lý chạy trên host. Mô phỏng RTL bằng công cụ mở không xác nhận được timing, công suất hay ánh xạ DSP thực tế. Vivado và phần cứng phù hợp sẽ cần ở giai đoạn triển khai. Chưa chốt phiên bản YOLOv5: bài báo mô tả Focus và ReLU, nên cần đối chiếu đồ thị và activation trước khi chọn commit upstream, tránh mặc định phiên bản mới nhất.

## Cấu trúc thư mục

```text
configs/                            Cấu hình mô hình, lượng tử và phần cứng
models/yolov5n/                      Mô hình tham chiếu và công cụ xuất mô hình
quantization/int8/                   Hiệu chuẩn và mô hình số học INT8
architecture/dual_multiply_packing/  Mô hình tham chiếu DMP bằng Python
architecture/ping_pong_buffer/       Mô hình dữ liệu và điều khiển hai bank
architecture/mqda/                   Mô hình căn chỉnh miền lượng tử
rtl/common/                         Khối RTL dùng chung
rtl/dual_multiply_packing/           RTL đóng gói phép nhân
rtl/ping_pong_buffer/                RTL bộ đệm
rtl/mqda/                           RTL căn chỉnh miền lượng tử
rtl/top/                            Tích hợp accelerator
testbench/unit/                     Testbench từng khối
testbench/integration/              Testbench tích hợp
tests/python/                      Kiểm thử mô hình tham chiếu
benchmarks/                         Kịch bản đo và so sánh
reports/                            Báo cáo, kết quả đã chọn để lưu Git
docs/                               Ghi chú kiến trúc và hướng dẫn
scripts/                            Công cụ hỗ trợ phát triển
data/                               Dataset cục bộ (bỏ qua trong Git)
weights/                            Trọng số cục bộ (bỏ qua trong Git)
outputs/                            Kết quả sinh ra (bỏ qua trong Git)
```

`.gitkeep` giữ lại các thư mục rỗng trong Git. Waveform/build/log không được commit; báo cáo tổng hợp có thể lưu ở `reports/`.

## Bắt đầu sử dụng

Trong cloud, dùng checkout hiện có; không cần tạo Git worktree mới. Chạy từ thư mục gốc repository:

```bash
cd /workspace/FPGA-CNN-YOLOv5n-Reproduction
python3 scripts/check_environment.py
git status --short
```

Lệnh kiểm kê chỉ đọc phiên bản và sự hiện diện của công cụ, không chạy CNN hay mô phỏng. Xem `reports/environment_phase1.md` cho kết quả tại thời điểm khởi tạo.

Ở giai đoạn tiếp theo, khi cần chạy mô hình số học Python, tạo môi trường riêng rồi cài phụ thuộc nhẹ:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Các lệnh cài đặt trên **chưa được chạy ở Giai đoạn 1**. `requirements.txt` dùng khoảng phiên bản, chưa phải lockfile. Khi chọn phiên bản YOLOv5, bổ sung bộ phụ thuộc tương thích và chốt phiên bản đã kiểm chứng. Cài PyTorch/torchvision theo hướng dẫn chính thức cho CPU hoặc CUDA; chưa cần GPU để bắt đầu kiểm chứng số học.

Trước giai đoạn RTL, cần cài Icarus Verilog (kèm `vvp`) và Verilator. Trên Debian/Ubuntu, quản trị viên có thể dùng `apt-get install iverilog verilator`. Đây là hướng dẫn cho bước sau; Giai đoạn 1 chưa cài công cụ hệ thống. Make và trình biên dịch C++ hỗ trợ quy trình Verilator. CMake/Yosys tùy quy trình build/tổng hợp, chưa bắt buộc cho bước khởi tạo.

## Lộ trình dự kiến

1. Khởi tạo cấu trúc, tài liệu và kiểm kê môi trường — giai đoạn hiện tại.
2. Đối chiếu kiến trúc bài báo, chọn commit YOLOv5n và xây dựng mô hình tham chiếu.
3. Kiểm chứng INT8 và MQDA bằng Python trên dữ liệu nhỏ.
4. Kiểm chứng DMP và ping-pong độc lập, rồi viết RTL/testbench.
5. Tích hợp và so sánh RTL với mô hình tham chiếu.
6. Benchmark và báo cáo; triển khai FPGA khi đủ công cụ/phần cứng.

Không bắt đầu bước tiếp theo trước khi người dùng kiểm tra bộ khung. Chưa có test thực thi: thư mục kiểm thử là chỗ dành sẵn, không phải bằng chứng thiết kế đã đúng.
