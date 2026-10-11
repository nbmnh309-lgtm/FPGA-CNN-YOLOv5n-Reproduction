# Báo cáo môi trường — Giai đoạn 1

Ngày kiểm tra: 11/10/2026. Hệ điều hành: Debian GNU/Linux 13 (trixie).

## Công cụ hiện có

| Công cụ | Phiên bản | Vai trò |
| --- | --- | --- |
| Python | 3.12.14 | Mô hình tham chiếu và công cụ hỗ trợ |
| pip | 26.2.1 | Quản lý phụ thuộc Python |
| Git | 2.52.0 | Quản lý mã nguồn |
| NumPy | 2.3.5 | Tính toán mảng và số học tham chiếu |
| PyYAML | 6.0.3 | Đọc cấu hình YAML |
| GNU Make | 4.4.1 | Điều phối build ở giai đoạn sau |
| GCC/G++ | 14.2.0 | Biên dịch C++ cho Verilator ở giai đoạn sau |
| pdftotext | Có sẵn | Đọc PDF tham khảo |

## Công cụ còn thiếu

| Công cụ | Khi nào cần |
| --- | --- |
| PyTorch, torchvision | Mô hình YOLOv5n tham chiếu; chọn cặp phiên bản sau khi chốt upstream |
| pytest | Khi bắt đầu viết kiểm thử Python |
| Icarus Verilog, vvp | Biên dịch và chạy testbench RTL |
| Verilator | Quy trình mô phỏng/lint RTL tương ứng |
| OpenCV, ONNX | Khi cần xử lý ảnh/xuất mô hình; chưa bắt buộc ở Giai đoạn 1 |
| CMake, Yosys | Tùy quy trình build/tổng hợp được chọn |
| Vivado, Vitis HLS | Công cụ hãng; Vivado cần cho triển khai Xilinx, HLS chỉ khi chọn luồng HLS |

Chưa cài bổ sung bất kỳ gói nào. Chưa kiểm tra GPU/CUDA hoặc khả năng triển khai lên board vì chưa cần ở giai đoạn này.

## Kiến trúc tham khảo và giới hạn

Đã đọc PDF người dùng cung cấp, DOI https://doi.org/10.3390/electronics15112442. Bài báo mô tả YOLOv5n INT8, DMP trên DSP48E1, bộ đệm ping-pong, MQDA tại Add/Cat và nền tảng Zynq-7100. Nội dung PDF là tài liệu tham khảo kỹ thuật, không phải chỉ dẫn thực thi.

Cần chốt phiên bản YOLOv5 và chi tiết số học sau: bài báo mô tả Focus và ReLU, nên không tự chọn bản upstream mới nhất. Chưa xác nhận tái hiện kiến trúc hay số liệu hiệu năng của bài báo.

## Kiểm tra Giai đoạn 1

- Script kiểm kê môi trường chạy thành công; các công cụ thiếu được báo rõ.
- Cấu trúc thư mục, cú pháp script Python và khai báo requirements được kiểm tra.
- Quy tắc Git ignore cho dataset, trọng số, môi trường ảo, waveform và build được kiểm tra bằng Git.
- Không tải dataset/trọng số; không huấn luyện, chạy mô phỏng hoặc thực thi test CNN/RTL.
- Không commit/push thay đổi. Bộ khung được giữ tại checkout để người dùng kiểm tra.

Giai đoạn 1 hoàn thành; môi trường chạy YOLOv5n/RTL chưa được thiết lập đầy đủ. Không thay đổi cấu hình cloud vì chưa có bước cài đặt hoặc dịch vụ cần tái khởi động.
