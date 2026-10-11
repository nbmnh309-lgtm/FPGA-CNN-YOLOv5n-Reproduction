# Kết quả Giai đoạn A

Ngày: 11/10/2026. Base: `origin/main` commit `e5abbd9` (merge Giai đoạn 1); branch `phase-a-paper-analysis`. Đây là báo cáo thực thi cloud, không phải báo cáo benchmark FPGA/GPU.

## Công việc hoàn thành

- Kiểm tra toàn bộ danh sách file Git, README, requirements, ignore rules, bộ khung, lịch sử và remote. Checkout sạch trước khi sửa; không tạo lại dự án.
- Đọc 25 trang PDF, kiểm tra Hình 2 và 7 trực tiếp; lưu checksum và phân tích công thức, architecture, 7 operators, memory map, table results và thiếu hụt.
- Tạo 18 yêu cầu truy vết A–I; B–I chỉ `planned`, không có evidence. Tách 18 chỉ số trích dẫn khỏi kết quả thực hiện.
- Viết validator, kiểm thử dữ liệu sai, collector Python và wrapper Windows. Không cần cài phụ thuộc cho A.

## Bằng chứng kiểm tra đã chạy

| Lệnh/check | Kết quả thực tế | Phạm vi |
| --- | --- | --- |
| `python3 scripts/check_environment.py` | Chạy thành công, log `phase_a_environment.txt` | Tool inventory; thiếu công cụ được báo, không phải readiness CNN |
| `python3 -m unittest discover -s tests/python -p 'test_phase_a.py' -v` | **10 tests pass**, 0 fail; log `phase_a_tests.txt` | Duplicate ID, phase/type, path escape, missing artifact, provenance, file hash/missing input |
| `python3 scripts/validate_phase_a.py --output reports/phase_a_validation.json` | **18 trace requirements, 18 paper metrics pass** | Validation tooling/documentation only |
| `python3 scripts/collect_reproduction_manifest.py --output outputs/cloud_phase_a_manifest.json` | Chạy thành công; optional CUDA không kiểm tra | Python/Git/package metadata cloud; output bị ignore |
| Collector CLI với YAML tạm và upstream checkout | Hash đúng; SHA đủ 40 ký tự; thiếu weights trả exit khác 0 và không ghi manifest kết quả | Không load model hoặc gọi GPU |
| `git diff --check` | Pass | Whitespace/diff integrity |

Không chạy: PowerShell trên Windows, CUDA, accuracy VOC, INT8, RTL, synthesis, Vivado, XDMA hoặc board FPGA. Không có dataset/weights và chưa cài PyTorch/Icarus/Verilator/pytest. Tests A dùng unittest có sẵn; không chuyển sang test CNN để giả định thành công.

## Phát hiện cần chú ý

1. Đồ thị YOLOv5n và checkpoint VOC chưa định danh; Focus/ReLU cần đối chiếu với upstream thực tế.
2. DMP ký hiệu unsigned chưa đủ xử lý signed weights; Add clamp q<Z khác oracle affine.
3. Metric GOPS/MAC và latency denominator có điểm chưa nhất quán; power 8.536 W là Vivado estimate. Chi tiết trong `docs/reproduction/paper_analysis.md`.
4. mAP chưa ghi rõ IoU/AP/split; chưa thể đặt mục tiêu bit-exact hoặc exact mAP reproduction.

## Việc người dùng cần làm

Chạy `scripts/collect_windows_manifest.ps1` theo `docs/reproduction/windows_handoff.md`, gửi `outputs/windows_manifest.json`, thông tin upstream/weights/YAML và dataset split đang dùng. Collector không cài gói/benchmark. Kết quả hai ảnh trước đây vẫn có nhãn user-reported smoke, không phải benchmark chính thức.

Chỉ tiếp tục B sau khi xác định đầu vào và thống nhất protocol; C–I theo gates trong plan. Giai đoạn A không tạo setup script cloud vì chưa có cài đặt/services cần tái tạo. Không có thao tác publication môi trường hoặc chạy GPU Windows.
