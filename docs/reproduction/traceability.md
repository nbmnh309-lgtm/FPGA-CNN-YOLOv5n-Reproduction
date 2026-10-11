# Bảng truy vết bài báo → hiện vật dự án

Các đường dẫn B–I là **dự kiến, chưa tồn tại**. Không coi file placeholder là triển khai. A chỉ kiểm chứng tooling/tài liệu, không kiểm chứng CNN/RTL. JSON là danh sách máy đọc được; bảng dưới là bản để đọc.

| ID | Nguồn PDF | Yêu cầu | Code/tài liệu | Test dự kiến hoặc hiện có | Kết quả | Trạng thái |
| --- | --- | --- | --- | --- | --- | --- |
| A-ANALYSIS | §1–5; pp.1–25 | Phân tích PDF và kiểm kê | `docs/reproduction/paper_analysis.md` | `tests/python/test_phase_a.py` | reports/phase_a_validation.json | implemented |
| B-FP32 | §4.1 p.18 | YOLOv5n graph/weights/preprocessing | `models/yolov5n/infer.py` | `tests/python/test_yolov5n.py` | Chưa chạy | planned |
| B-EVAL | §4.3.3 Table 3 | VOC accuracy và CUDA benchmark | `benchmarks/fp32.py` | `tests/python/test_benchmark.py` | Chưa chạy | planned |
| C-INT8 | §2.3 Eq.1–4 pp.5–6 | Affine calibration/quantization | `quantization/int8/affine.py` | `tests/python/test_int8.py` | Chưa chạy | planned |
| D-MQDA | §3.2.3–4 pp.14–15; §4.3.3 | Add/Cat domain alignment và ablation | `architecture/mqda/reference.py` | `tests/python/test_mqda.py` | Chưa chạy | planned |
| E-DMP | §3.2.1 p.13; §4.3.1 p.19 Fig.7 | Shared-activation DMP và sign correction | `architecture/dual_multiply_packing/reference.py` | `tests/python/test_dmp.py` | Chưa chạy | planned |
| E-RTL | §3.2.1; §4.3.1 | DMP RTL latency/valid và arithmetic | `rtl/dual_multiply_packing/dmp.sv` | `testbench/unit/tb_dmp.sv` | Chưa chạy | planned |
| F-BUFFER | §2.4; §3.2.1 Fig.4; §4.3.2 | Ping-pong overlap mô hình chu kỳ | `architecture/ping_pong_buffer/cycle_model.py` | `tests/python/test_ping_pong.py` | Chưa chạy | planned |
| F-RTL | §3.2.1; §3.2.6–7 | Ownership/stalls/reset hai bank RTL | `rtl/ping_pong_buffer/buffer.sv` | `testbench/unit/tb_buffer.sv` | Chưa chạy | planned |
| G-CONV33 | §3.2.1 Table 1 Fig.3–7 | Conv33 stride/padding/partial sum | `rtl/common/conv33.sv` | `testbench/unit/tb_conv33.sv` | Chưa chạy | planned |
| G-CONV11 | §3.2.2 Table 1 | Conv11 và optional activation | `rtl/common/conv11.sv` | `testbench/unit/tb_conv11.sv` | Chưa chạy | planned |
| G-ADD | §3.2.3 | Add RTL 16 lanes, pipeline MQDA | `rtl/mqda/add.sv` | `testbench/unit/tb_add.sv` | Chưa chạy | planned |
| G-CAT | §3.2.4 | Cat RTL ordering/domain | `rtl/mqda/cat.sv` | `testbench/unit/tb_cat.sv` | Chưa chạy | planned |
| G-SPPF | §3.2.5 | Ba maxpool 5×5 và concat | `rtl/common/sppf.sv` | `testbench/unit/tb_sppf.sv` | Chưa chạy | planned |
| G-UPSAMPLE | §3.2.6 | Nearest ×2 cả H/W | `rtl/common/upsample.sv` | `testbench/unit/tb_upsample.sv` | Chưa chạy | planned |
| G-FOCUS | §3.2.7 Fig.8 | Parity slice và padded channel layout | `rtl/common/focus.sv` | `testbench/unit/tb_focus.sv` | Chưa chạy | planned |
| H-INTEGRATION | §3.1 Fig.2 | NPU instruction/DDR/DMA behavioral integration | `rtl/top/npu.sv` | `testbench/integration/tb_npu.sv` | Chưa chạy | planned |
| I-RESULTS | §4.2–4.6 Tables 2–6 | Benchmark/resources/power so cùng phạm vi | `benchmarks/summarize.py` | `tests/python/test_results.py` | Chưa chạy | planned |
