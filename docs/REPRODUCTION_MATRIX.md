# Ma trận tái hiện thực tế

| Giai đoạn | Artifact chính | Trạng thái có bằng chứng | Thiếu so với paper |
|---|---|---|---|
| A | docs/reproduction/paper_analysis.md | PDF hash khớp, tài liệu/test kế thừa | Artifact tác giả |
| B | benchmarks/local_experiments.py | CUDA FP32,384 mẫu COCO128 | VOC checkpoint/graph |
| C | reproduction/reference.py + QDQ hooks | Affine tests,32 ảnh calibration, kernel CPU native | Full-network native INT8 |
| D | reproduction/reference.py + QDQ fuse | Random/boundary Add/Cat và AP sanity ablation | Q-format/calibration tác giả |
| E | rtl/dual_multiply_packing/dmp.sv | Python exhaustive, XSim vectors,1DSP XC7Z020 synth | XC7Z100/license/timing |
| F | rtl/ping_pong_buffer/buffer.sv | Backpressure TB + bank cycle oracle | DMA thực/hardware latency |
| G | rtl/top/npu.sv |27 operator cases SIMULATED | Parallel/synthesizable full datapath |
| H | MemoryNPU + tb_npu generator | Python sequence,RTL Focus→Upsample | Full YOLO/MIG/XDMA/PCIe |
| I | reports/LOCAL_REPRODUCTION_VI.md | JSON/CSV/plots với provenance | Physical FPS,power,LUT/BRAM,AP paper |

G không được coi hoàn thành physical NPU. H chưa tái hiện whole network RTL. Native CPU INT8 là kernel nhỏ, QDQ toàn mạng vẫn FP32 arithmetic.
