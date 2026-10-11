# Kế hoạch và điều kiện chuyển giai đoạn

Giai đoạn A không cài PyTorch, không tải YOLOv5, dataset hoặc weights, không chạy suy luận hay RTL. Dùng checkout cloud hiện có, không tạo worktree. Mỗi PR tiếp theo phải ghi rõ test chạy thật, đầu ra, seed, công cụ và giới hạn.

## Phân công theo khả năng thực tế

| Nơi chạy | Có thể thực hiện | Chưa thể kết luận |
| --- | --- | --- |
| Codex Cloud hiện tại | Đọc PDF, Git, Python chuẩn, NumPy, manifest/traceability validation | GPU Windows, CNN inference, RTL khi chưa cài simulator |
| Cloud sau khi cài gói nhẹ | pytest, oracle INT8/MQDA/DMP, mô hình chu kỳ; Icarus/Verilator sau khi cài | DSP placement/timing/power trên chip thật |
| Windows RTX 3060 | Thu thập manifest; sau B ready, FP32 accuracy và benchmark CUDA, calibration | Latency FPGA, công suất on-chip FPGA |
| Máy có Vivado 2022.2 | Synthesis/implementation cho XC7Z100, timing/resource và power estimate với activity | Công suất đo thực tế nếu không có cảm biến/board |
| Board đúng cấu hình | DDR/PCIe/XDMA, latency host–FPGA và điện năng theo thiết bị đo | Không hiện có quyền truy cập board |

Máy Windows do người dùng báo: i5-12400F, RAM 16 GB, RTX 3060 12 GB, Python 3.12.10, torch 2.11.0+cu128. Những thông tin này chưa được script cloud kiểm chứng. Không sao chép môi trường CUDA sang cloud để giả định có GPU.

## Kế hoạch A–I

| Giai đoạn | Đầu vào bắt buộc | Công việc và tiêu chí hoàn thành | Công cụ/nơi chạy |
| --- | --- | --- | --- |
| A | PDF và checkout Giai đoạn 1 | Phân tích, gap register, trace từng yêu cầu, ghi tool inventory; validator và negative tests pass; commit/PR | Cloud, stdlib |
| B | Manifest upstream commit/YAML/weights hash/classes; thống nhất VOC split và preprocessing; environment tương thích | Script inference, evaluator, benchmark warm-up/sync/lặp; lưu config và output JSON; FP32 chạy đúng ảnh và accuracy protocol xác định | Windows CUDA; CPU smoke có thể cloud sau cài |
| C | B có graph và calibration split độc lập test, quyết định PTQ/QAT, dtype/rounding/per-channel | Calibration có seed/manifest; affine scale/Z; test range hằng, clipping, biên, error metrics; so FP32/INT8 trên cùng eval split | Python/NumPy, PyTorch khi cần |
| D | C có domain specification và output ranges; chốt clamp/Q-format | Add/Cat oracle và pipeline riêng; unequal/equal domains, q<Z, saturation, concat ordering; ablation cùng checkpoint/protocol | Cloud NumPy, pytest |
| E | Signed/unsigned packing, guard/correction được đặc tả và DSP48E1 contract đối chiếu | Python oracle + RTL + TB; biên/random seed/cross-lane; latency/valid đúng; synthesis DSP mapping chỉ nếu Vivado thực chạy | Cloud Python/Icarus/Verilator; Vivado riêng |
| F | Bank depth, ownership, ready/valid/reset, transfer schedule rõ | Mô hình chu kỳ và RTL bank; stall/random reset/tail tests; so serialized/overlap với cùng workload; báo cycles dưới giả định bus | Cloud + simulator |
| G | C–F được kiểm chứng; từng operator layout và fixed point chốt | Conv33/11, Add/Cat, SPPF, Upsample, Focus có oracle + RTL + TB; tính đúng padding/stride/tail; synthesis nếu tool có | Cloud; Vivado tùy chọn |
| H | G pass; graph instruction/tensor address map; DMA contract | Scoreboard tích hợp so layer outputs, branch/fanout, stalls/CDC model; DDR/DMA mô hình có tham số, không gọi là XDMA thật | Cloud RTL/Python; board độc lập |
| I | Log B–H có provenance và input manifest | CSV/biểu đồ/report, accuracy cùng metric, latency tách pre/inference/post/transfer; đối chiếu paper và limitations; resource/power chỉ khi có tool/board log | Cloud tổng hợp, Windows benchmark |

B–I là kế hoạch, chưa có module thực thi. E/F có thể nghiên cứu độc lập khỏi accuracy nhưng chỉ bắt đầu khi đặc tả operand/bus sẵn sàng; không dùng benchmark Windows đang chờ làm bằng chứng. Trước build lớn: kiểm tra RAM/CPU, target/tool versions; hạn chế parallelism theo tài nguyên, không build toàn bộ upstream mặc định.

## Chính sách kết quả có thể tái lập

Mỗi kết quả có `result_kind`: `measured` (đo thực tế với thiết bị/phạm vi), `software_simulation`, `rtl_testbench`, `estimate`, `paper_reported`; lời kể người dùng có provenance riêng `user_reported`. Ghi command, git SHA, input hashes, environment, seed, shape/layout, dtype, repeat/warm-up count, timer và sync.

Benchmark B: đề xuất batch 1 và ảnh 640×640 để so phạm vi, nhưng xác nhận paper batch trước tuyên bố chính xác. Timer CUDA events cho GPU kernel và wall clock có synchronize cho toàn pipeline; warm-up riêng, nhiều lần lặp, mean/median/p95/std; tách preprocessing, inference, NMS và tổng. Không cộng các median rồi gọi là median end-to-end. Accuracy VOC cần split/classes/AP definition cụ thể, không đánh giá hai ảnh smoke rồi gọi mAP. Đây là thiết kế protocol tương lai, chưa chạy.

Không tự giả định băng thông DDR hoặc độ trễ DMA. Simulator F/H nhận config explicit; báo sensitivity theo transfer/compute, warm-up/drain, contention. Không điều chỉnh tham số chỉ để ép khớp 24.617 ms. Synthesis báo tool/part/constraints/slack; RTL generic multiplication đúng số học chưa chứng minh một DSP sinh hai MAC.

## Các đầu vào còn thiếu (gap register)

| ID | Thiếu | Cách giải quyết/ảnh hưởng |
| --- | --- | --- |
| GAP-01 | Commit YOLOv5, YAML, checkpoint, lớp/anchors/activation | Thu manifest Windows; đối chiếu Focus/ReLU/VOC; có thể cần checkpoint tác giả |
| GAP-02 | VOC split, label map, train recipe, AP/NMS protocol | Chọn baseline công khai có ghi giả định hoặc hỏi tác giả; chưa tái hiện mAP |
| GAP-03 | Calibration set/PTQ-QAT/per-channel/rounding | Chốt config sau B; không suy ra từ related work |
| GAP-04 | DMP sign correction và DSP port config | Oracle biên/exhaustive, đối chiếu tài liệu DSP48E1; chưa bit-exact |
| GAP-05 | MQDA Q-format, output Z/clamp/overflow | Hai oracle rõ nhãn; cần specification trước RTL |
| GAP-06 | RAM depth, tensor layout/tail, instruction/register ABI | Đề xuất contract ghi assumption, xác nhận theo graph |
| GAP-07 | DDR rate, PCIe generation, effective bandwidth/queue | Model tham số; board đo sau, không lấy 128bit×clock làm đo DDR |
| GAP-08 | GOPS denominator/MAC convention/power activity | Giữ nguyên paper metrics, hỏi tác giả hoặc báo giới hạn |

## Bước cần người dùng thực hiện

Chạy script trong `docs/reproduction/windows_handoff.md`, gửi manifest (không weights/dataset). Sau đó chọn baseline graph tương thích. Không cần cài thêm gói cho A. Không bắt đầu B khi chưa có manifest/đặc tả dữ liệu; nếu không có hiện vật tác giả, quyết định rõ tái hiện phương pháp dưới giả định thay vì exact reproduction.
