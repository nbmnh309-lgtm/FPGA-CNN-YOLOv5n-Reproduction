# Kiểm toán cuối cùng dự án FPGA-CNN-YOLOv5n-Reproduction

Ngày kiểm toán: **11/10/2026**, trên Windows PC tại `C:\Users\azzmi\FPGA-CNN-Temp`, nhánh `local/full-reproduction`. Đối chiếu báo cáo local, ma trận, tiến độ, mã nguồn, XML, CSV, log công cụ và SHA-256; không dùng trạng thái PASS đơn lẻ làm bằng chứng.

**Kết luận:** Có bằng chứng tái hiện một phần phương pháp và chạy thực nghiệm trên PC. Có **24 testcase Python + 2 subtest**, **8 testbench XSim**, **384 dòng benchmark CUDA**, **DMP tổng hợp dùng 1 DSP48E1 trên XC7Z020** và **runner Python có mã thoát thực tế 0**. Chưa tái hiện hệ thống FPGA/VOC nguyên bản, chưa chứng minh toàn NPU tổng hợp hoặc chạy 200 MHz. Cụm “maximum feasible” trong trạng thái cũ là quá mạnh: vẫn còn công việc khả thi trên PC này.

Bộ bằng chứng giữ trong Git: [final_audit_evidence.json](final_audit_evidence.json) và thư mục [audit_evidence](audit_evidence/). Script [audit_reproduction.py](../scripts/audit_reproduction.py) kiểm hash từng artifact, tính lại thống kê CSV, xác minh XML/log/part, kiểm dữ liệu giải nén và ghi mã thoát của tiến trình runner. Các log ghi cả những giới hạn/thất bại thực tế, không chỉ kết quả thành công.

## 1. Những phần đã thực sự chạy và có bằng chứng

### Môi trường và tên công cụ chính xác

Các tiến trình kiểm toán gọi Python `.venv` và phần mềm trên PC Windows này. Bằng chứng gồm đường dẫn executable, `platform`, `nvidia-smi`, CUDA khả dụng và log Vivado tại máy. Điều này xác nhận nơi chạy công cụ/thí nghiệm; không suy luận nơi xử lý mô hình trợ lý từ vị trí công cụ.

| Thành phần | Phiên bản / thiết bị xác minh |
|---|---|
| CPU | Intel Core i5-12400F, xác minh bằng `Get-CimInstance Win32_Processor` |
| GPU | NVIDIA GeForce RTX 3060, 12.288 MiB VRAM; driver 610.62 |
| Python dự án | 3.12.10, `C:\Users\azzmi\FPGA-CNN-Temp\.venv\Scripts\python.exe` |
| Python test runner | **pytest 9.1.1**; chạy cả unittest kế thừa và pytest tests; JUnit XML là định dạng đầu ra |
| NumPy / PyYAML | 2.5.2 / 6.0.3 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128; CUDA runtime 12.8, `torch.cuda.is_available() = True` |
| OpenCV / matplotlib | opencv-python 5.0.0.93 / matplotlib 3.11.2 |
| RTL compiler / elaborator / simulator | **Vivado XVLOG → XELAB → XSim (Vivado Simulator) 2020.2** |
| Tổng hợp RTL | **Vivado 2020.2 (64-bit), SW Build 3064766, IP Build 3064653** |
| FPGA đã tổng hợp thành công | **xc7z020clg484-1**, DMP-only, out-of-context |
| FPGA mục tiêu bài báo / local Tcl mặc định | **xc7z100ffg900-2**; chưa tổng hợp thành công do license |

Không gọi công cụ RTL này là Icarus Verilog, Verilator, ModelSim hay Vivado 2022.2. Bài báo dùng Vivado 2022.2, khác phiên bản local. Các header XSim trong log chạy thành công và `vivado.bat -version` xác nhận phiên bản. Thử `xsim --version` không phải phép kiểm chuẩn trên bản cài này: tùy chọn không được hỗ trợ có thể trả mã 1 dù banner hiển thị 2020.2; không dùng nó để kết luận testbench thất bại.

### Kiểm chứng năm tuyên bố chính

| Tuyên bố | Bằng chứng kiểm toán | Kết luận được phép |
|---|---|---|
| “26 trường hợp JUnit” | XML suite `tests=26`, nhưng chỉ **24 phần tử `<testcase>`**; terminal ghi `24 passed, 2 subtests passed`; failures/errors/skipped đều 0 | Sửa thành **24 testcase + 2 subtest đạt**; không có 26 testcase độc lập |
| “8 testbench XSim” | Đúng 8 tên trong `outputs/rtl/results.json`; mỗi tên có PASS và header XSim 2020.2/build; `commands.log` có xvlog/xelab/xsim; stage log và artifact hash khớp | Đủ bằng chứng 8 bài mô phỏng đã chạy thành công trong phạm vi test |
| “benchmark 384 mẫu” | CSV có **384 cặp (image, repeat) duy nhất**, 128 tên ảnh, mỗi ảnh đúng 3 lần, repeat 0/1/2; không có NaN/Inf hoặc thời gian âm; tính lại count/mean/median/p95/std/FPS khớp JSON | 384 lượt đo trên 128 ảnh, không phải 384 ảnh độc lập |
| “DMP 1 DSP48E1” | JSON `part=xc7z020clg484-1`, `dsp48e1=1`; utilization report và log synthesis khớp; Tcl đếm cell `REF_NAME == DSP48E1` rồi kiểm bằng 1 | Đã map DMP vào một DSP48E1 trên Z020; không phải toàn NPU hoặc Z100 |
| “mã thoát 0” | `runner_receipt.actual_process_returncode=0` từ `subprocess.run`; stdout và SHA log được lưu; state cuối có `last_exit_code=0`, các stage PASS và hash khớp | Runner Python local kết thúc 0; lần kiểm toán dùng cache hợp lệ, không ép chạy lại tất cả thí nghiệm |

XML và log Python trước/sau kiểm toán được lưu riêng trong `audit_evidence`. Sau sửa test, Python đã chạy lại và vẫn đạt 24 testcase + 2 subtest. Các lần runner tiếp theo dùng cache đã kiểm hash. Benchmark/RTL/synthesis trước đó được kiểm từ artifact và log gốc, không được trình bày như vừa chạy lại toàn bộ trong kiểm toán.

Wrapper `run_all_windows.ps1` bị Windows Execution Policy chặn ở lần thử kiểm toán, log giữ tại [powershell_policy_failure.log](audit_evidence/powershell_policy_failure.log). Đây là mã lỗi thật của việc mở wrapper, không phải lỗi CUDA/pytest. Đã gọi trực tiếp entry point Python tương đương và thu mã 0, **không thay đổi chính sách Windows**. Không tuyên bố wrapper mặc định luôn chạy được trên máy này.

### Phạm vi 8 bài XSim

| Testbench | Dữ liệu và hành vi kiểm tra |
|---|---|
| `tb_dmp` | 544.288 vector: toàn bộ 256² cặp weight ở 8 giá trị activation biên + 20.000 random; hai sản phẩm, latency/valid, bubble/reset |
| `tb_buffer` | 160 byte mới sau reset giữa luồng; input/output stalls, thứ tự dữ liệu, giữ output khi bị chặn |
| `tb_npu` | 27 ca toán tử, 3 kích thước tensor; Conv1/3, stride1/2, signed weight, bias/ReLU; chuỗi Focus→Upsample và reset-abort/restart |
| `tb_dot3` | Tile Conv3×3 16Cin×8Cout, 100 frame: 97 random + 3 biên; so Python, latency valid 3 stage |
| `tb_dot1` | Tile Conv1×1 16Cin×16Cout, 100 frame tương tự |
| `tb_stream` | Bank ping-pong→DMP, 256 vector, có stalls, đối chiếu thứ tự/kết quả |
| `tb_mqda0` | MQDA Add: 1.024 beat ×16 lane; domain/config đổi theo beat, biên/saturation, output backpressure |
| `tb_mqda1` | MQDA Cat alignment: 1.024 beat ×16 lane, cùng nhóm điều kiện |

Python còn kiểm DMP toàn miền **256×256×384 = 25.165.824 bộ ba**: weight signed8 và activation trong [-128,255]. Đây là miền oracle mở rộng; tile convolution dùng activation unsigned8. Một test exhaustive không tương đương hàng triệu testcase JUnit. Oracle convolution/spatial được kiểm chéo bằng PyTorch; tests runner kiểm cache hỏng, đổi nguồn, fail mã 7 và phục hồi.

### CUDA, QDQ và kernel INT8 CPU

Raw baseline: [fp32_samples.csv](fp32_samples.csv), aggregate: [local_results.json](local_results.json); bản gốc và log GPU giữ tại `audit_evidence`. Seed 20261011, batch1, 640×640, warm-up30, 128 ảnh COCO128 ×3, CUDA synchronization ở biên từng stage.

| Đại lượng | Mean (ms) | Median (ms) | P95 (ms) |
|---|---:|---:|---:|
| Preprocess + H2D | 0,9457 | 0,8450 | 1,4200 |
| Forward | 5,6178 | 5,4746 | 6,6733 |
| NMS + detection D2H | 1,1377 | 0,9809 | 1,3554 |
| End-to-end | **7,7011** | 7,3903 | 9,0890 |

FPS tính từ mean E2E: **129,851**. E2E gồm letterbox/BGR→RGB/H2D, forward với sigmoid/decode, GPU NMS và đưa detection về CPU; **không gồm đọc ảnh từ đĩa**, không overlap frame. Tổng ba stage khớp E2E từng dòng. NMS conf0.25/IoU0.45/max_det300. Đây là đo GPU PC của checkpoint COCO local, không phải FPS FPGA.

Full-network QDQ/MQDA và COCO128 AP thật sự được chạy: mAP@0.5 FP32 **54,616%**, QDQ+MQDA **52,573%**, bỏ alignment **0,483%**; mAP@0.5:0.95 tương ứng **34,896% / 32,566% / 0,202%**. QDQ convolution vẫn FP32 CUDA với weight/activation đã lượng tử rồi giải lượng tử, bias FP32. Chênh lệch ablation lớn do baseline bỏ domain alignment, không tái hiện mức cải thiện 0,53 điểm của bài báo. 32 ảnh calibration nằm trong đánh giá 128 ảnh/929 instance; COCO128 thuộc train2017, có nguy cơ trùng pretraining. Chỉ dùng AP như sanity check, không validation độc lập.

Có kernel Conv2d INT8 CPU thực bằng **oneDNN**, không phải x86/fbgemm trong build Windows này. Shape 1×16×32×32, Cout8 cho K3/Cout16 cho K1, 30 warm-up, 200 lượt/backend, 4 threads. Mean FP32/INT8: K3 **0,03836/0,06602 ms**, K1 **0,01772/0,06861 ms**. Trong workload nhỏ này INT8 chậm hơn FP32; không có bằng chứng tăng tốc toàn mạng. Log, backend, aggregate và MSE có lưu, nhưng **không có CSV từng lượt CPU hoặc latency QDQ** để tính lại distribution độc lập như baseline384. Mức đầy đủ của bằng chứng hai phép đo này thấp hơn baseline FP32.

## 2. Những phần chỉ mới mô phỏng chức năng

- `reproduction/reference.py`: affine quantization, MQDA Add/Cat, Q16 requantization, signed DMP, bảy toán tử, memory instruction engine và lịch hai bank. Có đối chiếu oracle nhưng không đo chip.
- `rtl/top/npu.sv`: behavioral memory engine với vòng lặp/longint, một phần tử đầu ra mỗi chu kỳ, TB nạp bộ nhớ trực tiếp. Không dùng các tile song song làm datapath thực và chưa inference toàn đồ thị YOLO.
- `rtl/common/conv_dot.sv`: mô phỏng arithmetic tile với 576/128 instance DMP trong nguồn. Chưa tổng hợp tile; chưa line buffer, scheduler/partial sum toàn kênh, requantization và kết nối NPU. Reduction cộng trực tiếp, latency local khác pipeline bài.
- `rtl/mqda/align.sv`: Add/Cat 16 lane, Q16, round half-away-from-zero, pipeline4 stage và stall. Chưa tổng hợp; không chứng minh trùng pipeline Add12/Cat11 chu kỳ của tác giả. Cat là alignment beat, chưa controller sắp toàn tensor.
- `rtl/ping_pong_buffer/buffer.sv` và tích hợp bank→DMP: simulation ready/valid/reset/stalls; chưa chứng minh BRAM mapping hoặc DMA/DDR thực.
- Lịch Python giả định 20 block, read5/compute20/write5: serialized600 so overlap410 chu kỳ. Đây là số chu kỳ mô hình, không latency/FPS hoặc speedup FPGA đo.
- Full-network QDQ và 100.000 cặp MQDA random thể hiện tác động số học lượng tử; không phải backend native INT8 toàn mạng hay bit-exact RTL của bài.

Seeds và script sinh golden/TB tồn tại. Tuy nhiên nhiều RTL golden dùng cùng `reference.py`; kiểm chéo PyTorch giảm rủi ro nhưng không thay thế oracle độc lập, formal proof hoặc RTL/checkpoint của tác giả. 100 frame mỗi tile và các tensor nhỏ không bao phủ mọi kích thước/full-network overflow.

## 3. Những phần đã tổng hợp RTL

**Chỉ DMP** có bằng chứng tổng hợp thành công: `scripts/synth_dmp.tcl`, top `dmp`, `synth_design -mode out_of_context`, part **xc7z020clg484-1**. Utilization: **1 DSP48E1, 2 Slice LUT, 34 Slice Register**, không phải tài nguyên của hệ thống CNN. Log đầy đủ, utilization/timing report và JSON được lưu.

Không có bằng chứng tổng hợp buffer, MQDA, Conv tiles hay full NPU; cách viết “synthesizable” trước đây đã sửa thành ứng viên RTL chưa được tổng hợp. DMP nhận hai weight dùng chung activation; không phải hai phép nhân với bốn operand độc lập.

Tcl đặt clock5ns sau synthesis và tạo timing summary; chưa có place/route, bitstream hay timing closure. Report còn **27 input port thiếu input delay, 33 output port thiếu output delay**. Vì vậy không dùng clock constraint5ns hoặc slack của synth-only để khẳng định chạy đạt200MHz trên board.

Lần tổng hợp part bài báo **xc7z100ffg900-2** thất bại thật với **Common17-348/17-345: thiếu license Synthesis/device xc7z100**; giữ log `outputs__rtl__synthesis.log`. Z020 là surrogate cho kiểm chứng primitive, không thay thế kết quả tài nguyên/timing của Z100.

## 4. Những phần chưa tái hiện được so với bài báo

| Nội dung bài báo | Trạng thái local / nguyên nhân |
|---|---|
| YOLOv5n Focus/ReLU, VOC20, checkpoint/train/calibration recipe | Local COCO80, stem Conv6×6/SiLU, 1.867.405 fused params; thiếu artifact tác giả |
| VOC2007 mAP71,54% / INT8+MQDA68,79% | Chưa chạy trên đúng model/split/định nghĩa AP; COCO128 không thay thế |
| MQDA rounding/Q-format/zero-point/bias units, Add12/Cat11 cycles | Local có giả định rõ ràng; chưa xác định tương đương bit-exact |
| Mảng convolution, line/weight buffers, partial sums và bảy toán tử tích hợp toàn mạng | Có oracle, tile simulation và behavioral engine; chưa datapath/controller toàn hệ thống |
| XC7Z100, Vivado2022.2, clock200MHz | Local2020.2, Z100 bị license chặn; chỉ DMP Z020 synth-only |
| AXI/MIG/DDR3, PCIe/XDMA, host pipeline thực | Chưa triển khai/chạy; memory TB không phải giao tiếp vật lý |
| 752DSP, 128.535LUT, 176.588FF, 273,5BRAM | Chỉ1DSP của DMP; chưa full resource report |
| NPU24,617ms, E2E36,686ms, 27,2FPS | Chưa đo board; benchmark CUDA khác workload/graph |
| Power8,536W; 182,8GOPS; 21,42GOPS/W | Chưa có full implementation/power estimate hoặc phép đo phù hợp |
| Full-network native INT8 throughput | QDQ dùng FP32; oneDNN chỉ hai kernel CPU nhỏ |

Không chứng minh các hạng mục vật lý trên PC hiện tại chỉ bằng software simulation. Những sai khác định nghĩa throughput/AP đã ghi trong phân tích bài báo phải giữ là câu hỏi nghiên cứu, không tự sửa số gốc hay điền số suy đoán vào kết quả local.

## 5. Các kết quả có thể sử dụng trong báo cáo với giảng viên

Có thể viết: **“Dự án tái hiện một phần phương pháp YOLOv5n accelerator bằng oracle Python, mô phỏng RTL XSim và thí nghiệm CUDA trên PC. 24 testcase cùng 2 subtest đạt; 8 testbench XSim đạt; benchmark FP32 có384 lượt trên128 ảnh COCO128; DMP map vào1DSP48E1 trênXC7Z020 bằng Vivado2020.2. Chưa tái hiện hệ thống FPGA/VOC nguyên bản.”**

Được dùng bảng thời gian CUDA ở mục1 kèm đúng scope, bảng QDQ AP với nhãn COCO128 sanity và calibration overlap, kết quả correctness DMP/MQDA/buffer, bảng tổng hợp DMP-only, các hạn chế và lộ trình. Có thể báo INT8 oneDNN chậm hơn FP32 ở kernel nhỏ đã thử; chưa đủ để kết luận INT8 nói chung chậm hơn.

Không dùng 129,851FPS như tốc độ FPGA hay speedup so với27,2FPS bài báo; không dùng1DSP như tài nguyên full NPU; không gọi26 testcase độc lập; không gọi AP COCO128 là VOC reproduction; không báo200MHz/power/GOPS/W nếu chưa có implementation tương ứng. Exit0 chỉ chứng minh workflow trong scope kết thúc thành công, không chứng minh mọi mục tiêu nghiên cứu đã hoàn thành.

Các sửa an toàn đã thực hiện trong kiểm toán:

1. Sửa generator và tài liệu về JUnit24+2, tile97random+3biên, chưa tổng hợp tile, coverage giai đoạnG/H và phạm vi tái hiện một phần; giữ sửa trong generator để runner không ghi đè lại lỗi.
2. Sửa constraint pytest từ `>=8,<9` sang `>=9,<10` để tương thích pytest9.1.1 đã tạo bằng chứng. Đây chưa phải lock môi trường đầy đủ.
3. Hai test cache/signature chuyển sang thư mục tạm riêng thay vì tên cố định dưới outputs, tránh ghi đè file có sẵn của người dùng. Suite chạy lại vẫn đạt.
4. Runner ghi `NOT_VERIFIED` và trả lỗi nếu không tìm thấy Vivado, thay vì có thể bỏ qua synthesis rồi báo0. Nhánh thiếu Vivado được rà mã; môi trường hiện tại có Vivado nên không giả lập đã thực chạy nhánh đó.
5. Lưu log, artifact snapshot, hash và receipt mã thoát thật. Giữ benchmark384 gốc, không tạo lại số đo để thay các kết quả đã báo.

## 6. Những việc cần làm tiếp theo trên PC hiện tại

### Tái lập và độ đầy đủ bằng chứng

Nguồn upstream đang đúng commit **f9578796b8a64889aa9e66dc06914d19f40d338e**. Hash checkpoint **4f180cf23ba0717ada0badd6c685026d73d48f184d00fc159c2641284b2ac0a3**, COCO128 zip **61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e** khớp file hiện tại. Kiểm toán còn so byte hash **257 file .jpg/.txt** giải nén với zip. Hash xác nhận identity, không xác nhận checkpoint là của tác giả; file hiện có của người dùng chưa có provenance tải gốc.

Từ PowerShell ở thư mục dự án, chạy entry point này nếu wrapper bị chặn:

```powershell
& .\.venv\Scripts\python.exe -B windows_runner/runner.py
# Chạy lại tất cả stage, không dùng cache benchmark:
& .\.venv\Scripts\python.exe -B windows_runner/runner.py --force
# Kiểm toán artifact và thu receipt runner:
& .\.venv\Scripts\python.exe -B scripts/audit_reproduction.py
```

Lần `--force` sẽ sinh thời gian mới; sao lưu bằng chứng cũ trước và báo distribution/run mới riêng. Seed cố định giúp tái tạo dữ liệu kiểm thử, không bảo đảm latency hoặc CUDA/AP bitwise identical. `.venv`, dữ liệu, weights, generated vectors và simulator workspace không có trong Git; cần phục hồi đúng inputs, dependencies và Vivado để chạy lại trên máy mới. Metadata packages hiện có không phải một installer lock đã được kiểm nghiệm từ môi trường trống.

Thứ tự ưu tiên khả thi:

1. **Củng cố workflow:** pin dependency/CUDA wheel và tool paths trong hướng dẫn cài mới; lưu raw từng lượt CPU/QDQ, AP per-image/predictions, returncode từng lệnh XVLOG/XELAB/XSim và config đầy đủ. Nội bộ FP32 cache hiện chỉ so metadata checkpoint/archive/torch/device/shape/warm-up và sự tồn tại CSV, chưa hash source/ảnh/CSV; runner ngoài có hash artifact/source nhưng khi source đổi mà benchmark nội bộ dùng lại cache vẫn có rủi ro số cũ. Dùng `--force` cho thí nghiệm mới cho đến khi cache được bổ sung manifest/hash và test corruption. Giới hạn này đã sửa trong tài liệu, chưa triển khai lại cache trong đợt kiểm toán.
2. **Tổng hợp các module nhỏ trên Z020:** buffer và MQDA riêng, rồi Conv tiles giảm song song phù hợp tài nguyên. Full Conv33 với576 DMP lớn hơn220 DSP của Z020, không thể giữ nguyên cấu hình đó để fit; dùng giảm lane/time multiplexing và ghi rõ khác paper. Đặt constraints trước synthesis, thêm I/O constraints phù hợp, chạy implementation/place-route và báo timing/utilization theo scope thực.
3. **Tăng độ đúng RTL:** oracle fixed-point độc lập, boundary/overflow/tail channels, formal assertions bank ownership/ready-valid, tích hợp line/weight buffers, partial-sum/requant/bias. So layer/tensor với software, rồi inference đồ thị nhỏ trước khi toàn YOLO. Giữ giả định rounding/Q-format rõ tên; không tự nhận bit-exact tác giả.
4. **Đánh giá model nghiêm túc trên RTX3060:** tách calibration khỏi validation; dùng VOC20 và mô hình Focus/ReLU khi có đủ recipe. Có thể huấn luyện/đánh giá một cấu hình công khai riêng nếu thiếu checkpoint tác giả, nhưng phải gọi là tái hiện theo giả định. Thử backend native INT8 toàn mạng khả dụng sau khi kiểm support/operator accuracy; không lấy QDQ latency làm INT8 speedup.
5. **Điều kiện ngoài PC:** xin checkpoint/YAML/calibration/AP protocol và RTL/instructions của tác giả; licenseXC7Z100; board phù hợp và giao tiếp/driver. Sau đó mới có thể đo DDR/PCIe/NPU/full E2E, full resource/power và so cùng workload với bài báo.

Đợt kiểm toán đã đạt mục tiêu xác minh và sửa các sai lệch an toàn, không đồng nghĩa đã thực hiện các nghiên cứu tiếp theo ở trên. Các thay đổi và bằng chứng được commit cục bộ; không push GitHub hoặc tạo Pull Request.
