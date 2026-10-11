# Thu thập đầu vào Windows cho Giai đoạn B

Chưa cần chạy benchmark hoặc tải thêm model. Cloud không có quyền truy cập GPU Windows. Script chỉ đọc phiên bản, tên GPU/CUDA, Git commit và SHA-256 các file bạn chỉ định; không đọc checkpoint bằng pickle, không load model, không suy luận. Không thu thập biến môi trường, remote URL, token hoặc nội dung dataset.

Sau khi PR được merge, chạy trong PowerShell:

```powershell
Set-Location 'C:\Users\azzmi\FPGA-CNN-Temp'
git pull --ff-only origin main
# Dùng Python của môi trường PyTorch bạn đã chuẩn bị; thay đường dẫn nếu cần.
& .\scripts\collect_windows_manifest.ps1 -Python '.\.venv\Scripts\python.exe'
```

Nếu môi trường Python không ở `.venv`, dùng `-Python 'đường-dẫn-python.exe-thực-tế'`. Nếu PowerShell chặn script, có thể gọi trực tiếp (không cần đổi execution policy):

```powershell
& .\.venv\Scripts\python.exe .\scripts\collect_reproduction_manifest.py --output .\outputs\windows_manifest.json --torch-info
```

Để xác định model đã chạy, thêm các đường dẫn **thực tế**; ví dụ dưới đây chỉ minh họa, không có các file này trong bộ khung:

```powershell
& .\scripts\collect_windows_manifest.ps1 `
  -Python '.\.venv\Scripts\python.exe' `
  -Upstream 'C:\duong-dan-thuc-te\yolov5' `
  -Weights 'C:\duong-dan-thuc-te\yolov5n.pt' `
  -ModelYaml 'C:\duong-dan-thuc-te\yolov5n.yaml' `
  -DataYaml 'C:\duong-dan-thuc-te\voc.yaml'
```

Nếu chưa biết YAML/weights ở đâu, chạy lệnh tối thiểu và báo phần chưa có. Script không đoán release theo tên file. Dirty upstream cần diff/model graph để giải thích; hash YAML không tự xác nhận các layer bên trong.

Đầu ra mong đợi: `outputs/windows_manifest.json`, có phiên bản Python/torch, `cuda.available`, tên GPU, upstream SHA và checksum khi cung cấp file. Thư mục outputs bị Git ignore. Gửi JSON sau khi tự kiểm tra nội dung; không cần gửi trọng số/dataset hoặc thông tin đăng nhập.

Kèm thông tin: bạn đã dùng YOLOv5 từ repository/commit nào, checkpoint COCO hay VOC, YAML graph/model summary nếu có, bộ dữ liệu và split dự định đánh giá. YAML chỉ gửi nếu không có đường dẫn riêng/tokens cần giữ kín. Sau khi nhận dữ liệu này mới chốt baseline B, preprocessing và accuracy protocol.

Collector Python đã smoke-test trên cloud không có CUDA. **Wrapper PowerShell chưa được chạy trên Windows trong phiên này**; cần log Windows để xác nhận CUDA. Không diễn giải chạy collector thành benchmark thành công.
