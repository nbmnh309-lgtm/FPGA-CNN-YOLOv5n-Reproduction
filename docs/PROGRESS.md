# Tiến độ local

Nhánh `local/full-reproduction`; kế thừa87dda2c từ remote phase-a bằng fast-forward, giữ checkpoint/outputs/upstream người dùng. Chưa push hay PR.

- python_tests: PASS (outputs\logs\python_tests_20261011_191210_783649.log)
- inputs: PASS (outputs\logs\inputs_20261011_185959.log)
- rtl: PASS (outputs\logs\rtl_20261011_191009.log)
- gpu_experiments: PASS (outputs\logs\gpu_experiments_20261011_190615.log)
- dmp_synthesis_z020: PASS (outputs\logs\dmp_synthesis_z020_20261011_190038.log)

Đã chạy Python/XSim/CUDA và synthesis DMP surrogate XC7Z020; xem báo cáo tiếng Việt và ma trận. XC7Z100 thiếu license. Chưa hoàn thành original FPGA/VOC reproduction.

Tiếp tục: `& .\windows_runner\run_all_windows.ps1`. Runner source/artifact hash kiểm checkpoint; không skip FAIL. Raw benchmark có cache riêng đã kiểm checkpoint/dataset/PyTorch.
