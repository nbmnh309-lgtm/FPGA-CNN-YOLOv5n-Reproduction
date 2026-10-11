"""Kiểm kê công cụ; không cài đặt, huấn luyện hoặc chạy mô phỏng."""
import importlib.metadata
import platform
import shutil
import subprocess

print(f"Python: {platform.python_version()}")
for package in ("pip", "numpy", "PyYAML", "pytest", "torch", "torchvision", "onnx", "opencv-python-headless"):
    try:
        print(f"{package}: {importlib.metadata.version(package)}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: CHƯA CÓ")
for tool, flag in (("git", "--version"), ("iverilog", "-V"), ("vvp", "-V"), ("verilator", "--version"), ("make", "--version"), ("g++", "--version"), ("cmake", "--version"), ("yosys", "-V")):
    path = shutil.which(tool)
    if not path:
        print(f"{tool}: CHƯA CÓ")
        continue
    result = subprocess.run([path, flag], capture_output=True, text=True, timeout=15)
    lines = (result.stdout + result.stderr).strip().splitlines()
    print(f"{tool}: {lines[0] if lines else path} (exit={result.returncode})")
for tool in ("vivado", "vitis_hls"):
    print(f"{tool}: {shutil.which(tool) or 'CHƯA CÓ'}")
print("Đây là báo cáo kiểm kê, không phải kiểm thử CNN/RTL. Thiếu công cụ không làm lệnh này thất bại.")
