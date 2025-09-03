#!/usr/bin/env python3
import subprocess
import sys
import torch

def run_cmd(cmd):
try:
out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
return out.strip()
except Exception as e:
return f"cmd_failed: {e}"

def main():
print("python:", sys.version.splitlines()[0])
print("torch:", torch.version)
try:
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
for i in range(torch.cuda.device_count()):
print(f"device[{i}]:", torch.cuda.get_device_name(i))
except Exception as e:
print("torch.cuda check failed:", e)

print("--- nvidia-smi output (if available) ---")
print(run_cmd("nvidia-smi || true"))
if name == 'main':
main()
