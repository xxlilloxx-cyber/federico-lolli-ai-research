import json
import platform
import shutil
import subprocess
import time


def main():
    info = {"python": platform.python_version(), "free_disk_bytes": shutil.disk_usage('.').free}
    try:
        smi = subprocess.Popen(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 8
        while smi.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if smi.poll() is None:
            smi.kill()
            info["nvidia_smi"] = "timed out (possible uninterruptible kernel wait)"
        else:
            stdout, stderr = smi.communicate()
            info["nvidia_smi"] = stdout.strip() or stderr.strip()
    except Exception as exc:
        info["nvidia_smi"] = str(exc)
    try:
        import torch
        info.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                    cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
            a = torch.randn(64, 64, device='cuda', requires_grad=True)
            with torch.autocast('cuda', dtype=torch.float16):
                loss = (a @ a).square().mean()
            loss.backward()
            info["fp16_forward_backward"] = bool(torch.isfinite(loss).item())
            info["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    except Exception as exc:
        info["torch_error"] = repr(exc)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
