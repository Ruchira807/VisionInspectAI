import os
import psutil
import torch


def get_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)

    result = {
        "process_memory_mb": round(memory_mb, 2),
    }

    if torch.cuda.is_available():
        result["gpu_allocated_mb"] = round(
            torch.cuda.memory_allocated() / (1024 * 1024), 2
        )
        result["gpu_reserved_mb"] = round(
            torch.cuda.memory_reserved() / (1024 * 1024), 2
        )

    return result


def log_memory(stage: str):
    memory = get_memory_usage()
    print(f"[MEMORY] {stage}: {memory}")