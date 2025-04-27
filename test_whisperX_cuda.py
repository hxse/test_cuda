# from dotenv import load_dotenv
import os
import whisperx
import torch
import time

# load_dotenv()
# print("CUDA_PATH:", os.environ.get("CUDA_PATH"))

# 这行代码看起来是为了添加 CUDA bin 目录到 PATH，
# 但在大多数标准安装中，如果你通过 pip 或 conda 安装了 PyTorch GPU 版本，
# 并且系统有兼容的 NVIDIA 驱动，通常不需要手动设置 CUDA_BIN_PATH 环境变量。
# 如果你确实需要这个，请确保 CUDA_BIN_PATH 环境变量已正确设置。
# print(os.environ.get("CUDA_BIN_PATH", ""))
# os.environ["PATH"] += os.pathsep + os.environ.get("CUDA_BIN_PATH", "")


def run_whisperx(audio_file, model_name="base", batch_size=16, device="auto"):
    """
    运行 WhisperX 转录音频，并可选择指定设备。
    函数内部减少日志输出，主要用于返回结果和时间。
    优化了错误处理和资源释放逻辑以避免 Linter 警告。

    Args:
        audio_file (str): 音频文件路径。
        model_name (str): WhisperX 模型名称。
        batch_size (int): 转录的批处理大小。
        device (str): 指定运行设备 ('cpu', 'cuda', 'auto')。
                       'auto' 会优先使用 CUDA，如果不可用则使用 CPU。

    Returns:
        tuple: (result, transcribe_time)。result 是转录结果字典，transcribe_time 是转录耗时（秒）。
               如果运行失败或 CUDA 不可用，返回 (None, None)。
    """
    model = None  # 初始化 model 为 None
    result = None  # 初始化 result 为 None
    transcribe_time = None  # 初始化 transcribe_time 为 None

    if device == "auto":
        run_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        run_device = device

    if run_device == "cuda":
        if not torch.cuda.is_available():
            print("CUDA 不可用！无法在 CUDA 设备上运行。")
            return None, None
        # print(f"将在 CUDA 设备上运行: {torch.cuda.get_device_name(0)}") # 移到main中打印
        compute_type = "float16"
    else:
        # print("将在 CPU 设备上运行。") # 移到main中打印
        compute_type = "int8"
        if batch_size > 4:
            # print(f"注意：在 CPU 上运行，batch_size 从 {batch_size} 调整为 4。") # 保留此调整提示
            batch_size = 4

    # 加载模型
    # print(f"加载 WhisperX 模型: {model_name} (设备: {run_device}, 计算类型: {compute_type})") # 移到main中打印
    start_time = time.time()
    try:
        model = whisperx.load_model(
            model_name, device=run_device, compute_type=compute_type
        )
        load_time = time.time() - start_time
        # print(f"模型加载耗时: {load_time:.2f} 秒") # 移到main中打印
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None  # 加载失败，直接返回 None, None

    # 转录音频
    # print(f"转录音频: {audio_file}") # 移到main中打印
    start_time = time.time()
    try:
        result = model.transcribe(audio_file, batch_size=batch_size)
        transcribe_time = time.time() - start_time
        # print(f"转录耗时: {transcribe_time:.2f} 秒") # 移到main中打印
        # print(f"语言: {result['language']}") # 语言信息可以在main中按需打印
        # ... (注释掉的结果打印) ...

    except Exception as e:
        print(f"转录失败: {e}")
        # 转录失败，让 model 在函数结束时由 Python 垃圾回收
        # 清理显存 (如果是在 CUDA 设备上)
        if run_device == "cuda":
            torch.cuda.empty_cache()
        return None, None  # 转录失败，返回 None, None

    # 运行成功后的清理
    # 让 model 在函数结束时由 Python 垃圾回收
    # 清理显存 (如果是在 CUDA 设备上)
    if run_device == "cuda":
        torch.cuda.empty_cache()

    # 返回结果和时间
    return result, transcribe_time


if __name__ == "__main__":
    audio_file = "20240102 I Was Adopted to Keep an Eye on a Billionaire’s Daughter rDbqN7YHw1U.mp3"  # 替换为你的音频文件路径
    model_to_use = "base"  # 你可以尝试其他模型如 tiny, small, medium, large-v2/v3
    comparison_batch_size = 16  # 用于对比测试的 batch_size

    # --- CPU 性能测试 ---
    print("\n--- 开始 CPU 性能测试 ---")
    print(f"运行设备: CPU")
    print(f"加载模型: {model_to_use}")
    print(f"音频文件: {audio_file}")
    print(f"Batch Size: {comparison_batch_size}")
    cpu_result, cpu_time = run_whisperx(
        audio_file,
        model_name=model_to_use,
        batch_size=comparison_batch_size,
        device="cpu",
    )
    if cpu_time is not None:
        print(f"CPU 模型加载及转录总耗时: {cpu_time:.2f} 秒")
        if cpu_result and "language" in cpu_result:
            print(f"CPU 转录语言: {cpu_result['language']}")

    # --- GPU 性能测试 (如果可用) ---
    if torch.cuda.is_available():
        print("\n--- 开始 GPU 性能测试 ---")
        print(f"运行设备: GPU ({torch.cuda.get_device_name(0)})")
        print(f"加载模型: {model_to_use}")
        print(f"音频文件: {audio_file}")
        print(f"Batch Size: {comparison_batch_size}")
        gpu_result, gpu_time = run_whisperx(
            audio_file,
            model_name=model_to_use,
            batch_size=comparison_batch_size,
            device="cuda",
        )
        if gpu_time is not None:
            print(f"GPU 模型加载及转录总耗时: {gpu_time:.2f} 秒")
            if gpu_result and "language" in gpu_result:
                print(f"GPU 转录语言: {gpu_result['language']}")

        # --- 性能对比 ---
        if cpu_time is not None and gpu_time is not None:
            print("\n--- 性能对比 ---")
            print(f"CPU 耗时: {cpu_time:.2f} 秒")
            print(f"GPU 耗时: {gpu_time:.2f} 秒")
            if gpu_time < cpu_time:
                print(f"GPU 比 CPU 快约 {cpu_time / gpu_time:.2f} 倍")
            else:
                print(
                    f"CPU 比 GPU 快约 {gpu_time / cpu_time:.2f} 倍 (这通常不正常，请检查GPU是否正常工作)"
                )
    else:
        print("\n未检测到 CUDA 可用设备，跳过 GPU 性能测试。")
