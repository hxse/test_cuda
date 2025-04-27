import numpy as np
from numba import cuda, njit
import time
import math
import sys  # 导入 sys 模块用于退出

# 定义一个小的容差用于浮点数与零的比较
ZERO_TOLERANCE = 1e-9  # 或者根据需要调整

# --- GPU 版本 (RSI) ---


@cuda.jit
def calculate_changes_kernel(prices, changes_out):
    """
    CUDA 核函数：计算价格变化。changes_out[i] = prices[i] - prices[i-1]
    从索引 1 开始计算。
    """
    idx = cuda.grid(1)
    # 确保索引在有效范围内 (prices.size) 且大于 0
    if idx > 0 and idx < prices.size:
        changes_out[idx] = prices[idx] - prices[idx - 1]


@cuda.jit
def calculate_gains_losses_kernel(changes, gains_out, losses_out):
    """
    CUDA 核函数：将价格变化分离为增益和损失。
    从索引 1 开始处理，因为 changes[0] 通常是 0。
    """
    idx = cuda.grid(1)
    # 确保索引在有效范围内 (changes.size) 且大于 0
    if idx > 0 and idx < changes.size:
        change = changes[idx]
        if change > 0:
            gains_out[idx] = change
            losses_out[idx] = 0.0
        else:
            gains_out[idx] = 0.0
            losses_out[idx] = -change  # 损失存储为正值


@cuda.jit
def calculate_rolling_sum_for_rsi_kernel(data, rolling_sum_out, period):
    """
    CUDA 核函数：计算用于 RSI 的滑动窗口求和。
    每个线程计算 rolling_sum_out 数组中一个元素的求和值。
    窗口大小为 period。输出索引 i 对应 data 数组中从索引 i+1 到 i+period 的求和。
    """
    idx = cuda.grid(1)  # 对应 rolling_sum_out 的索引 (0 到 rolling_sum_out.size - 1)

    # 窗口在 data 数组中的开始索引是 idx + 1
    # 窗口在 data 数组中的结束索引是 idx + period
    window_start_idx = idx + 1
    window_end_idx = idx + period

    # 修复：确保最后一个有效的窗口被计算
    if idx < rolling_sum_out.size:  # 确保 idx 在输出数组范围内
        current_sum = 0.0
        for i in range(window_start_idx, window_end_idx + 1):
            current_sum += data[i]
        rolling_sum_out[idx] = current_sum


@cuda.jit
def calculate_rsi_from_sums_kernel(
    rolling_sum_gains, rolling_sum_losses, period, rsi_out
):
    """
    CUDA 核函数：根据平滑后的增益和损失总和计算 RSI 值。
    每个线程计算 rsi_out 数组中一个元素。
    在判断平均损失是否为零时使用容差。
    """
    idx = cuda.grid(1)  # 对应 rsi_out 的索引 (0 到 rsi_out.size - 1)

    if idx < rsi_out.size:
        avg_gain = rolling_sum_gains[idx] / period
        avg_loss = rolling_sum_losses[idx] / period

        # 使用容差判断 avg_loss 是否接近零
        if math.fabs(avg_loss) > ZERO_TOLERANCE:  # <-- 修改这里，使用容差
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            # 惯例：当平均损失接近零时，RSI 为 100
            rsi = 100.0

        rsi_out[idx] = rsi


def calculate_rsi_gpu(prices, period):
    """
    使用 Numba CUDA 在 GPU 上计算相对强度指标 (RSI)，使用 SMA 平滑。

    参数：
        prices (np.ndarray): 输入价格时间序列数组 (float32)。
        period (int): RSI 计算周期。

    返回：
        tuple: (rsi_array, total_gpu_time, gpu_gains, gpu_losses, gpu_rolling_sum_gains, gpu_rolling_sum_losses)。
               rsi_array 是 GPU 计算结果，total_gpu_time 是 GPU 整个计算过程耗时 (不含初始传输)，
               返回中间数组用于调试。
               如果 CUDA 不可用或输入无效，返回 (None, None, ...)。
    """
    if not cuda.is_available():
        print("CUDA 不可用，无法运行 GPU RSI 版本！")
        return None, None, None, None, None, None

    if period <= 1:
        print("错误：RSI 周期必须大于 1。")
        return None, None, None, None, None, None

    if prices.size < period:
        print(f"警告：价格数据长度 ({prices.size}) 小于周期 ({period})，无法计算 RSI。")
        # 返回空数组和 0 时间，以及 None 中间数组
        return np.array([], dtype=np.float32), 0.0, None, None, None, None

    if prices.dtype != np.float32:
        print("警告: 输入价格数组不是 float32 类型，正在转换...")
        prices = prices.astype(np.float32)

    N = prices.size
    # RSI 的输出长度是 N - period + 1
    rsi_output_size = N - period + 1

    # 检查输出长度是否有效
    if rsi_output_size <= 0:
        print(
            f"警告：价格数据长度 ({prices.size}) 小于周期 ({period})，RSI 输出长度为 {rsi_output_size}。"
        )
        # 返回空数组和 0 时间，以及 None 中间数组
        return np.array([], dtype=np.float32), 0.0, None, None, None, None

    # --- 在设备上分配中间和结果数组 ---
    d_prices = cuda.to_device(prices)
    # changes 数组大小 N，从索引 1 到 N-1 有效
    d_changes = cuda.device_array(N, dtype=np.float32)
    # gains/losses 数组大小 N，从索引 1 到 N-1 有效
    d_gains = cuda.device_array(N, dtype=np.float32)
    d_losses = cuda.device_array(N, dtype=np.float32)

    # 平滑后的增益/损失总和数组大小：N - period + 1 (与 RSI 输出大小一致)
    sums_output_size = rsi_output_size

    d_rolling_sum_gains = cuda.device_array(sums_output_size, dtype=np.float32)
    d_rolling_sum_losses = cuda.device_array(sums_output_size, dtype=np.float32)

    d_rsi_out = cuda.device_array(rsi_output_size, dtype=np.float32)

    threads_per_block = 256
    blocks_cgl = int(math.ceil(N / threads_per_block))  # For changes, gains, losses
    blocks_sums = int(
        math.ceil(sums_output_size / threads_per_block)
    )  # For rolling sums
    blocks_rsi = int(math.ceil(rsi_output_size / threads_per_block))  # For final RSI

    # --- 总 GPU 过程耗时 ---
    total_start_time = time.time()

    # 核函数 1: 计算价格变化
    calculate_changes_kernel[blocks_cgl, threads_per_block](d_prices, d_changes)
    # cuda.synchronize() # 每个核函数后同步用于调试，计算总时间时移除

    # 核函数 2: 计算增益和损失
    calculate_gains_losses_kernel[blocks_cgl, threads_per_block](
        d_changes, d_gains, d_losses
    )
    # cuda.synchronize()

    # 核函数 3: 计算增益的滑动求和
    calculate_rolling_sum_for_rsi_kernel[blocks_sums, threads_per_block](
        d_gains, d_rolling_sum_gains, period
    )
    # cuda.synchronize()

    # 核函数 4: 计算损失的滑动求和
    calculate_rolling_sum_for_rsi_kernel[blocks_sums, threads_per_block](
        d_losses, d_rolling_sum_losses, period
    )
    # cuda.synchronize()

    # 核函数 5: 根据总和计算 RS 和 RSI
    calculate_rsi_from_sums_kernel[blocks_rsi, threads_per_block](
        d_rolling_sum_gains, d_rolling_sum_losses, period, d_rsi_out
    )
    cuda.synchronize()  # 最终同步

    total_end_time = time.time()
    total_gpu_time = total_end_time - total_start_time

    # --- 从设备取回结果和中间数组 ---
    rsi_array = d_rsi_out.copy_to_host()
    gpu_gains = d_gains.copy_to_host()
    gpu_losses = d_losses.copy_to_host()
    gpu_rolling_sum_gains = d_rolling_sum_gains.copy_to_host()
    gpu_rolling_sum_losses = d_rolling_sum_losses.copy_to_host()

    # 显式删除设备数组释放内存
    del (
        d_prices,
        d_changes,
        d_gains,
        d_losses,
        d_rolling_sum_gains,
        d_rolling_sum_losses,
        d_rsi_out,
    )
    cuda.current_context().deallocations.clear()  # 必要时强制释放

    # 返回结果、时间、和中间数组用于调试
    return (
        rsi_array,
        total_gpu_time,
        gpu_gains,
        gpu_losses,
        gpu_rolling_sum_gains,
        gpu_rolling_sum_losses,
    )


# --- CPU 版本 (RSI) - 使用 Numba @njit ---


@njit
def calculate_rsi_cpu_njit(prices, period):
    """
    使用 Numba @njit 在 CPU 上计算相对强度指标 (RSI)，使用 SMA 平滑。

    参数：
        prices (np.ndarray): 输入价格时间序列数组 (float32)。
        period (int): RSI 计算周期。

    返回：
        tuple: (rsi_out, gains, losses, rolling_sum_gains, rolling_sum_losses)。
               rsi_out 是计算结果，返回中间数组用于调试。
    抛出异常：
        ValueError: 如果周期无效或价格数据不足。
    """
    if period <= 1:
        raise ValueError("RSI 周期必须大于 1。")

    if prices.size < period:
        raise ValueError("价格数据长度不足以计算指定的 RSI 周期。")

    N = prices.size
    rsi_output_size = N - period + 1
    rsi_out = np.zeros(rsi_output_size, dtype=prices.dtype)

    # 计算价格变化 (大小 N)
    # 使用 Numba 友好的循环替代 np 切片
    changes = np.zeros(N, dtype=prices.dtype)  # 大小 N，changes[0] 为 0
    for i in range(1, N):
        changes[i] = prices[i] - prices[i - 1]

    # 分离增益和损失 (大小 N)
    gains = np.zeros(N, dtype=prices.dtype)
    losses = np.zeros(N, dtype=prices.dtype)

    for i in range(1, N):
        if changes[i] > 0:
            gains[i] = changes[i]
            losses[i] = 0.0  # 确保 losses[i] 是 float 类型 0.0
        else:
            gains[i] = 0.0  # 确保 gains[i] 是 float 类型 0.0
            losses[i] = -changes[i]  # 损失存储为正值

    # 计算平滑后的增益/损失总和 (SMA 平滑)
    rolling_sum_gains = np.zeros(rsi_output_size, dtype=prices.dtype)
    rolling_sum_losses = np.zeros(rsi_output_size, dtype=prices.dtype)

    # 计算第一个滑动求和 (k=0)
    current_gain_sum = 0.0
    current_loss_sum = 0.0

    for i in range(1, period + 1):
        current_gain_sum += gains[i]
        current_loss_sum += losses[i]

    rolling_sum_gains[0] = current_gain_sum
    rolling_sum_losses[0] = current_loss_sum

    # 计算后续滑动求和
    for k in range(1, rsi_output_size):
        current_gain_sum = current_gain_sum - gains[k] + gains[k + period]
        current_loss_sum = current_loss_sum - losses[k] + losses[k + period]
        rolling_sum_gains[k] = current_gain_sum
        rolling_sum_losses[k] = current_loss_sum

    # 计算 RSI
    for i in range(rsi_output_size):
        avg_gain = rolling_sum_gains[i] / period
        avg_loss = rolling_sum_losses[i] / period

        # 使用容差判断 avg_loss 是否接近零
        if abs(avg_loss) > ZERO_TOLERANCE:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 100.0

        rsi_out[i] = rsi

    # 返回结果和中间数组用于调试
    return rsi_out, gains, losses, rolling_sum_gains, rolling_sum_losses


# --- 主函数进行对比 ---


def main():
    # 数据规模
    N_prices = 50_000_000
    rsi_period = 14

    print(f"创建大小为 {N_prices} 的随机价格时间序列...")
    prices = np.cumsum(np.random.randn(N_prices)).astype(np.float32) + 100.0

    # --- 运行 CPU 版本 (RSI - Numba @njit) ---
    print(f"\n--- 开始 CPU 版本测试 (RSI 周期: {rsi_period}) ---")
    cpu_rsi_result = None
    cpu_total_run_time = None
    cpu_gains = None
    cpu_losses = None
    cpu_rolling_sum_gains = None
    cpu_rolling_sum_losses = None

    try:
        # 预热 Numba 函数
        if N_prices > rsi_period:
            preheat_size = min(N_prices, rsi_period + 100)
            if preheat_size > rsi_period:
                _, _, _, _, _ = calculate_rsi_cpu_njit(
                    prices[:preheat_size], rsi_period
                )
                print("CPU Numba 预热完成。")
            else:
                print("跳过 CPU Numba 预热，数据大小不足以预热。")
        else:
            print("跳过 CPU Numba 预热，价格数据不足。")

        start_run_time_cpu = time.time()
        (
            cpu_rsi_result,
            cpu_gains,
            cpu_losses,
            cpu_rolling_sum_gains,
            cpu_rolling_sum_losses,
        ) = calculate_rsi_cpu_njit(prices, rsi_period)
        end_run_time_cpu = time.time()
        cpu_total_run_time = end_run_time_cpu - start_run_time_cpu

        if cpu_rsi_result is not None and cpu_rsi_result.size > 0:
            print(
                f"CPU Numba RSI 计算总运行耗时 (含可能的少量开销): {cpu_total_run_time:.4f} 秒"
            )
        else:
            print("CPU RSI 计算失败：周期无效或价格数据不足。")

    except ValueError as e:
        print(f"CPU RSI 计算失败: {e}")
    except Exception as e:
        print(f"CPU RSI 计算发生未知错误: {e}")

    # --- 运行 GPU 版本 (RSI) (如果可用) ---
    print(f"\n--- 开始 GPU 版本测试 (RSI 周期: {rsi_period}) ---")
    gpu_rsi_result = None
    gpu_total_run_time = None
    gpu_gains = None
    gpu_losses = None
    gpu_rolling_sum_gains = None
    gpu_rolling_sum_losses = None

    if cuda.is_available():
        try:
            (
                gpu_rsi_result,
                gpu_total_run_time,
                gpu_gains,
                gpu_losses,
                gpu_rolling_sum_gains,
                gpu_rolling_sum_losses,
            ) = calculate_rsi_gpu(prices, rsi_period)

            if gpu_total_run_time is not None:
                print(f"GPU RSI 计算（总 GPU 过程）耗时: {gpu_total_run_time:.4f} 秒")

                # 验证 GPU 结果
                results_match = False
                if (
                    cpu_rsi_result is not None
                    and cpu_rsi_result.size > 0
                    and gpu_rsi_result is not None
                ):
                    print("\n--- 验证计算结果一致性 ---")
                    try:
                        np.testing.assert_allclose(
                            gpu_rsi_result,
                            cpu_rsi_result,
                            rtol=1e-4,
                            atol=1e-5,
                            verbose=True,
                        )
                        print("GPU RSI 计算结果与 CPU Numba 一致 (在容差范围内)！")
                        results_match = True
                    except AssertionError as e:
                        print(f"GPU RSI 计算结果验证失败: {e}")
                        print("结果不一致，跳过性能对比。")
                        results_match = False

                        # 诊断差异
                        print("\n--- 差异诊断 ---")
                        mismatch_indices = np.where(
                            ~np.isclose(
                                gpu_rsi_result,
                                cpu_rsi_result,
                                rtol=1e-4,
                                atol=1e-5,
                            )
                        )[0]

                        if mismatch_indices.size > 0:
                            print(
                                f"找到 {mismatch_indices.size} 个不匹配索引。前几个：{mismatch_indices[: min(5, mismatch_indices.size)]}"
                            )
                            indices_to_inspect = mismatch_indices[
                                : min(5, mismatch_indices.size)
                            ]

                            for mismatch_idx in indices_to_inspect:
                                print(
                                    f"\n检查 RSI 输出索引 {mismatch_idx} (对应价格索引 {mismatch_idx + rsi_period - 1}) 的差异："
                                )
                                print(
                                    f"CPU RSI[{mismatch_idx}]: {cpu_rsi_result[mismatch_idx]:.6f}"
                                )
                                print(
                                    f"GPU RSI[{mismatch_idx}]: {gpu_rsi_result[mismatch_idx]:.6f}"
                                )
                                print(
                                    f"CPU rolling_sum_losses[{mismatch_idx}]: {cpu_rolling_sum_losses[mismatch_idx]:.6f}"
                                )
                                print(
                                    f"GPU rolling_sum_losses[{mismatch_idx}]: {gpu_rolling_sum_losses[mismatch_idx]:.6f}"
                                )
                                print(
                                    f"CPU rolling_sum_gains[{mismatch_idx}]: {cpu_rolling_sum_gains[mismatch_idx]:.6f}"
                                )
                                print(
                                    f"GPU rolling_sum_gains[{mismatch_idx}]: {gpu_rolling_sum_gains[mismatch_idx]:.6f}"
                                )

                                cpu_avg_loss_at_mismatch = (
                                    cpu_rolling_sum_losses[mismatch_idx] / rsi_period
                                )
                                gpu_avg_loss_at_mismatch = (
                                    gpu_rolling_sum_losses[mismatch_idx] / rsi_period
                                )
                                cpu_avg_gain_at_mismatch = (
                                    cpu_rolling_sum_gains[mismatch_idx] / rsi_period
                                )
                                gpu_avg_gain_at_mismatch = (
                                    gpu_rolling_sum_gains[mismatch_idx] / rsi_period
                                )

                                print(
                                    f"CPU 平均损失[{mismatch_idx}]: {cpu_avg_loss_at_mismatch:.9f}"
                                )
                                print(
                                    f"GPU 平均损失[{mismatch_idx}]: {gpu_avg_loss_at_mismatch:.9f}"
                                )
                                print(
                                    f"CPU 平均增益[{mismatch_idx}]: {cpu_avg_gain_at_mismatch:.9f}"
                                )
                                print(
                                    f"GPU 平均增益[{mismatch_idx}]: {gpu_avg_gain_at_mismatch:.9f}"
                                )

                                print(
                                    f"CPU 平均损失接近 0.0 (atol={ZERO_TOLERANCE})? {np.isclose(cpu_avg_loss_at_mismatch, 0.0, atol=ZERO_TOLERANCE)}"
                                )
                                print(
                                    f"GPU 平均损失接近 0.0 (atol={ZERO_TOLERANCE})? {np.isclose(gpu_avg_loss_at_mismatch, 0.0, atol=ZERO_TOLERANCE)}"
                                )

                                window_start_idx_in_gains_losses = mismatch_idx + 1
                                window_end_idx_in_gains_losses = (
                                    mismatch_idx + rsi_period
                                )
                                print(
                                    f"原始增益/损失值 (大小 N) 在窗口 {window_start_idx_in_gains_losses} 到 {window_end_idx_in_gains_losses} (索引)："
                                )
                                print(
                                    f"CPU gains[{window_start_idx_in_gains_losses} : {window_end_idx_in_gains_losses + 1}]: {cpu_gains[window_start_idx_in_gains_losses : window_end_idx_in_gains_losses + 1]}"
                                )
                                print(
                                    f"GPU gains[{window_start_idx_in_gains_losses} : {window_end_idx_in_gains_losses + 1}]: {gpu_gains[window_start_idx_in_gains_losses : window_end_idx_in_gains_losses + 1]}"
                                )
                                print(
                                    f"CPU losses[{window_start_idx_in_gains_losses} : {window_end_idx_in_gains_losses + 1}]: {cpu_losses[window_start_idx_in_gains_losses : window_end_idx_in_gains_losses + 1]}"
                                )
                                print(
                                    f"GPU losses[{window_start_idx_in_gains_losses} : {window_end_idx_in_gains_losses + 1}]: {gpu_losses[window_start_idx_in_gains_losses : window_end_idx_in_gains_losses + 1]}"
                                )

                            # sys.exit(1)  # 取消注释以在验证失败后停止

                elif gpu_rsi_result is not None:
                    print("跳过 GPU 结果验证，因为 CPU Numba 结果不可用或计算失败。")
            else:
                print("GPU RSI 计算失败。")
        except Exception as e:
            print(f"GPU RSI 计算发生未知错误: {e}")
            gpu_total_run_time = None
    else:
        print("CUDA 不可用，跳过 GPU 版本测试。")

    # --- 最终性能对比总结 ---
    if (
        cpu_total_run_time is not None
        and gpu_total_run_time is not None
        and cpu_rsi_result is not None
        and cpu_rsi_result.size > 0
        and gpu_rsi_result is not None
        and results_match
    ):
        print("\n--- 性能对比总结 ---")
        print(f"CPU 总运行时间: {cpu_total_run_time:.4f} 秒")
        print(f"GPU 总运行时间: {gpu_total_run_time:.4f} 秒")
        speedup = cpu_total_run_time / gpu_total_run_time
        print(f"GPU 加速比: {speedup:.2f}x")
    else:
        if not cuda.is_available():
            print("\n无法提供最终性能对比总结，因为 CUDA 不可用。")
        elif (
            cpu_total_run_time is None
            or cpu_rsi_result is None
            or cpu_rsi_result.size == 0
        ):
            print("\n无法提供最终性能对比总结，因为 CPU Numba 计算失败或结果无效。")
        elif gpu_total_run_time is None or gpu_rsi_result is None:
            print("\n无法提供最终性能对比总结，因为 GPU 计算失败或结果无效。")
        elif not results_match:
            print("\n无法提供最终性能对比总结，因为 CPU 和 GPU 计算结果不一致。")
        else:
            print("\n性能对比未执行，原因不明。")


if __name__ == "__main__":
    main()
