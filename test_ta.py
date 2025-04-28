import numpy as np
from numba import njit, float64, float32, int32, types
import pandas as pd
import pandas_ta as ta
import time

# 定义容差
ZERO_TOLERANCE = 1e-7


# --- Helper functions for comparison ---


def _print_comparison_results(
    indicator_name,
    label,
    max_diff,
    mean_diff,
    diff_threshold,
    mismatch_indices=None,
    result1_values=None,
    result2_values=None,
):
    """Helper to print formatted comparison results."""
    print(f"{indicator_name} {label} 最大差异: {max_diff:.6f}")
    print(f"{indicator_name} {label} 平均差异: {mean_diff:.6f}")
    if max_diff < diff_threshold:
        print(
            f"{indicator_name} {label} 差异较小，{'float32 可接受' if 'float32' in label else '计算一致'}"
        )
    else:
        print(f"{indicator_name} {label} 差异较大，需检查实现")
        if (
            mismatch_indices is not None
            and mismatch_indices.size > 0
            and result1_values is not None
            and result2_values is not None
        ):
            print(
                f"找到 {mismatch_indices.size} 个差异较大索引，前几个：{mismatch_indices[:5]}"
            )
            for i, original_idx in enumerate(mismatch_indices[:5]):
                # Ensure index is within bounds of the provided value arrays
                if original_idx < len(result1_values) and original_idx < len(
                    result2_values
                ):
                    print(
                        f"索引 {original_idx}: {indicator_name} {label.split(' vs. ')[0]} = {result1_values[original_idx]:.6f}, "
                        f"{indicator_name} {label.split(' vs. ')[1]} = {result2_values[original_idx]:.6f}"
                    )


def _compare_results_f64_f32(indicator_name, result_f64, result_f32, diff_threshold):
    """Compares float64 and float32 results."""
    if result_f64.size > 0 and result_f32.size > 0:
        # Find the common length for comparison
        compare_len = min(result_f64.size, result_f32.size)
        if compare_len > 0:
            # Extract the comparable parts
            result_f64_compare = result_f64[:compare_len]
            result_f32_compare = result_f32[:compare_len]

            # 检查NaN的位置和数量是否完全相同
            if not np.array_equal(
                np.isnan(result_f64_compare), np.isnan(result_f32_compare)
            ):
                print(
                    f"{indicator_name} float64 vs. float32: NaN不匹配 - 数量或位置差异"
                )
                return False  # NaN不匹配

            # Find valid (non-NaN) indices within the comparable parts
            valid_indices = ~np.isnan(result_f64_compare) & ~np.isnan(
                result_f32_compare
            )

            if np.sum(valid_indices) > 0:
                diff = np.abs(
                    result_f64_compare[valid_indices]
                    - result_f32_compare[valid_indices]
                )
                if diff.size > 0:
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    # Find original indices where difference > threshold (within comparable length)
                    mismatch_relative_indices = np.where(
                        np.abs(result_f64_compare - result_f32_compare) > diff_threshold
                    )[0]
                    mismatch_original_indices = mismatch_relative_indices
                    _print_comparison_results(
                        indicator_name,
                        "float64 vs. float32",
                        max_diff,
                        mean_diff,
                        diff_threshold,
                        mismatch_original_indices,
                        result_f64,
                        result_f32,
                    )
                else:
                    print(f"{indicator_name} float64 vs. float32 有效差异数据不足")
            else:
                print(f"{indicator_name} float64 vs. float32 无有效数据进行对比")
        else:
            print(f"{indicator_name} float64 vs. float32 可对比长度不足")
    else:
        print(f"{indicator_name} float64 vs. float32 计算失败，数据不足或结果为空")


def _compare_results_f64_ta(indicator_name, result_f64, ta_result, diff_threshold):
    """Compares float64 and pandas-ta results."""
    if result_f64.size > 0 and ta_result.size > 0:
        # Find the first non-NaN index in both arrays
        first_valid_f64 = (
            np.argmax(~np.isnan(result_f64))
            if np.any(~np.isnan(result_f64))
            else result_f64.size
        )
        first_valid_ta = (
            np.argmax(~np.isnan(ta_result))
            if np.any(~np.isnan(ta_result))
            else ta_result.size
        )

        # Compare from the later of the two first valid indices
        start_idx_compare = max(first_valid_f64, first_valid_ta)

        # Determine the number of elements available for comparison
        if start_idx_compare < result_f64.size and start_idx_compare < ta_result.size:
            compare_len = min(
                result_f64.size - start_idx_compare, ta_result.size - start_idx_compare
            )

            if compare_len > 0:
                # Extract the comparable parts
                result_f64_compare = result_f64[
                    start_idx_compare : start_idx_compare + compare_len
                ]
                ta_result_compare = ta_result[
                    start_idx_compare : start_idx_compare + compare_len
                ]

                # 检查NaN的位置和数量是否完全相同
                if not np.array_equal(
                    np.isnan(result_f64_compare), np.isnan(ta_result_compare)
                ):
                    print(
                        f"{indicator_name} float64 vs. pandas-ta: NaN不匹配 - 数量或位置差异"
                    )
                    return False  # NaN不匹配

                # Find valid (non-NaN) indices within the comparable parts
                valid_compare_indices = ~np.isnan(result_f64_compare) & ~np.isnan(
                    ta_result_compare
                )

                if np.sum(valid_compare_indices) > 0:
                    diff_ta = np.abs(
                        result_f64_compare[valid_compare_indices]
                        - ta_result_compare[valid_compare_indices]
                    )
                    if diff_ta.size > 0:
                        max_diff_ta = np.max(diff_ta)
                        mean_diff_ta = np.mean(diff_ta)
                        # Find original indices where difference > threshold (within the aligned comparison start)
                        mismatch_relative_indices = np.where(
                            np.abs(result_f64_compare - ta_result_compare)
                            > diff_threshold
                        )[0]
                        mismatch_original_indices = (
                            start_idx_compare + mismatch_relative_indices
                        )

                        _print_comparison_results(
                            indicator_name,
                            "float64 vs. pandas-ta",
                            max_diff_ta,
                            mean_diff_ta,
                            diff_threshold,
                            mismatch_original_indices,
                            result_f64,
                            ta_result,
                        )
                    else:
                        print(
                            f"{indicator_name} float64 vs. pandas-ta 有效差异数据不足"
                        )
                else:
                    print(
                        f"{indicator_name} float64 vs. pandas-ta 对比切片中无有效数据"
                    )
            else:
                print(f"{indicator_name} float64 vs. pandas-ta 可对比长度不足")
        else:
            print(
                f"{indicator_name} float64 vs. pandas-ta 有效数据起始点问题或对比长度不足"
            )
    else:
        print(f"{indicator_name} pandas-ta 计算失败，数据不足或结果为空")


# --- Main comparison function ---
def compare_precision(
    indicator_name,
    f64_func,
    f32_func,
    data_f64,
    data_f32,
    func_args,
    ta_func=None,
    ta_kwargs=None,
    diff_threshold=1e-4,
):
    """
    比较 float64、float32 和 pandas-ta 函数的精度差异
    参数：
        indicator_name: str，指标名称（如 "RSI"）
        f64_func: function，float64 版本的自定义函数
        f32_func: function，float32 版本的自定义函数
        data_f64: list 或 np.ndarray，float64 输入数据（单数组或多数组）
        data_f32: list 或 np.ndarray，float32 输入数据
        func_args: tuple，f64/f32 函数参数（如 (period,)）
        ta_func: function，pandas-ta 函数（如 pandas_ta.rsi）
        ta_kwargs: dict，pandas-ta 函数参数（如 {'close': prices, 'length': period}）
        diff_threshold: float，差异阈值（默认 1e-4）
    返回：
        None（打印对比结果）
    """
    print(f"\n--- {indicator_name} 精度对比 ---")

    # 确保数据为列表，统一处理单/多输入
    if isinstance(data_f64, np.ndarray):
        data_f64 = [data_f64]
        data_f32 = [data_f32]

    # Calculate float64 version
    start_time = time.time()
    # Handle MACD returning multiple arrays
    if indicator_name == "MACD":
        result_f64_macd, result_f64_signal, result_f64_histogram = f64_func(
            *data_f64, *func_args
        )
        result_f64_for_comparison = result_f64_macd  # Use MACD line for comparison
    else:
        result_f64_for_comparison = f64_func(*data_f64, *func_args)
    time_f64 = time.time() - start_time

    # Calculate float32 version
    start_time = time.time()
    # Handle MACD returning multiple arrays
    if indicator_name == "MACD":
        result_f32_macd, result_f32_signal, result_f32_histogram = f32_func(
            *data_f32, *func_args
        )
        result_f32_for_comparison = result_f32_macd  # Use MACD line for comparison
    else:
        result_f32_for_comparison = f32_func(*data_f32, *func_args)
    time_f32 = time.time() - start_time

    print(f"{indicator_name} float64 计算时间: {time_f64:.4f} 秒")
    print(f"{indicator_name} float32 计算时间: {time_f32:.4f} 秒")

    # Compare float64 vs float32 results
    _compare_results_f64_f32(
        indicator_name,
        result_f64_for_comparison,
        result_f32_for_comparison,
        diff_threshold,
    )

    # Compare float64 vs pandas-ta results
    if ta_func is not None and ta_kwargs is not None:
        start_time = time.time()
        ta_result = ta_func(**ta_kwargs)  # pandas-ta returns DataFrame or Series
        time_ta = time.time() - start_time
        print(f"{indicator_name} pandas-ta 计算时间: {time_ta:.4f} 秒")

        # Process pandas-ta output for comparison
        if isinstance(ta_result, pd.Series):
            ta_result_for_comparison = ta_result.to_numpy()
        elif isinstance(ta_result, pd.DataFrame):
            # For MACD, extract the first column (MACD line)
            if indicator_name == "MACD":
                if ta_result.shape[1] > 0:
                    ta_result_for_comparison = ta_result.iloc[:, 0].to_numpy()
                else:
                    print(f"{indicator_name} pandas-ta 返回的DataFrame不包含列或为None")
                    ta_result_for_comparison = np.empty(0)
            else:  # For other indicators returning DataFrame unexpectedly, take first col
                if ta_result.shape[1] > 0:
                    ta_result_for_comparison = ta_result.iloc[:, 0].to_numpy()
                else:
                    print(f"{indicator_name} pandas-ta 返回的DataFrame不包含列或为None")
                    ta_result_for_comparison = np.empty(0)
        else:
            print(f"{indicator_name} pandas-ta 返回结果类型未知")
            ta_result_for_comparison = np.empty(0)

        _compare_results_f64_ta(
            indicator_name,
            result_f64_for_comparison,
            ta_result_for_comparison,
            diff_threshold,
        )
    else:
        print(f"{indicator_name} pandas-ta 函数未提供或参数不足")


# --- EMA CPU 版本 ---
@njit(float64[:](float64[:], int32))
def calculate_ema_cpu_f64(prices, period):
    """EMA，float64 版本"""
    if period <= 0:
        return np.empty(0, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    n = prices.size
    ema_out = np.zeros(n, dtype=np.float64)
    # EMA的第一个值通常就是第一个价格
    if n > 0:  # Check if prices array is not empty
        ema_out[0] = prices[0]
        for i in range(1, n):
            ema_out[i] = alpha * prices[i] + (1 - alpha) * ema_out[i - 1]
    return ema_out


@njit(float32[:](float32[:], int32))
def calculate_ema_cpu_f32(prices, period):
    """EMA，float32 版本"""
    if period <= 0:
        return np.empty(0, dtype=np.float32)
    alpha = 2.0 / (period + 1)
    n = prices.size
    ema_out = np.zeros(n, dtype=np.float32)
    if n > 0:  # Check if prices array is not empty
        ema_out[0] = prices[0]
        for i in range(1, n):
            ema_out[i] = alpha * prices[i] + (1 - alpha) * ema_out[i - 1]
    return ema_out


@njit(float64[:](float64[:], int32))
def calculate_rma_cpu_f64(
    series: np.ndarray[np.float64], period: int
) -> np.ndarray[np.float64]:
    """
    使用 Numba 计算 Wilder's Moving Average (RMA) (float64)。
    RMA 等同于 alpha = 1 / period 的 EMA。
    处理 NaNs：初始 NaN 会被传播，计算开始后的 NaN 输入通常会导致前一个 RMA 值被沿用。

    Args:
        series (np.ndarray[float64]): 输入数据序列。
        period (int): 平滑周期。

    Returns:
        np.ndarray[float64]: RMA 序列，与输入等长，包含初始 NaNs。
    """
    n = series.size
    if period <= 0:
        # 周期无效则返回 NaN 数组
        return np.full(n, np.nan, dtype=np.float64)

    rma_out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return rma_out  # 如果输入为空，返回空的 NaN 数组

    alpha = 1.0 / float(period)  # 明确浮点数除法
    one_minus_alpha = 1.0 - alpha
    current_rma = np.nan  # RMA 初始状态为 NaN

    for i in range(n):
        current_value = series[i]

        # 处理输入中的 NaN
        if np.isnan(current_value):
            # 如果 RMA 尚未开始计算 (仍为 NaN)，则继续传播 NaN
            if np.isnan(current_rma):
                rma_out[i] = np.nan
                continue
            # 如果 RMA 已开始计算，则沿用上一个有效的 RMA 值 (不更新 current_rma)
            else:
                rma_out[i] = current_rma
                continue

        # 更新 RMA 值
        if np.isnan(current_rma):
            # 第一个非 NaN 值直接作为 RMA 的初始值
            current_rma = current_value
        else:
            # 应用 RMA 平滑公式
            current_rma = alpha * current_value + one_minus_alpha * current_rma

        rma_out[i] = current_rma  # 存储当前计算的 RMA 值

    return rma_out


@njit(float32[:](float32[:], int32))
def calculate_rma_cpu_f32(
    series: np.ndarray[np.float32], period: int
) -> np.ndarray[np.float32]:
    """
    使用 Numba 计算 Wilder's Moving Average (RMA) (float32)。
    RMA 等同于 alpha = 1 / period 的 EMA。
    处理 NaNs：初始 NaN 会被传播，计算开始后的 NaN 输入通常会导致前一个 RMA 值被沿用。

    Args:
        series (np.ndarray[float32]): 输入数据序列。
        period (int): 平滑周期。

    Returns:
        np.ndarray[float32]: RMA 序列，与输入等长，包含初始 NaNs。
    """
    n = series.size
    if period <= 0:
        return np.full(n, np.nan, dtype=np.float32)

    rma_out = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return rma_out

    # 明确指定 float32 类型
    alpha = np.float32(1.0) / np.float32(period)
    one_minus_alpha = np.float32(1.0) - alpha
    current_rma = np.float32(np.nan)

    for i in range(n):
        current_value = series[i]

        if np.isnan(current_value):
            if np.isnan(current_rma):
                rma_out[i] = np.float32(np.nan)
                continue
            else:
                rma_out[i] = current_rma
                continue

        if np.isnan(current_rma):
            current_rma = current_value
        else:
            current_rma = alpha * current_value + one_minus_alpha * current_rma

        rma_out[i] = current_rma

    return rma_out


# --- RSI CPU 版本（尝试模仿 pandas-ta ewm(adjust=False) 初始化） ---
@njit(float64[:](float64[:], int32))
def calculate_rsi_cpu_f64(
    prices: np.ndarray[np.float64], period: int
) -> np.ndarray[np.float64]:
    """RSI，float64 版本，尝试模仿 pandas-ta ewm(adjust=False)"""
    if period <= 1 or prices.size <= period:
        return np.full(
            prices.size, np.nan, dtype=np.float64
        )  # Return full size nan array

    n = prices.size
    # RSI 输出与输入等长，包含初始 NaNs
    rsi_out = np.full(n, np.nan, dtype=np.float64)

    changes = np.zeros(n, dtype=np.float64)
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)  # Note: losses will store positive values

    # Calculate price changes and separate gains/losses
    if n > 0:
        # First change (at index 1)
        for i in range(1, n):
            changes[i] = prices[i] - prices[i - 1]

        # Separate gains and losses from changes
        # Gains are positive changes, losses are positive values of negative changes
        for i in range(1, n):
            if changes[i] > 0:
                gains[i] = changes[i]
            elif changes[i] < 0:
                losses[i] = -changes[i]  # Store absolute loss as positive

    alpha = 1.0 / period  # Alpha for Wilder's Smoothing

    # Initialize smoothed averages (mimicking ewm(adjust=False) start)
    # The smoothing starts from the first data point (index 1 of changes/gains/losses)
    # The first smoothed value is the value itself at index 1.
    # Subsequent smoothed values use the alpha formula.

    # Initialize avg_gain and avg_loss with the values at index 1 (first change)
    # These will be the starting points for the EMA calculation loop below.
    # If prices.size < 2, gains/losses at index 1 might not exist, handle this.
    avg_gain = 0.0
    avg_loss = 0.0

    # Find the first non-zero gain or loss to potentially start the smoothing
    first_valid_change_idx = -1
    for i in range(1, n):
        if changes[i] != 0:
            first_valid_change_idx = i
            break

    # If there's at least one change (i.e., n >= 2)
    if first_valid_change_idx != -1:
        # Initialize the first smoothed values with the gain/loss from the first change (at index 1)
        # Even if changes[1] is 0, we use it as the starting point for smoothing if adjust=False
        avg_gain = gains[1]
        avg_loss = losses[1]

        # Calculate smoothed averages and RSI starting from the data point after the first change (index 2)
        # The loop for smoothing and RSI calculation should iterate through the rest of the data points.
        for i in range(2, n):
            current_gain = gains[i]
            current_loss = losses[i]

            # Apply smoothing formula (alpha * current + (1 - alpha) * previous)
            avg_gain = alpha * current_gain + (1 - alpha) * avg_gain
            avg_loss = alpha * current_loss + (1 - alpha) * avg_loss

            # Calculate RSI only after 'period' changes have been processed
            # The first valid RSI will be at original data index 'period',
            # corresponding to processing changes up to original data index 'period'.
            # The loop iterates from i=2. So when i = period, we have processed changes up to index 'period'.
            if i >= period:
                # Calculate RSI. The RSI value corresponds to the data point at index i.
                # The RSI output index is i.
                # Ensure avg_gain and avg_loss are not both zero before checking rs
                if abs(avg_loss) > ZERO_TOLERANCE:
                    rs = avg_gain / avg_loss
                    rsi_out[i] = 100.0 - (100.0 / (1.0 + rs))
                elif (
                    abs(avg_gain) > ZERO_TOLERANCE
                ):  # avg_loss is zero, but avg_gain is not
                    rsi_out[i] = 100.0  # RSI is 100
                else:  # Both avg_gain and avg_loss are zero
                    rsi_out[i] = 50.0  # RSI is 50

    return rsi_out


@njit(float32[:](float32[:], int32))
def calculate_rsi_cpu_f32(
    prices: np.ndarray[np.float32], period: int
) -> np.ndarray[np.float32]:
    """RSI，float32 版本，尝试模仿 pandas-ta ewm(adjust=False) 初始化"""
    if period <= 1 or prices.size <= period:
        return np.full(
            prices.size, np.nan, dtype=np.float32
        )  # Return full size nan array

    n = prices.size
    # RSI 输出与输入等长，包含初始 NaNs
    rsi_out = np.full(n, np.nan, dtype=np.float32)

    changes = np.zeros(n, dtype=np.float32)
    gains = np.zeros(n, dtype=np.float32)
    losses = np.zeros(n, dtype=np.float32)  # Note: losses will store positive values

    # Calculate price changes and separate gains/losses
    if n > 0:
        # First change (at index 1)
        for i in range(1, n):
            changes[i] = prices[i] - prices[i - 1]

        # Separate gains and losses from changes
        # Gains are positive changes, losses are positive values of negative changes
        for i in range(1, n):
            if changes[i] > np.float32(0):
                gains[i] = changes[i]
            elif changes[i] < np.float32(0):
                losses[i] = -changes[i]  # Store absolute loss as positive

    alpha = 1.0 / period  # Alpha for Wilder's Smoothing

    # Initialize smoothed averages (mimicking ewm(adjust=False) start)
    avg_gain = np.float32(0.0)
    avg_loss = np.float32(0.0)

    # Find the first non-zero gain or loss to potentially start the smoothing
    first_valid_change_idx = -1
    for i in range(1, n):
        if changes[i] != np.float32(0):
            first_valid_change_idx = i
            break

    # If there's at least one change (i.e., n >= 2)
    if first_valid_change_idx != -1:
        # Initialize the first smoothed values with the gain/loss from the first change (at index 1)
        avg_gain = gains[1]
        avg_loss = losses[1]

        # Calculate smoothed averages and RSI starting from index 2
        for i in range(2, n):
            current_gain = gains[i]
            current_loss = losses[i]

            # Apply smoothing formula (alpha * current + (1 - alpha) * previous)
            avg_gain = alpha * current_gain + (1 - alpha) * avg_gain
            avg_loss = alpha * current_loss + (1 - alpha) * avg_loss

            # Calculate RSI only after 'period' changes have been processed
            if i >= period:
                # Calculate RSI. The RSI value corresponds to the data point at index i.
                # The RSI output index is i.
                f32_tolerance = np.float32(ZERO_TOLERANCE)  # Use float32 tolerance

                if abs(avg_loss) > f32_tolerance:
                    rs = avg_gain / avg_loss
                    rsi_out[i] = np.float32(100.0) - (
                        np.float32(100.0) / (np.float32(1.0) + rs)
                    )
                elif abs(avg_gain) > f32_tolerance:
                    rsi_out[i] = np.float32(100.0)
                else:
                    rsi_out[i] = np.float32(50.0)

    return rsi_out


# --- ATR CPU 版本 ---
@njit(float64[:](float64[:], float64[:], float64[:], int32))
def calculate_atr_cpu_f64(high, low, close, period):
    """ATR，float64 版本"""
    if period <= 0 or close.size < period:
        return np.empty(0, dtype=np.float64)
    n = close.size
    tr = np.zeros(n, dtype=np.float64)
    atr = np.zeros(n, dtype=np.float64)
    # True Range 从索引 1 开始计算
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
    # 第一个 ATR 值是前 period 个 TR 的简单平均 (从索引 1 到 period)
    # ATR 在索引 period - 1 处有第一个有效值
    if period - 1 < n:  # Ensure index is within bounds
        atr[period - 1] = np.mean(tr[1 : period + 1])
        # 后续 ATR 值使用平滑平均
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        # 返回从第一个有效ATR值开始的数组
        return atr[period - 1 :]
    else:
        return np.empty(0, dtype=np.float64)  # Not enough data for the first ATR


@njit(float32[:](float32[:], float32[:], float32[:], int32))
def calculate_atr_cpu_f32(high, low, close, period):
    """ATR，float32 版本"""
    if period <= 0 or close.size < period:
        return np.empty(0, dtype=np.float32)
    n = close.size
    tr = np.zeros(n, dtype=np.float32)
    atr = np.zeros(n, dtype=np.float32)
    # True Range 从索引 1 开始计算
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
    # 第一个 ATR 值是前 period 个 TR 的简单平均 (从索引 1 到 period)
    # ATR 在索引 period - 1 处有第一个有效值
    if period - 1 < n:  # Ensure index is within bounds
        atr[period - 1] = np.mean(tr[1 : period + 1])
        # 后续 ATR 值使用平滑平均
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        # 返回从第一个有效ATR值开始的数组
        return atr[period - 1 :]
    else:
        return np.empty(0, dtype=np.float32)  # Not enough data for the first ATR


# --- MACD CPU 版本 (修改以匹配 pandas-ta 输出结构和信号线计算) ---
# Corrected return type annotation for Numba - list return types, then input types
@njit(
    types.Tuple((float64[:], float64[:], float64[:]))(float64[:], int32, int32, int32)
)
def calculate_macd_cpu_f64(prices, fast_period, slow_period, signal_period):
    """MACD, float64 版本，返回 MACD 线, 信号线, 直方图"""
    # Check if there is enough data for the signal line calculation to have at least one valid point
    min_required_size = max(fast_period, slow_period) + signal_period - 1
    if (
        fast_period <= 0
        or slow_period <= 0
        or signal_period <= 0
        or prices.size < min_required_size
    ):
        # Return empty arrays for all three components wrapped in a tuple
        empty_arr = np.empty(0, dtype=np.float64)
        return empty_arr, empty_arr, empty_arr

    # Calculate fast and slow EMAs (full size arrays)
    ema_fast = calculate_ema_cpu_f64(prices, fast_period)
    ema_slow = calculate_ema_cpu_f64(prices, slow_period)

    # Calculate MACD line (full size array with leading invalid values)
    # MACD line has valid values from max(fast, slow) - 1
    macd_line = ema_fast - ema_slow

    # Determine the index where the MACD line first becomes valid
    macd_start_idx = max(fast_period, slow_period) - 1

    # Calculate Signal line (EMA of MACD line)
    signal_line = np.full(prices.size, np.nan, dtype=np.float64)  # Initialize with NaN
    signal_alpha = 2.0 / (signal_period + 1)

    # Find the index where the signal line first becomes valid
    # The signal line EMA starts calculating from the first valid MACD line value (macd_start_idx)
    # The first valid signal line value will be after signal_period more data points from macd_start_idx
    signal_first_valid_idx = macd_start_idx + signal_period - 1

    # Ensure there is enough data for the signal line calculation (should be covered by initial check, but defensive)
    if signal_first_valid_idx < prices.size and macd_start_idx < prices.size:
        # Calculate the first signal line value (Simple average of the first 'signal_period' valid MACD values)
        sum_macd_valid = 0.0
        # Sum the first 'signal_period' valid MACD values starting from macd_start_idx
        sum_end_idx = macd_start_idx + signal_period

        # Ensure there are enough valid MACD values to sum before summing loop
        if sum_end_idx <= prices.size and macd_start_idx < sum_end_idx:
            for i in range(macd_start_idx, sum_end_idx):
                # Ensure index i is within macd_line bounds
                sum_macd_valid += macd_line[i]

            if (
                signal_period > 0
            ):  # Avoid division by zero if signal_period is somehow 0 here
                signal_line[signal_first_valid_idx] = (
                    sum_macd_valid / signal_period
                )  # First signal is SMA of first valid MACD values
            # else: # Handle signal_period 0 edge case - signal_line remains NaN

            # Calculate subsequent signal line values using the EMA formula
            for i in range(signal_first_valid_idx + 1, prices.size):
                # Apply the smoothing formula to the MACD line values
                # Ensure previous signal is not nan (shouldn't be after first valid point)
                if not np.isnan(signal_line[i - 1]):
                    signal_line[i] = (
                        signal_alpha * macd_line[i]
                        + (1 - signal_alpha) * signal_line[i - 1]
                    )
        # else: # Not enough valid MACD values to even calculate the first signal point
        # signal_line will remain NaN

    # Calculate Histogram (valid where both MACD and Signal are valid)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# Corrected return type annotation for Numba - list return types, then input types
@njit(
    types.Tuple((float32[:], float32[:], float32[:]))(float32[:], int32, int32, int32)
)
def calculate_macd_cpu_f32(prices, fast_period, slow_period, signal_period):
    """MACD, float32 版本，返回 MACD 线, 信号线, 直方图"""
    min_required_size = max(fast_period, slow_period) + signal_period - 1
    if (
        fast_period <= 0
        or slow_period <= 0
        or signal_period <= 0
        or prices.size < min_required_size
    ):
        # Return empty arrays for all three components
        empty_arr = np.empty(0, dtype=np.float32)
        return empty_arr, empty_arr, empty_arr

    # Calculate fast and slow EMAs (full size arrays)
    ema_fast = calculate_ema_cpu_f32(prices, fast_period)
    ema_slow = calculate_ema_cpu_f32(prices, slow_period)

    # Calculate MACD line (full size array with leading invalid values)
    macd_line = ema_fast - ema_slow

    # Determine the index where the MACD line first becomes valid
    macd_start_idx = max(fast_period, slow_period) - 1

    # Calculate Signal line (EMA of MACD line)
    signal_line = np.full(prices.size, np.nan, dtype=np.float32)  # Initialize with NaN
    signal_alpha = 2.0 / (signal_period + 1)

    # Find the index where the signal line first becomes valid
    signal_first_valid_idx = macd_start_idx + signal_period - 1

    # Ensure there is enough data for the signal line calculation (should be covered by initial check, but defensive)
    if signal_first_valid_idx < prices.size and macd_start_idx < prices.size:
        # Calculate the first signal line value (Simple average of the first 'signal_period' valid MACD values)
        sum_macd_valid = 0.0
        # Sum the first 'signal_period' valid MACD values starting from macd_start_idx
        sum_end_idx = macd_start_idx + signal_period
        if sum_end_idx > prices.size:
            sum_end_idx = prices.size

        if sum_end_idx > macd_start_idx:
            for i in range(macd_start_idx, sum_end_idx):
                # Ensure index i is within macd_line bounds
                sum_macd_valid += macd_line[i]

            if (
                signal_period > 0
            ):  # Avoid division by zero if signal_period is somehow 0 here
                signal_line[signal_first_valid_idx] = (
                    sum_macd_valid / signal_period
                )  # First signal is SMA of first valid MACD values
            # else: # Handle signal_period 0 edge case - signal_line remains NaN

            # Calculate subsequent signal line values using the EMA formula
            for i in range(signal_first_valid_idx + 1, prices.size):
                # Apply the smoothing formula to the MACD line values
                signal_line[i] = (
                    signal_alpha * macd_line[i]
                    + (1 - signal_alpha) * signal_line[i - 1]
                )
        # else: # Not enough valid MACD values to even calculate the first signal point
        # signal_line will remain NaN

    # Calculate Histogram (valid where both MACD and Signal are valid)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# --- 主函数：测试所有指标 ---
def main():
    # 数据规模
    n_prices = 20_000
    rsi_period = 14
    ema_period = 20
    atr_period = 14
    macd_params = (12, 26, 9)

    print(f"生成 {n_prices} 根 K 线数据...")
    # 使用固定的随机种子以便结果可复现
    np.random.seed(42)
    prices_f64 = np.cumsum(np.random.randn(n_prices) * 0.05).astype(np.float64) + 100.0
    prices_f32 = prices_f64.astype(np.float32)
    high_f64 = prices_f64 + np.abs(np.random.randn(n_prices) * 0.1)
    low_f64 = prices_f64 - np.abs(np.random.randn(n_prices) * 0.1)
    high_f32 = high_f64.astype(np.float32)
    low_f32 = low_f64.astype(np.float32)

    # 创建 Pandas DataFrame 用于 pandas-ta
    df = pd.DataFrame({"close": prices_f64, "high": high_f64, "low": low_f64})

    # EMA 对比
    compare_precision(
        "EMA",
        calculate_ema_cpu_f64,
        calculate_ema_cpu_f32,
        prices_f64,
        prices_f32,
        (ema_period,),
        ta_func=ta.ema,
        ta_kwargs={"close": df["close"], "length": ema_period},
    )

    # RSI 对比
    compare_precision(
        "RSI",
        calculate_rsi_cpu_f64,
        calculate_rsi_cpu_f32,
        prices_f64,
        prices_f32,
        (rsi_period,),
        ta_func=ta.rsi,
        ta_kwargs={"close": df["close"], "length": rsi_period},
    )

    # ATR 对比
    compare_precision(
        "ATR",
        calculate_atr_cpu_f64,
        calculate_atr_cpu_f32,
        [high_f64, low_f64, prices_f64],
        [high_f32, low_f32, prices_f32],
        (atr_period,),
        ta_func=ta.atr,
        ta_kwargs={
            "high": df["high"],
            "low": df["low"],
            "close": df["close"],
            "length": atr_period,
        },
    )

    # MACD 对比
    # pandas-ta 的 macd 返回 DataFrame，取第一列（MACD线）进行对比
    compare_precision(
        "MACD",
        calculate_macd_cpu_f64,
        calculate_macd_cpu_f32,
        prices_f64,
        prices_f32,
        macd_params,
        ta_func=ta.macd,
        ta_kwargs={
            "close": df["close"],
            "fast": macd_params[0],
            "slow": macd_params[1],
            "signal": macd_params[2],
        },
    )


if __name__ == "__main__":
    main()
