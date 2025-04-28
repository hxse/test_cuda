import numpy as np
from numba import njit, float64, float32, int32
import pandas as pd
import pandas_ta as ta
import time

# 定义容差
ZERO_TOLERANCE = 1e-7


# --- 更新后的精度对比工具函数 ---
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

    # 计算 float64 版本
    start_time = time.time()
    result_f64 = f64_func(*data_f64, *func_args)
    time_f64 = time.time() - start_time

    # 计算 float32 版本
    start_time = time.time()
    result_f32 = f32_func(*data_f32, *func_args)
    time_f32 = time.time() - start_time

    # float64 vs. float32 对比
    print(f"{indicator_name} float64 计算时间: {time_f64:.4f} 秒")
    print(f"{indicator_name} float32 计算时间: {time_f32:.4f} 秒")
    if result_f64.size > 0 and result_f32.size > 0:
        diff = np.abs(result_f64 - result_f32)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f"{indicator_name} float64 vs. float32 最大差异: {max_diff:.6f}")
        print(f"{indicator_name} float64 vs. float32 平均差异: {mean_diff:.6f}")
        if max_diff < diff_threshold:
            print(f"{indicator_name} float64 vs. float32 差异较小，float32 可接受")
        else:
            print(f"{indicator_name} float64 vs. float32 差异较大，建议使用 float64")
            mismatch_indices = np.where(diff > diff_threshold)[0]
            if mismatch_indices.size > 0:
                print(
                    f"找到 {mismatch_indices.size} 个差异较大索引，前几个：{mismatch_indices[:5]}"
                )
                for idx in mismatch_indices[:5]:
                    print(
                        f"索引 {idx}: {indicator_name} float64 = {result_f64[idx]:.6f}, "
                        f"{indicator_name} float32 = {result_f32[idx]:.6f}"
                    )
    else:
        print(f"{indicator_name} float64 vs. float32 计算失败，数据不足")

    # float64 vs. pandas-ta 对比
    if ta_func is not None and ta_kwargs is not None:
        start_time = time.time()
        ta_result = ta_func(**ta_kwargs)
        time_ta = time.time() - start_time
        print(f"{indicator_name} pandas-ta 计算时间: {time_ta:.4f} 秒")

        # 处理 pandas-ta 输出
        if isinstance(ta_result, pd.Series):
            ta_result = ta_result.to_numpy()
        elif isinstance(ta_result, pd.DataFrame):
            # MACD 返回多列，选择 MACD 线
            ta_result = ta_result.iloc[:, 0].to_numpy()  # 假设第一列是 MACD 线

        # 对齐长度（裁剪 NaN 或多余数据）
        # 找到第一个非NaN值的索引进行对齐
        first_valid_idx_f64 = (
            0 if result_f64.size == 0 else np.argmax(~np.isnan(result_f64))
        )
        first_valid_idx_ta = (
            0 if ta_result.size == 0 else np.argmax(~np.isnan(ta_result))
        )

        # 确定有效数据的起始点，取两者中最晚的起始点
        start_idx = max(first_valid_idx_f64, first_valid_idx_ta)

        # 从有效数据起始点到末尾进行对比
        if start_idx < result_f64.size and start_idx < ta_result.size:
            # 需要确保切片长度一致，且不超过原始数组长度
            len_f64 = result_f64.size - start_idx
            len_ta = ta_result.size - start_idx
            min_compare_len = min(len_f64, len_ta)

            if min_compare_len > 0:
                result_f64_compare = result_f64[start_idx : start_idx + min_compare_len]
                ta_result_compare = ta_result[start_idx : start_idx + min_compare_len]

                # 再次检查是否存在NaN值（理论上在起始点之后应该没有NaN了，但以防万一）
                valid_indices = ~np.isnan(result_f64_compare) & ~np.isnan(
                    ta_result_compare
                )
                result_f64_valid = result_f64_compare[valid_indices]
                ta_result_valid = ta_result_compare[valid_indices]

                if result_f64_valid.size > 0 and ta_result_valid.size > 0:
                    diff_ta = np.abs(result_f64_valid - ta_result_valid)
                    max_diff_ta = np.max(diff_ta)
                    mean_diff_ta = np.mean(diff_ta)
                    print(
                        f"{indicator_name} float64 vs. pandas-ta 最大差异: {max_diff_ta:.6f}"
                    )
                    print(
                        f"{indicator_name} float64 vs. pandas-ta 平均差异: {mean_diff_ta:.6f}"
                    )
                    if max_diff_ta < diff_threshold:
                        print(
                            f"{indicator_name} float64 vs. pandas-ta 差异较小，计算一致"
                        )
                    else:
                        print(
                            f"{indicator_name} float64 vs. pandas-ta 差异较大，需检查实现"
                        )
                        # 找到差异较大的原始索引
                        mismatch_relative_indices = np.where(diff_ta > diff_threshold)[
                            0
                        ]
                        if mismatch_relative_indices.size > 0:
                            # 将相对索引转换回原始数组的索引
                            mismatch_original_indices = (
                                np.where(
                                    (
                                        np.abs(result_f64_compare - ta_result_compare)
                                        > diff_threshold
                                    )
                                )[0]
                                + start_idx
                            )  # 使用原始切片做对比找索引更直接

                            print(
                                f"找到 {mismatch_original_indices.size} 个差异较大索引，前几个：{mismatch_original_indices[:5]}"
                            )
                            # 显示原始索引和对应的值
                            for idx in mismatch_original_indices[:5]:
                                # 需要获取原始的result_f64和ta_result的值，而不是切片后的
                                print(
                                    f"索引 {idx}: {indicator_name} float64 = {result_f64[idx]:.6f}, "
                                    f"{indicator_name} pandas-ta = {ta_result[idx]:.6f}"
                                )
                else:
                    print(f"{indicator_name} float64 vs. pandas-ta 有效对比数据不足")
            else:
                print(f"{indicator_name} float64 vs. pandas-ta 对比长度不足")
        else:
            print(f"{indicator_name} float64 vs. pandas-ta 有效数据起始点问题")

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
    ema_out[0] = prices[0]
    for i in range(1, n):
        ema_out[i] = alpha * prices[i] + (1 - alpha) * ema_out[i - 1]
    return ema_out


# --- RSI CPU 版本（平滑平均，调整初始化） ---
@njit(float64[:](float64[:], int32))
def calculate_rsi_cpu_f64(prices, period):
    """RSI，float64 版本，使用平滑平均，调整初始化"""
    if period <= 1 or prices.size <= period:
        return np.empty(0, dtype=np.float64)
    n = prices.size
    # RSI 输出长度为 n - period
    rsi_out = np.zeros(n - period, dtype=np.float64)
    changes = np.zeros(n, dtype=np.float64)
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)

    # 计算价格变化
    for i in range(1, n):
        changes[i] = prices[i] - prices[i - 1]
        if changes[i] > 0:
            gains[i] = changes[i]
        else:
            losses[i] = -changes[i]

    # 初始化平滑平均增益和损失为0
    avg_gain = 0.0
    avg_loss = 0.0

    # 从第一个变化量开始应用平滑公式，并在数据足够时计算RSI
    for i in range(1, n):
        # 应用平滑公式
        # 注意这里的 avg_gain 和 avg_loss 在每次迭代中都会更新
        # 相当于从0开始累积平滑效应
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        # 在收集到足够的数据（period周期）后开始计算RSI
        if i >= period:
            # 计算RS
            if abs(avg_loss) > ZERO_TOLERANCE:
                rs = avg_gain / avg_loss
                # 计算RSI并存储到输出数组，索引 i 对应 rsi_out 中的索引 i - period
                rsi_out[i - period] = 100.0 - (100.0 / (1.0 + rs))
            else:
                # 如果 avg_loss 接近于零
                rsi_out[i - period] = (
                    100.0
                    if avg_gain > ZERO_TOLERANCE
                    else (0.0 if avg_gain < -ZERO_TOLERANCE else 50.0)
                )  # 当avg_gain和avg_loss都接近0时，RSI倾向于50

    return rsi_out


@njit(float32[:](float32[:], int32))
def calculate_rsi_cpu_f32(prices, period):
    """RSI，float32 版本，使用平滑平均，调整初始化"""
    if period <= 1 or prices.size <= period:
        return np.empty(0, dtype=np.float32)
    n = prices.size
    # RSI 输出长度为 n - period
    rsi_out = np.zeros(n - period, dtype=np.float32)
    changes = np.zeros(n, dtype=np.float32)
    gains = np.zeros(n, dtype=np.float32)
    losses = np.zeros(n, dtype=np.float32)

    # 计算价格变化
    for i in range(1, n):
        changes[i] = prices[i] - prices[i - 1]
        if changes[i] > 0:
            gains[i] = changes[i]
        else:
            losses[i] = -changes[i]

    # 初始化平滑平均增益和损失为0
    avg_gain = 0.0
    avg_loss = 0.0

    # 从第一个变化量开始应用平滑公式，并在数据足够时计算RSI
    for i in range(1, n):
        # 应用平滑公式
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        # 在收集到足够的数据（period周期）后开始计算RSI
        if i >= period:
            # 计算RS
            if abs(avg_loss) > ZERO_TOLERANCE:
                rs = avg_gain / avg_loss
                # 计算RSI并存储到输出数组，索引 i 对应 rsi_out 中的索引 i - period
                rsi_out[i - period] = 100.0 - (100.0 / (1.0 + rs))
            else:
                # 如果 avg_loss 接近于零
                rsi_out[i - period] = (
                    100.0
                    if avg_gain > ZERO_TOLERANCE
                    else (0.0 if avg_gain < -ZERO_TOLERANCE else 50.0)
                )  # 当avg_gain和avg_loss都接近0时，RSI倾向于50

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
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
    atr[period - 1] = np.mean(tr[1 : period + 1])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


@njit(float32[:](float32[:], float32[:], float32[:], int32))
def calculate_atr_cpu_f32(high, low, close, period):
    """ATR，float32 版本"""
    if period <= 0 or close.size < period:
        return np.empty(0, dtype=np.float32)
    n = close.size
    tr = np.zeros(n, dtype=np.float32)
    atr = np.zeros(n, dtype=np.float32)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
    atr[period - 1] = np.mean(tr[1 : period + 1])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# --- MACD CPU 版本 ---
@njit(float64[:](float64[:], int32, int32, int32))
def calculate_macd_cpu_f64(prices, fast_period, slow_period, signal_period):
    """MACD，float64 版本，返回 MACD 线"""
    if (
        fast_period <= 0
        or slow_period <= 0
        or signal_period <= 0
        or prices.size < max(fast_period, slow_period, signal_period)
    ):
        return np.empty(0, dtype=np.float64)
    ema_fast = calculate_ema_cpu_f64(prices, fast_period)
    ema_slow = calculate_ema_cpu_f64(prices, slow_period)
    # MACD 线从 slow_period - 1 索引开始有有效值（取决于EMA实现）
    # 我们只返回两个EMA计算完成后，MACD线有值的区域
    start_idx = max(fast_period, slow_period) - 1
    if start_idx >= prices.size:
        return np.empty(0, dtype=np.float64)
    macd_line = ema_fast[start_idx:] - ema_slow[start_idx:]
    # 返回MACD线，注意这里没有计算信号线并返回
    return macd_line


@njit(float32[:](float32[:], int32, int32, int32))
def calculate_macd_cpu_f32(prices, fast_period, slow_period, signal_period):
    """MACD，float32 版本，返回 MACD 线"""
    if (
        fast_period <= 0
        or slow_period <= 0
        or signal_period <= 0
        or prices.size < max(fast_period, slow_period, signal_period)
    ):
        return np.empty(0, dtype=np.float32)
    ema_fast = calculate_ema_cpu_f32(prices, fast_period)
    ema_slow = calculate_ema_cpu_f32(prices, slow_period)
    # MACD 线从 slow_period - 1 索引开始有有效值（取决于EMA实现）
    # 我们只返回两个EMA计算完成后，MACD线有值的区域
    start_idx = max(fast_period, slow_period) - 1
    if start_idx >= prices.size:
        return np.empty(0, dtype=np.float32)
    macd_line = ema_fast[start_idx:] - ema_slow[start_idx:]
    # 返回MACD线，注意这里没有计算信号线并返回
    return macd_line


# --- 主函数：测试所有指标 ---
def main():
    # 数据规模
    n_prices = 10_000
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
