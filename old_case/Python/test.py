# -*- coding: utf-8 -*-
from __future__ import annotations

"""
铁电单轴压缩 Twin / Single 三图分析脚本（observable diagnostics v4）
========================================================================

这版代码直接针对上一版图像“不合理”的问题重构：

为什么上一版不合理
------------------
1. 用 early-stage reference line 全程外推会被后期 strain hardening 主导，
   于是 Δσ_nl = σ_ref - σ_sm 在整个后段都变成大幅负值，
   这并不能再表示“非线性容纳”，而只是说明后期曲线比早期斜率更硬。
2. κ = E_tan / C_ref 里若 Twin 的 C_ref 很小，会把 κ 人为放大，
   导致 Twin κ > 6 这类不适合直接解释的图像。
3. P_obs vs Δσ_nl / W_nl 的 closure test 在当前数据上会因为横轴不单调、
   尺度过小或符号裁剪，产生竖线、回跳等强数值伪影，不适合作为主图。

因此，这版改成“只使用原始数据直接支持的诊断量”，避免过度依赖参考线外推。

新的三图框架
------------
Figure 1: 原始事实层
    1a) raw stress-strain
    1b) raw polarization-strain

Figure 2: 机械诊断层（全部直接来自 stress）
    2a) apparent tangent modulus
        E_tan(e) = dσ_sm / de
    2b) serration activity amplitude
        δσ_serr(e) = σ_raw(e) - σ_trend(e)
        S_rms(e)   = running RMS of δσ_serr

Figure 3: 电-机共演化层
    3a) polarization rate
        R_P(e) = dP_sm / de
    3b) normalized cumulative progression
        A_sigma(e) = ∫ S_rms(e) de
        A_P(e)     = ∫ |R_P(e)| de
        Ā_sigma    = A_sigma / A_sigma(end)
        Ā_P        = A_P / A_P(end)

每幅图的物理含义
----------------
Figure 1:
- 直接展示原始事实：Twin 是否更早出现持续 serration；Single 是否有 delayed hump。

Figure 2a:
- E_tan 是“表观切线刚度”，反映局部承载状态如何演化。
- 不再用 Twin_C_ref=13.7 之类的判据线。

Figure 2b:
- S_rms 是锯齿/小事件强度的局部幅值指标。
- 若 Twin 前期更高，说明其更早进入持续的小幅结构活动。

Figure 3a:
- R_P = dP_sm/de 是极化响应速率。
- Single hump 会对应先正后负的变化；Twin 更可能更早转负。

Figure 3b:
- Ā_sigma 和 Ā_P 只比较“进程形状”，不比较绝对量级。
- 若 Twin 的 Ā_sigma 更早上升，说明机械活动更早累积；
  若 Single 的 Ā_P 在中段才明显上升，说明极化响应存在 delayed evolution。

重要边界
--------
- 这版不反演真实 switching fraction。
- 这版不把任何量称作 intrinsic modulus。
- 这版所有诊断量都直接建立在 raw stress / raw polarization 上，
  目的是得到更稳健、更少数值伪影的文章图。
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ============================================================
# 0. 路径与文件配置
# ============================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

FILES = {
    "Single": {
        "ss": BASE_DIR / "single_ss_y_inst_2pct_cont.txt",
        "pol": BASE_DIR / "Pol_Pro_stats_overall_single.dat",
    },
    "Twin": {
        "ss": BASE_DIR / "twin_ss_y_inst_2pct_cont.txt",
        "pol": BASE_DIR / "Pol_Pro_stats_overall_twin.dat",
    },
}

OUT_DIR = BASE_DIR / "results_three_figures_observable_v4"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. 数据列配置
# ============================================================
STRAIN_COL = "s2"
STRESS_COL = "p2"
POL_COL_OVERRIDE = None
USE_ABS_POL = False

POL_CANDIDATES_SIGNED = ["Py_mean", "Py", "py", "P_y", "p_y"]
POL_CANDIDATES_ABS = ["|Py|_mean", "abs_Py_mean", "abs(Py)_mean", "abs_Py", "|Py|"]


# ============================================================
# 2. 平滑 / 局部导数参数
# ============================================================
STRESS_SMOOTH_WINDOW = 9
STRESS_TREND_WINDOW = 41
POL_SMOOTH_WINDOW = 9
TANGENT_MODULUS_WINDOW = 31
POL_RATE_WINDOW = 21
SERRATION_RMS_WINDOW = 21
POLYORDER = 2

SHOW_SMOOTH_OVERLAY_IN_FIG1 = True


# ============================================================
# 3. 绘图风格
# ============================================================
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial Unicode MS", "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "PingFang SC", "Heiti SC",
        "Arial", "Helvetica", "DejaVu Sans"
    ],
    "axes.linewidth": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "mathtext.fontset": "stix",
})

C_SINGLE = "#d62728"
C_TWIN = "#1f77b4"
C_SINGLE_GUIDE = "black"
C_TWIN_GUIDE = "gray"


# ============================================================
# 4. 工具函数
# ============================================================
def safe_odd_window(n: int, default_window: int, polyorder: int = POLYORDER) -> int:
    if n <= polyorder + 2:
        return 0
    w = min(default_window, n)
    if w % 2 == 0:
        w -= 1
    if w <= polyorder:
        w = polyorder + 3
        if w % 2 == 0:
            w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    if w <= polyorder or w < 3:
        return 0
    return w


def smooth_with_window(y: np.ndarray, window: int, polyorder: int = POLYORDER) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    w = safe_odd_window(len(y), window, polyorder)
    if w == 0:
        return y.copy()
    return savgol_filter(y, window_length=w, polyorder=polyorder, mode="interp")


def local_linear_slope(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = safe_odd_window(n, window, polyorder=1)
    if w == 0 or n < 3:
        return np.gradient(y, x)

    half = w // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs = x[lo:hi]
        ys = y[lo:hi]
        if len(xs) < 2:
            out[i] = np.nan
            continue
        x0 = xs.mean()
        out[i] = np.polyfit(xs - x0, ys, 1)[0]
    return out


def running_rms(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = safe_odd_window(n, window, polyorder=1)
    if w == 0:
        return np.abs(y)
    half = w // 2
    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        ys = y[lo:hi]
        out[i] = np.sqrt(np.mean(ys ** 2))
    return out


def cumulative_trapezoid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    if len(y) < 2:
        return out
    dx = np.diff(x)
    area = 0.5 * (y[:-1] + y[1:]) * dx
    out[1:] = np.cumsum(area)
    return out


def normalize_to_final(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return y
    final = y[-1]
    if not np.isfinite(final) or abs(final) < 1e-30:
        return np.full_like(y, np.nan)
    return y / final


# ============================================================
# 5. 数据读入与清洗
# ============================================================
def load_table(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    return pd.read_csv(file_path, sep=r"\s+", comment="#", engine="python")


def clean_stress_strain_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    if "TimeStep" not in df.columns:
        raise KeyError("应力应变文件中未找到 TimeStep 列。")
    df = df.drop_duplicates(subset=["TimeStep"], keep="first")
    df = df.sort_values("TimeStep").reset_index(drop=True)
    return df


def clean_pol_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    time_col = "TimeStep" if "TimeStep" in df.columns else "Time"
    if time_col not in df.columns:
        raise KeyError("极化文件中未找到 Time 或 TimeStep 列。")
    df = df.rename(columns={time_col: "TimeStep"})
    df = df.drop_duplicates(subset=["TimeStep"], keep="first")
    df = df.sort_values("TimeStep").reset_index(drop=True)
    return df


def resolve_polarization_column(pol_df: pd.DataFrame) -> str:
    cols = list(pol_df.columns)
    if POL_COL_OVERRIDE is not None and POL_COL_OVERRIDE in cols:
        return POL_COL_OVERRIDE
    primary = POL_CANDIDATES_ABS if USE_ABS_POL else POL_CANDIDATES_SIGNED
    secondary = POL_CANDIDATES_SIGNED if USE_ABS_POL else POL_CANDIDATES_ABS
    for c in primary:
        if c in cols:
            return c
    for c in secondary:
        if c in cols:
            warnings.warn(f"未找到首选极化列，退化使用: {c}")
            return c
    raise KeyError(f"无法在极化文件中找到匹配的极化列。可用列: {cols}")


# ============================================================
# 6. 数据准备与状态量
# ============================================================
def prepare_dataset(name: str, ss_file: Path, pol_file: Path) -> dict:
    ss = clean_stress_strain_df(load_table(ss_file))
    pol = clean_pol_df(load_table(pol_file))

    for required in ["TimeStep", STRAIN_COL, STRESS_COL]:
        if required not in ss.columns:
            raise KeyError(f"{name}: 应力-应变文件缺少列 {required}")

    pol_col = resolve_polarization_column(pol)

    ss = ss.copy()
    ss["e"] = ss[STRAIN_COL].abs()
    ss["s_raw"] = ss[STRESS_COL].abs()
    ss = ss[np.isfinite(ss["e"]) & np.isfinite(ss["s_raw"])].copy()
    ss = ss.sort_values("e").reset_index(drop=True)

    e = ss["e"].to_numpy(dtype=float)
    s_raw = ss["s_raw"].to_numpy(dtype=float)
    s_sm = smooth_with_window(s_raw, STRESS_SMOOTH_WINDOW)
    s_trend = smooth_with_window(s_raw, STRESS_TREND_WINDOW)
    E_tan = local_linear_slope(e, s_sm, TANGENT_MODULUS_WINDOW)
    delta_sigma_serr = s_raw - s_trend
    S_rms = running_rms(delta_sigma_serr, SERRATION_RMS_WINDOW)
    A_sigma = cumulative_trapezoid(e, S_rms)
    A_sigma_norm = normalize_to_final(A_sigma)

    merged_pol = pd.merge(
        ss[["TimeStep", "e"]],
        pol[["TimeStep", pol_col]],
        on="TimeStep",
        how="inner",
    ).sort_values("e").reset_index(drop=True)

    p_e = merged_pol["e"].to_numpy(dtype=float) if len(merged_pol) > 0 else np.array([])
    p_raw = merged_pol[pol_col].to_numpy(dtype=float) if len(merged_pol) > 0 else np.array([])
    p_sm = smooth_with_window(p_raw, POL_SMOOTH_WINDOW) if len(p_raw) > 0 else np.array([])
    R_P = local_linear_slope(p_e, p_sm, POL_RATE_WINDOW) if len(p_e) > 2 else np.array([])
    A_P = cumulative_trapezoid(p_e, np.abs(R_P)) if len(p_e) > 2 else np.array([])
    A_P_norm = normalize_to_final(A_P) if len(A_P) > 0 else np.array([])

    processed = pd.DataFrame({
        "TimeStep": ss["TimeStep"].to_numpy(),
        "e": e,
        "s_raw": s_raw,
        "s_sm": s_sm,
        "s_trend": s_trend,
        "E_tan": E_tan,
        "delta_sigma_serr": delta_sigma_serr,
        "S_rms": S_rms,
        "A_sigma": A_sigma,
        "A_sigma_norm": A_sigma_norm,
    })

    if len(merged_pol) > 0:
        pol_processed = pd.DataFrame({
            "TimeStep": merged_pol["TimeStep"].to_numpy(),
            "e_pol": p_e,
            "P_obs": p_raw,
            "P_sm": p_sm,
            "R_P": R_P,
            "A_P": A_P,
            "A_P_norm": A_P_norm,
        })
        processed = pd.merge(processed, pol_processed, on="TimeStep", how="left")

    processed.to_csv(OUT_DIR / f"processed_{name.lower()}.csv", index=False)

    return {
        "name": name,
        "pol_col": pol_col,
        "e": e,
        "s_raw": s_raw,
        "s_sm": s_sm,
        "s_trend": s_trend,
        "E_tan": E_tan,
        "delta_sigma_serr": delta_sigma_serr,
        "S_rms": S_rms,
        "A_sigma": A_sigma,
        "A_sigma_norm": A_sigma_norm,
        "p_e": p_e,
        "p_raw": p_raw,
        "p_sm": p_sm,
        "R_P": R_P,
        "A_P": A_P,
        "A_P_norm": A_P_norm,
        "processed": processed,
    }


# ============================================================
# 7. 作图
# ============================================================
def make_figure1(single: dict, twin: dict):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 8.0), sharex=True,
        gridspec_kw={"hspace": 0.03}, constrained_layout=True
    )

    ax1.plot(single["e"], single["s_raw"], color=C_SINGLE, lw=1.4, alpha=0.75, label="Single raw")
    ax1.plot(twin["e"], twin["s_raw"], color=C_TWIN, lw=1.4, alpha=0.75, label="Twin raw")
    if SHOW_SMOOTH_OVERLAY_IN_FIG1:
        ax1.plot(single["e"], single["s_sm"], color=C_SINGLE_GUIDE, lw=1.1, ls="--", label="Single smooth")
        ax1.plot(twin["e"], twin["s_sm"], color=C_TWIN_GUIDE, lw=1.1, ls="--", label="Twin smooth")
    ax1.set_ylabel("Stress $s$")
    ax1.set_title("Figure 1a. Raw stress-strain comparison")
    ax1.legend(ncol=2, fontsize=10)

    if len(single["p_e"]) > 0:
        ax2.plot(single["p_e"], single["p_raw"], color=C_SINGLE, lw=1.4, alpha=0.75, label="Single raw")
        ax2.plot(single["p_e"], single["p_sm"], color=C_SINGLE_GUIDE, lw=1.1, ls="--", label="Single smooth")
    if len(twin["p_e"]) > 0:
        ax2.plot(twin["p_e"], twin["p_raw"], color=C_TWIN, lw=1.4, alpha=0.75, label="Twin raw")
        ax2.plot(twin["p_e"], twin["p_sm"], color=C_TWIN_GUIDE, lw=1.1, ls="--", label="Twin smooth")
    ax2.set_xlabel("Compressive strain $e$")
    ax2.set_ylabel("Observed polarization $P_{obs}$")
    ax2.set_title("Figure 1b. Raw polarization-strain comparison")
    ax2.legend(ncol=2, fontsize=10)

    fig.savefig(OUT_DIR / "Figure1_raw_compare.png", bbox_inches="tight")


def make_figure2(single: dict, twin: dict):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 8.0), sharex=True,
        gridspec_kw={"hspace": 0.03}, constrained_layout=True
    )

    ax1.plot(single["e"], single["E_tan"], color=C_SINGLE, lw=1.8, label="Single E_tan")
    ax1.plot(twin["e"], twin["E_tan"], color=C_TWIN, lw=1.8, label="Twin E_tan")
    ax1.set_ylabel("Apparent tangent modulus")
    ax1.set_title("Figure 2a. Apparent tangent modulus from smoothed stress")
    ax1.legend(fontsize=10)

    ax2.plot(single["e"], single["S_rms"], color=C_SINGLE, lw=1.8, label="Single S_rms")
    ax2.plot(twin["e"], twin["S_rms"], color=C_TWIN, lw=1.8, label="Twin S_rms")
    ax2.set_xlabel("Compressive strain $e$")
    ax2.set_ylabel("Serration activity amplitude")
    ax2.set_title("Figure 2b. Local RMS serration activity")
    ax2.legend(fontsize=10)

    fig.savefig(OUT_DIR / "Figure2_mechanical_diagnostics.png", bbox_inches="tight")


def make_figure3(single: dict, twin: dict):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 8.0), sharex=True,
        gridspec_kw={"hspace": 0.03}, constrained_layout=True
    )

    if len(single["p_e"]) > 0:
        ax1.plot(single["p_e"], single["R_P"], color=C_SINGLE, lw=1.8, label="Single dP/de")
    if len(twin["p_e"]) > 0:
        ax1.plot(twin["p_e"], twin["R_P"], color=C_TWIN, lw=1.8, label="Twin dP/de")
    ax1.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax1.set_ylabel("Polarization rate dP/de")
    ax1.set_title("Figure 3a. Polarization response rate")
    ax1.legend(fontsize=10)

    ax2.plot(single["e"], single["A_sigma_norm"], color=C_SINGLE, lw=1.8, label="Single cumulative stress activity")
    ax2.plot(twin["e"], twin["A_sigma_norm"], color=C_TWIN, lw=1.8, label="Twin cumulative stress activity")
    if len(single["A_P_norm"]) > 0:
        ax2.plot(single["p_e"], single["A_P_norm"], color=C_SINGLE, lw=1.2, ls="--", label="Single cumulative |dP/de|")
    if len(twin["A_P_norm"]) > 0:
        ax2.plot(twin["p_e"], twin["A_P_norm"], color=C_TWIN, lw=1.2, ls="--", label="Twin cumulative |dP/de|")
    ax2.set_xlabel("Compressive strain $e$")
    ax2.set_ylabel("Normalized cumulative progression")
    ax2.set_title("Figure 3b. Cumulative progression of mechanical and polarization activity")
    ax2.legend(fontsize=9, ncol=2)

    fig.savefig(OUT_DIR / "Figure3_electromechanical_progression.png", bbox_inches="tight")


# ============================================================
# 8. 汇总输出
# ============================================================
def write_report(single: dict, twin: dict):
    rows = []
    for d in [single, twin]:
        rows.append({
            "dataset": d["name"],
            "pol_col_used": d["pol_col"],
            "max_E_tan": np.nanmax(d["E_tan"]),
            "min_E_tan": np.nanmin(d["E_tan"]),
            "max_S_rms": np.nanmax(d["S_rms"]),
            "max_abs_dPde": np.nanmax(np.abs(d["R_P"])) if len(d["R_P"]) > 0 else np.nan,
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary.csv", index=False)

    report = []
    report.append("# observable diagnostics v4\n\n")
    report.append("## Figure 2 formulas\n")
    report.append("- E_tan(e) = dσ_sm/de\n")
    report.append("- δσ_serr(e) = σ_raw(e) - σ_trend(e)\n")
    report.append("- S_rms(e) = running RMS of δσ_serr\n\n")
    report.append("## Figure 3 formulas\n")
    report.append("- R_P(e) = dP_sm/de\n")
    report.append("- A_sigma(e) = ∫ S_rms(e) de\n")
    report.append("- A_P(e) = ∫ |R_P(e)| de\n")
    report.append("- normalized cumulative curves compare progression shape only\n")
    (OUT_DIR / "report_v4.md").write_text("".join(report), encoding="utf-8")


# ============================================================
# 9. 主程序
# ============================================================
def main():
    print("=" * 72)
    print(">>> observable diagnostics v4")
    print("=" * 72)
    print(f"工作目录: {BASE_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print()

    single = prepare_dataset("Single", FILES["Single"]["ss"], FILES["Single"]["pol"])
    twin = prepare_dataset("Twin", FILES["Twin"]["ss"], FILES["Twin"]["pol"])

    make_figure1(single, twin)
    make_figure2(single, twin)
    make_figure3(single, twin)
    write_report(single, twin)

    print(f"结果已保存到: {OUT_DIR}")
    print("生成文件包括 Figure1_raw_compare.png、Figure2_mechanical_diagnostics.png、Figure3_electromechanical_progression.png、processed CSV、summary.csv 和 report_v4.md。")
    plt.show()


if __name__ == "__main__":
    main()