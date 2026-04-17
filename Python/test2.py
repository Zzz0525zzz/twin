# -*- coding: utf-8 -*-
from __future__ import annotations

"""
BaTiO3 Twin / Single 补充分析脚本
=================================

本脚本不是替代你现有的 observable diagnostics v4，
而是专门补充它没有做的内容：

1. 三维整体极化 discrimination
   - P_parallel = <Py>
   - <|P_parallel|> = <|Py|>
   - <|P|>
   - |<P>|
   - chi_parallel = |<Py>| / <|Py|>
   - chi_vec      = |<P>|_net / <|P|>
   - eta_perp_comp = (<|Px|> + <|Pz|>) / <|P|>
   - 上述量的导数与累计量

2. quick section 异质性分析
   - 使用 stats_sections.dat 中的 all / xy / xz / yz 统计
   - 计算各中截面的 P_parallel、chi_parallel
   - 计算 quick heterogeneity H_sec_quick
   - 计算 quick onset strain

3. 机械-极化耦合 state table
   - 继承机械侧 E_tan, delta_sigma_serr, S_rms, A_sigma
   - 与整体极化指标对齐到同一 Time/strain 轴

输出
----
1. results_supplemental_discrimination/
   - state_table_single.csv
   - state_table_twin.csv
   - section_quick_single.csv
   - section_quick_twin.csv
   - summary_supplemental.csv
   - report_supplemental.md
   - Figure4_discrimination_single_vs_twin.png
   - Figure5_quick_sections.png
   - Figure6_qc_and_onset.png

注意
----
1. 本脚本默认压缩轴是 y，因此 P_parallel = Py。
2. 本脚本优先按 Time/TimeStep 对齐机械与极化；若失败，则退化为按行序/归一化进程对齐。
3. 本脚本只做整体 discrimination + quick section，不做 ALL.plt 局域 proxy 分类。
4. 这版新增量都属于 diagnostics，不应称为 true wall / true bulk / true switching fraction。
"""

import io
import re
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
        "overall": BASE_DIR / "Pol_Pro_stats_overall_single.dat",
        "sections": BASE_DIR / "Pol_Pro_stats_sections_single.dat",
    },
    "Twin": {
        "ss": BASE_DIR / "twin_ss_y_inst_2pct_cont.txt",
        "overall": BASE_DIR / "Pol_Pro_stats_overall_twin.dat",
        "sections": BASE_DIR / "Pol_Pro_stats_sections_twin.dat",
    },
}

OUT_DIR = BASE_DIR / "results_supplemental_discrimination"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. 数据列配置
# ============================================================
STRAIN_COL = "s2"
STRESS_COL = "p2"
TIME_COL_CANDIDATES_MECH = ["TimeStep", "Time", "Step", "step"]
TIME_COL_CANDIDATES_POL = ["TimeStep", "Time", "time", "step"]

LOAD_AXIS = "y"  # 当前项目默认 y 向压缩

# ============================================================
# 2. 平滑 / 导数 / 窗口参数
# ============================================================
STRESS_SMOOTH_WINDOW = 9
STRESS_TREND_WINDOW = 41
TANGENT_MODULUS_WINDOW = 31
SERRATION_RMS_WINDOW = 21

GLOBAL_SMOOTH_WINDOW = 11
GLOBAL_RATE_WINDOW = 21
POLYORDER = 2

SECTION_SMOOTH_WINDOW = 9
ONSET_ALPHA = 0.10
EPS = 1e-12

# ============================================================
# 3. 绘图风格
# ============================================================
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10.5
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial Unicode MS", "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "PingFang SC", "Heiti SC",
        "Arial", "Helvetica", "DejaVu Sans"
    ],
    "axes.linewidth": 1.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "mathtext.fontset": "stix",
})

C_SINGLE = "#d62728"
C_TWIN = "#1f77b4"
C_ALL = "black"
C_XY = "#2ca02c"
C_XZ = "#9467bd"
C_YZ = "#8c564b"

# ============================================================
# 4. 通用工具函数
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


def onset_from_cumulative(x: np.ndarray, A: np.ndarray, alpha: float = ONSET_ALPHA) -> float:
    x = np.asarray(x, dtype=float)
    A = np.asarray(A, dtype=float)
    if len(x) == 0 or len(A) == 0:
        return np.nan
    idx = np.where(A >= alpha)[0]
    return float(x[idx[0]]) if len(idx) > 0 else np.nan


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def clean_name(name: str) -> str:
    """将 VARIABLES 表头规范化为稳定列名。"""
    out = str(name).strip()

    # 正确处理 |Px|_mean -> absPx_mean，而不是 absPxabs_mean
    out = re.sub(r"\|\s*([^|]+?)\s*\|", lambda m: f"abs{m.group(1).strip()}", out)

    out = out.replace("(", "").replace(")", "")
    out = out.replace("/", "_")
    out = out.replace("-", "_")
    out = out.replace(" ", "")
    out = out.replace("<", "").replace(">", "")
    return out

# ============================================================
# 5. 读入函数：兼容普通表格 / VARIABLES= 表格
# ============================================================
def load_whitespace_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    return pd.read_csv(path, sep=r"\s+", comment="#", engine="python")


def load_variables_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()

    if first.strip().startswith("VARIABLES"):
        cols = [clean_name(x) for x in re.findall(r'"([^"]+)"', first)]
        df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, names=cols, engine="python")
    else:
        df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        df.columns = [clean_name(c) for c in df.columns]

    # 兼容旧版错误清洗名
    rename_map = {
        "absPxabs_mean": "absPx_mean",
        "absPyabs_mean": "absPy_mean",
        "absPzabs_mean": "absPz_mean",
        "absPabs_mean": "absP_mean",

        "absPxabs_mean_all": "absPx_mean_all",
        "absPyabs_mean_all": "absPy_mean_all",
        "absPzabs_mean_all": "absPz_mean_all",

        "absPxabs_mean_xy": "absPx_mean_xy",
        "absPyabs_mean_xy": "absPy_mean_xy",
        "absPzabs_mean_xy": "absPz_mean_xy",

        "absPxabs_mean_xz": "absPx_mean_xz",
        "absPyabs_mean_xz": "absPy_mean_xz",
        "absPzabs_mean_xz": "absPz_mean_xz",

        "absPxabs_mean_yz": "absPx_mean_yz",
        "absPyabs_mean_yz": "absPy_mean_yz",
        "absPzabs_mean_yz": "absPz_mean_yz",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df

# ============================================================
# 6. 清洗机械数据
# ============================================================
def clean_stress_strain_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

    time_col = find_first_existing_column(df, TIME_COL_CANDIDATES_MECH)
    if time_col is None:
        warnings.warn(f"{name}: 机械文件未找到 TimeStep/Time，后续将退化为按行序或进程对齐。")
        df["_row_id_"] = np.arange(len(df))
    else:
        df = df.rename(columns={time_col: "Time"})
        df["Time"] = df["Time"].astype(float)  # 强制统一转换为 float，解决 merge 时的类型冲突
        df = df.drop_duplicates(subset=["Time"], keep="first")
        df = df.sort_values("Time").reset_index(drop=True)

    if STRAIN_COL not in df.columns or STRESS_COL not in df.columns:
        raise KeyError(f"{name}: 机械文件缺少列 {STRAIN_COL} 或 {STRESS_COL}")

    # 统一压缩为正向量，便于比较
    s = pd.to_numeric(df[STRAIN_COL], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df[STRESS_COL], errors="coerce").to_numpy(float)

    e = np.abs(s)
    sigma = np.abs(p)

    out = pd.DataFrame({
        "strain": e,
        "stress_raw": sigma,
    })
    if "Time" in df.columns:
        out["Time"] = df["Time"].to_numpy()
    else:
        out["_row_id_"] = df["_row_id_"].to_numpy()

    out = out[np.isfinite(out["strain"]) & np.isfinite(out["stress_raw"])].copy()
    if "Time" in out.columns:
        out = out.sort_values("Time").reset_index(drop=True)
    else:
        out = out.sort_values("strain").reset_index(drop=True)
    return out

# ============================================================
# 7. 清洗整体 / section 极化统计
# ============================================================
def clean_pol_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

    time_col = find_first_existing_column(df, TIME_COL_CANDIDATES_POL)
    if time_col is None:
        raise KeyError(f"{name}: 极化文件中未找到 Time 或 TimeStep 列")
    df = df.rename(columns={time_col: "Time"})
    df["Time"] = df["Time"].astype(float)  # 强制统一转换为 float
    df = df.drop_duplicates(subset=["Time"], keep="first")
    df = df.sort_values("Time").reset_index(drop=True)
    return df


def require_columns(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name}: 缺少必要列 {missing}；可用列为 {list(df.columns)}")

# ============================================================
# 8. 机械诊断
# ============================================================
def add_mech_metrics(mech: pd.DataFrame) -> pd.DataFrame:
    out = mech.copy()
    e = out["strain"].to_numpy(dtype=float)
    s_raw = out["stress_raw"].to_numpy(dtype=float)

    out["stress_sm"] = smooth_with_window(s_raw, STRESS_SMOOTH_WINDOW)
    out["stress_trend"] = smooth_with_window(s_raw, STRESS_TREND_WINDOW)
    out["E_tan"] = local_linear_slope(e, out["stress_sm"].to_numpy(), TANGENT_MODULUS_WINDOW)
    out["delta_sigma_serr"] = s_raw - out["stress_trend"].to_numpy()
    out["S_rms"] = running_rms(out["delta_sigma_serr"].to_numpy(), SERRATION_RMS_WINDOW)
    A_sigma = cumulative_trapezoid(e, out["S_rms"].to_numpy())
    out["A_sigma"] = A_sigma
    out["A_sigma_norm"] = normalize_to_final(A_sigma)
    out["onset_sigma"] = onset_from_cumulative(e, out["A_sigma_norm"].to_numpy())
    return out

# ============================================================
# 9. 对齐：优先 Time，其次 row，最后 progress
# ============================================================
def align_mech_to_pol(mech: pd.DataFrame, pol: pd.DataFrame) -> pd.DataFrame:
    mech = mech.copy()
    pol = pol.copy()

    if "Time" in mech.columns and "Time" in pol.columns:
        mech["Time"] = mech["Time"].astype(float)  # 二重保险，防止 merge_asof 报错
        pol["Time"] = pol["Time"].astype(float)
        out = pd.merge_asof(
            pol.sort_values("Time"),
            mech.sort_values("Time"),
            on="Time",
            direction="nearest",
        )
        out["align_method"] = "time"
        return out

    if len(mech) == len(pol):
        out = pol.copy()
        out["strain"] = mech["strain"].to_numpy()
        out["stress_raw"] = mech["stress_raw"].to_numpy()
        for c in ["stress_sm", "stress_trend", "E_tan", "delta_sigma_serr", "S_rms", "A_sigma", "A_sigma_norm"]:
            if c in mech.columns:
                out[c] = mech[c].to_numpy()
        out["align_method"] = "row"
        return out

    # 退化：按归一化进程对齐
    u_mech = np.linspace(0.0, 1.0, len(mech))
    u_pol = np.linspace(0.0, 1.0, len(pol))
    out = pol.copy()
    out["strain"] = np.interp(u_pol, u_mech, mech["strain"].to_numpy())
    for c in ["stress_raw", "stress_sm", "stress_trend", "E_tan", "delta_sigma_serr", "S_rms", "A_sigma", "A_sigma_norm"]:
        if c in mech.columns:
            out[c] = np.interp(u_pol, u_mech, mech[c].to_numpy())
    out["align_method"] = "progress"
    return out

# ============================================================
# 10. 整体 discrimination
# ============================================================
def add_global_discrimination(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    required = [
        "Px_mean", "Py_mean", "Pz_mean",
        "absPx_mean", "absPy_mean", "absPz_mean", "absP_mean",
    ]
    require_columns(out, required, name)

    out["P_parallel"] = out["Py_mean"]
    out["Pabs_parallel_mean"] = out["absPy_mean"]
    out["Pmag_mean"] = out["absP_mean"]
    out["Pvec_norm"] = np.sqrt(out["Px_mean"]**2 + out["Py_mean"]**2 + out["Pz_mean"]**2)

    out["chi_parallel"] = np.abs(out["P_parallel"]) / (out["Pabs_parallel_mean"] + EPS)
    out["chi_vec"] = out["Pvec_norm"] / (out["Pmag_mean"] + EPS)

    # 快速横向泄露 proxy：只用整体统计即可构造
    out["eta_perp_comp"] = (out["absPx_mean"] + out["absPz_mean"]) / (out["Pmag_mean"] + EPS)

    # 平滑、导数、累计量
    e = out["strain"].to_numpy(dtype=float)
    for col in [
        "P_parallel", "Pabs_parallel_mean", "Pmag_mean",
        "Pvec_norm", "chi_parallel", "chi_vec", "eta_perp_comp",
    ]:
        sm = smooth_with_window(out[col].to_numpy(dtype=float), GLOBAL_SMOOTH_WINDOW)
        out[f"{col}_sm"] = sm
        rate = local_linear_slope(e, sm, GLOBAL_RATE_WINDOW)
        out[f"d{col}_de"] = rate
        A = cumulative_trapezoid(e, np.abs(rate))
        out[f"A_{col}"] = A
        out[f"A_{col}_norm"] = normalize_to_final(A)
        out[f"onset_{col}"] = onset_from_cumulative(e, out[f"A_{col}_norm"].to_numpy())

    return out

# ============================================================
# 11. quick section
# ============================================================
def build_quick_section_table(sec_df: pd.DataFrame, mech_state: pd.DataFrame, name: str) -> pd.DataFrame:
    sec_df = clean_pol_df(sec_df, f"{name} quick sections")

    needed = [
        "Py_mean_xy", "Py_mean_xz", "Py_mean_yz",
        "absPy_mean_xy", "absPy_mean_xz", "absPy_mean_yz",
        "Pmag_mean_xy", "Pmag_mean_xz", "Pmag_mean_yz",
    ]
    require_columns(sec_df, needed, f"{name} quick sections")

    out = align_mech_to_pol(mech_state, sec_df)

    rows = []
    for sec in ["xy", "xz", "yz"]:
        tmp = pd.DataFrame({
            "dataset": name,
            "Time": out["Time"],
            "section": sec,
            "strain": out["strain"],
            "P_parallel": out[f"Py_mean_{sec}"],
            "Pabs_parallel_mean": out[f"absPy_mean_{sec}"],
            "Pmag_mean": out[f"Pmag_mean_{sec}"],
        })
        tmp["chi_parallel"] = np.abs(tmp["P_parallel"]) / (tmp["Pabs_parallel_mean"] + EPS)

        # 平滑、导数、累计量与 onset
        e = tmp["strain"].to_numpy(dtype=float)
        for col in ["P_parallel", "chi_parallel"]:
            sm = smooth_with_window(tmp[col].to_numpy(dtype=float), SECTION_SMOOTH_WINDOW)
            tmp[f"{col}_sm"] = sm
            rate = local_linear_slope(e, sm, GLOBAL_RATE_WINDOW)
            tmp[f"d{col}_de"] = rate
            A = cumulative_trapezoid(e, np.abs(rate))
            tmp[f"A_{col}"] = A
            tmp[f"A_{col}_norm"] = normalize_to_final(A)
            tmp[f"onset_{col}"] = onset_from_cumulative(e, tmp[f"A_{col}_norm"].to_numpy())
        rows.append(tmp)

    quick = pd.concat(rows, axis=0, ignore_index=True)

    # 计算每个时刻跨 xy/xz/yz 的 quick heterogeneity
    het_rows = []
    for t, grp in quick.groupby("Time", sort=True):
        vals = grp["P_parallel"].to_numpy(dtype=float)
        chi_vals = grp["chi_parallel"].to_numpy(dtype=float)
        e_val = float(grp["strain"].iloc[0])
        h_quick = (np.nanmax(vals) - np.nanmin(vals)) / (np.nanmean(np.abs(vals)) + EPS)
        h_chi = (np.nanmax(chi_vals) - np.nanmin(chi_vals)) / (np.nanmean(np.abs(chi_vals)) + EPS)
        het_rows.append({
            "dataset": name,
            "Time": t,
            "strain": e_val,
            "H_sec_quick": h_quick,
            "H_sec_chi_quick": h_chi,
            "P_env_min": np.nanmin(vals),
            "P_env_max": np.nanmax(vals),
            "chi_env_min": np.nanmin(chi_vals),
            "chi_env_max": np.nanmax(chi_vals),
        })
    het = pd.DataFrame(het_rows).sort_values("Time").reset_index(drop=True)

    for col in ["H_sec_quick", "H_sec_chi_quick"]:
        sm = smooth_with_window(het[col].to_numpy(dtype=float), SECTION_SMOOTH_WINDOW)
        het[f"{col}_sm"] = sm
        rate = local_linear_slope(het["strain"].to_numpy(dtype=float), sm, GLOBAL_RATE_WINDOW)
        het[f"d{col}_de"] = rate
        A = cumulative_trapezoid(het["strain"].to_numpy(dtype=float), np.abs(rate))
        het[f"A_{col}"] = A
        het[f"A_{col}_norm"] = normalize_to_final(A)
        het[f"onset_{col}"] = onset_from_cumulative(het["strain"].to_numpy(dtype=float), het[f"A_{col}_norm"].to_numpy())

    quick = quick.merge(het, on=["dataset", "Time", "strain"], how="left")
    return quick

# ============================================================
# 12. 数据集准备
# ============================================================
def prepare_dataset(name: str, cfg: dict) -> dict:
    mech_raw = load_whitespace_table(cfg["ss"])
    mech = clean_stress_strain_df(mech_raw, f"{name} mechanical")
    mech = add_mech_metrics(mech)

    overall_raw = load_variables_table(cfg["overall"])
    overall = clean_pol_df(overall_raw, f"{name} overall")
    overall_state = align_mech_to_pol(mech, overall)
    overall_state = add_global_discrimination(overall_state, f"{name} overall")
    overall_state["dataset"] = name

    sections_path = cfg.get("sections", None)
    quick_sections = None
    if sections_path is not None and Path(sections_path).exists():
        sec_raw = load_variables_table(Path(sections_path))
        quick_sections = build_quick_section_table(sec_raw, mech, name)
    else:
        warnings.warn(f"{name}: 未找到 section 文件，跳过 quick section 分析。")

    overall_state.to_csv(OUT_DIR / f"state_table_{name.lower()}.csv", index=False)
    if quick_sections is not None:
        quick_sections.to_csv(OUT_DIR / f"section_quick_{name.lower()}.csv", index=False)

    return {
        "name": name,
        "mech": mech,
        "state": overall_state,
        "quick_sections": quick_sections,
    }

# ============================================================
# 13. 作图：Figure 4 discrimination
# ============================================================
def make_figure4(single: dict, twin: dict):
    """简化版 discrimination 图：只保留最有信息量的三条主线。"""
    s = single["state"]
    t = twin["state"]

    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 8.8), sharex=True,
        gridspec_kw={"hspace": 0.05}, constrained_layout=True
    )

    # 4a: 净平行极化
    axes[0].plot(s["strain"], s["P_parallel_sm"], color=C_SINGLE, lw=2.0, label="Single P_parallel")
    axes[0].plot(t["strain"], t["P_parallel_sm"], color=C_TWIN, lw=2.0, label="Twin P_parallel")
    axes[0].set_ylabel("P_parallel")
    axes[0].set_title("Figure 4a. Net parallel polarization")
    axes[0].legend(fontsize=9, ncol=2)

    # 4b: 平行方向占有量
    axes[1].plot(s["strain"], s["Pabs_parallel_mean_sm"], color=C_SINGLE, lw=2.0,
                 label="Single <|P_parallel|>")
    axes[1].plot(t["strain"], t["Pabs_parallel_mean_sm"], color=C_TWIN, lw=2.0,
                 label="Twin <|P_parallel|>")
    axes[1].set_ylabel("<|P_parallel|>")
    axes[1].set_title("Figure 4b. Parallel occupancy diagnostic")
    axes[1].legend(fontsize=9, ncol=2)

    # 4c: 只保留最关键的 chi_parallel
    axes[2].plot(s["strain"], s["chi_parallel"], color=C_SINGLE, lw=2.0, label="Single chi_parallel")
    axes[2].plot(t["strain"], t["chi_parallel"], color=C_TWIN, lw=2.0, label="Twin chi_parallel")
    axes[2].set_xlabel("Compressive strain e")
    axes[2].set_ylabel("chi_parallel")
    axes[2].set_title("Figure 4c. Parallel-channel coherence / cancellation proxy")
    axes[2].legend(fontsize=9, ncol=2)

    fig.savefig(OUT_DIR / "Figure4_discrimination_single_vs_twin.png", bbox_inches="tight")

# ============================================================
# 14. 作图：Figure 5 quick sections
# ============================================================
def make_figure5(single: dict, twin: dict):
    """简化版 quick section 图：保留截面主线和异质性，不再画截面 chi。"""
    if single["quick_sections"] is None or twin["quick_sections"] is None:
        return

    qs = single["quick_sections"]
    qt = twin["quick_sections"]

    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 8.8), sharex=True,
        gridspec_kw={"hspace": 0.05}, constrained_layout=True
    )

    color_map = {"xy": C_XY, "xz": C_XZ, "yz": C_YZ}

    # 5a Single 三个代表性中截面
    for sec in ["xy", "xz", "yz"]:
        g = qs[qs["section"] == sec]
        axes[0].plot(g["strain"], g["P_parallel"], color=color_map[sec], lw=1.8, label=f"Single {sec}")
    axes[0].set_ylabel("P_parallel^(s)")
    axes[0].set_title("Figure 5a. Single quick sections: net parallel polarization")
    axes[0].legend(fontsize=9, ncol=3)

    # 5b Twin 三个代表性中截面
    for sec in ["xy", "xz", "yz"]:
        g = qt[qt["section"] == sec]
        axes[1].plot(g["strain"], g["P_parallel"], color=color_map[sec], lw=1.8, label=f"Twin {sec}")
    axes[1].set_ylabel("P_parallel^(s)")
    axes[1].set_title("Figure 5b. Twin quick sections: net parallel polarization")
    axes[1].legend(fontsize=9, ncol=3)

    # 5c 只保留平滑后的 quick heterogeneity 主线
    hs = qs[["Time", "strain", "H_sec_quick_sm"]].drop_duplicates().sort_values("Time")
    ht = qt[["Time", "strain", "H_sec_quick_sm"]].drop_duplicates().sort_values("Time")
    axes[2].plot(hs["strain"], hs["H_sec_quick_sm"], color=C_SINGLE, lw=2.0, label="Single")
    axes[2].plot(ht["strain"], ht["H_sec_quick_sm"], color=C_TWIN, lw=2.0, label="Twin")
    axes[2].set_xlabel("Compressive strain e")
    axes[2].set_ylabel("H_sec_quick")
    axes[2].set_title("Figure 5c. Quick section heterogeneity")
    axes[2].legend(fontsize=9, ncol=2)

    fig.savefig(OUT_DIR / "Figure5_quick_sections.png", bbox_inches="tight")

# ============================================================
# 15. 作图：Figure 6 QC + onset
# ============================================================
def make_figure6(single: dict, twin: dict):
    """简化版 progression + onset 图：去掉几乎无信息的 qdiff 曲线。"""
    s = single["state"]
    t = twin["state"]

    fig, axes = plt.subplots(
        2, 1, figsize=(8.2, 6.6), sharex=True,
        gridspec_kw={"hspace": 0.05}, constrained_layout=True
    )

    # 6a 只保留最有用的累计进程对比
    axes[0].plot(s["strain"], s["A_sigma_norm"], color=C_SINGLE, lw=2.0, label="Single A_sigma")
    axes[0].plot(t["strain"], t["A_sigma_norm"], color=C_TWIN, lw=2.0, label="Twin A_sigma")
    axes[0].plot(s["strain"], s["A_P_parallel_norm"], color=C_SINGLE, lw=1.4, ls="--", label="Single A_P_parallel")
    axes[0].plot(t["strain"], t["A_P_parallel_norm"], color=C_TWIN, lw=1.4, ls="--", label="Twin A_P_parallel")
    axes[0].set_ylabel("Normalized cumulative")
    axes[0].set_title("Figure 6a. Cumulative mechanical vs polarization progression")
    axes[0].legend(fontsize=8.5, ncol=2)

    # 6b onset markers on chi_parallel
    axes[1].plot(s["strain"], s["chi_parallel"], color=C_SINGLE, lw=2.0, label="Single chi_parallel")
    axes[1].plot(t["strain"], t["chi_parallel"], color=C_TWIN, lw=2.0, label="Twin chi_parallel")

    s_on_sigma = float(s["onset_sigma"].iloc[0]) if "onset_sigma" in s.columns else np.nan
    t_on_sigma = float(t["onset_sigma"].iloc[0]) if "onset_sigma" in t.columns else np.nan
    s_on_p = float(s["onset_P_parallel"].iloc[0]) if "onset_P_parallel" in s.columns else np.nan
    t_on_p = float(t["onset_P_parallel"].iloc[0]) if "onset_P_parallel" in t.columns else np.nan

    if np.isfinite(s_on_sigma):
        axes[1].axvline(s_on_sigma, color=C_SINGLE, lw=1.0, ls="--", alpha=0.85)
    if np.isfinite(t_on_sigma):
        axes[1].axvline(t_on_sigma, color=C_TWIN, lw=1.0, ls="--", alpha=0.85)
    if np.isfinite(s_on_p):
        axes[1].axvline(s_on_p, color=C_SINGLE, lw=1.0, ls=":", alpha=0.9)
    if np.isfinite(t_on_p):
        axes[1].axvline(t_on_p, color=C_TWIN, lw=1.0, ls=":", alpha=0.9)

    axes[1].set_xlabel("Compressive strain e")
    axes[1].set_ylabel("chi_parallel")
    axes[1].set_title("Figure 6b. Onset markers: dashed=A_sigma, dotted=A_P_parallel")
    axes[1].legend(fontsize=9, ncol=2)

    fig.savefig(OUT_DIR / "Figure6_qc_and_onset.png", bbox_inches="tight")

# ============================================================
# 16. 汇总输出
# ============================================================
def write_report(single: dict, twin: dict):
    rows = []
    for d in [single, twin]:
        st = d["state"]
        rows.append({
            "dataset": d["name"],
            "onset_sigma": float(st["onset_sigma"].iloc[0]) if "onset_sigma" in st.columns else np.nan,
            "onset_P_parallel": float(st["onset_P_parallel"].iloc[0]) if "onset_P_parallel" in st.columns else np.nan,
            "max_chi_parallel": np.nanmax(st["chi_parallel"]),
            "min_chi_parallel": np.nanmin(st["chi_parallel"]),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "summary_supplemental.csv", index=False)

    report = []
    report.append("# Supplemental Discrimination Report\n\n")
    report.append("## 包含内容\n")
    report.append("- Figure 4: 精简后的整体 discrimination 主线，只保留 P_parallel、<|P_parallel|>、chi_parallel。\n")
    report.append("- Figure 5: 精简后的 quick section 图，只保留三个中截面的 P_parallel 与 H_sec_quick。\n")
    report.append("- Figure 6: 精简后的 progression / onset 图，只保留 A_sigma、A_P_parallel 与 chi_parallel onset 标记。\n")
    (OUT_DIR / "report_supplemental.md").write_text("".join(report), encoding="utf-8")


# ============================================================
# 17. 主程序入口
# ============================================================
def main():
    print("=" * 72)
    print(">>> BaTiO3 Twin / Single 补充分析脚本")
    print("=" * 72)
    print(f"工作目录: {BASE_DIR}")
    print(f"输出目录: {OUT_DIR}")
    print()

    try:
        print("正在处理 Single 数据集...")
        single = prepare_dataset("Single", FILES["Single"])
        print("正在处理 Twin 数据集...")
        twin = prepare_dataset("Twin", FILES["Twin"])
    except Exception as e:
        print(f"\n[错误] 数据加载或处理失败: {e}")
        return

    print("开始绘制图表...")
    make_figure4(single, twin)
    make_figure5(single, twin)
    make_figure6(single, twin)

    print("输出汇总报告...")
    write_report(single, twin)

    print(f"\n运行完成！所有图表和数据已保存到:\n{OUT_DIR}")
    plt.show()

if __name__ == "__main__":
    main()