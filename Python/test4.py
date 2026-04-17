# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths

# ============================================================
# BaTiO3 Twin / Single 协议化能量分析脚本（test4）
# ============================================================
# 目标：
# 1. 在 test1-3 的稳健证据层之上，增加“协议化能量层”；
# 2. 不追求完整热力学闭环，不把 proxy 直接当成真实热量/耗散；
# 3. 只做当前数据真正支持的三类分析：
#    (a) 全局能量路径：ΔPE, ΔKE, ΔETOT
#    (b) 路径几何：能量下探 / 能量回升 的累计量
#    (c) 事件级能量-组织耦合：ΔPE_i, ΔETOT_i 与 Δchi_parallel_i / ΔO_parallel_i
#
# 重要边界：
# - 当前机械文件只提供应变/应力/极化，不提供体积，因此
#   ∫s de 与 ΔETOT (eV) 不能直接做严格数值闭环。
# - 因此脚本中的 W_mech_proxy 只作为“机械输入强度 proxy”，
#   不与 ΔETOT 直接相减，不构造假 protocol residual。
# - 若以后机械输出加入 lx/ly/lz 或 volume，则可升级为能量密度闭环。
# ============================================================

# ============================================================
# 0. 路径配置
# ============================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

FILES = {
    "Single": {
        "ss": BASE_DIR / "single_ss_y_inst_2pct_cont.txt",
        "overall": BASE_DIR / "Pol_Pro_stats_overall_single.dat",
        "energy": BASE_DIR / "single_energy_2pct_cont.txt",
    },
    "Twin": {
        "ss": BASE_DIR / "twin_ss_y_inst_2pct_cont.txt",
        "overall": BASE_DIR / "Pol_Pro_stats_overall_twin.dat",
        "energy": BASE_DIR / "twin_energy_2pct_cont.txt",
    },
}

IN_FILE = BASE_DIR / "in.txt"
OUT_DIR = BASE_DIR / "results_protocol_energy_v1"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. 通用参数
# ============================================================
STRAIN_COL = "s2"
STRESS_COL = "p2"
TIME_MECH = ["TimeStep", "Time", "Step", "step"]
TIME_POL = ["TimeStep", "Time", "time", "step"]
TIME_ENE = ["TimeStep", "Time", "Step", "step"]
EPS = 1e-12

POLYORDER = 2
STRESS_SMOOTH_WINDOW = 9
STRESS_TREND_WINDOW = 41
SERRATION_RMS_WINDOW = 21
GLOBAL_SMOOTH_WINDOW = 11
GLOBAL_RATE_WINDOW = 21
ENERGY_SMOOTH_WINDOW = 21
ENERGY_RATE_WINDOW = 31

EVENT_SIGNAL_WINDOW = 7
EVENT_HEIGHT_SCALE = 0.80
EVENT_PROM_SCALE = 0.50
EVENT_MIN_DISTANCE = 4
EVENT_MIN_WIDTH = 2
EVENT_WIDTH_REL_HEIGHT = 0.65
EVENT_MERGE_GAP = 0
EVENT_PREPOST_PAD = 2
EVENT_ENERGY_FRAC = 0.05

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10.5,
    "axes.unicode_minus": False,
    "axes.linewidth": 1.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
    "mathtext.fontset": "stix",
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial Unicode MS", "Microsoft YaHei", "SimHei",
        "Noto Sans CJK SC", "PingFang SC", "Heiti SC",
        "Arial", "Helvetica", "DejaVu Sans"
    ],
})
C_SINGLE = "#d62728"
C_TWIN = "#1f77b4"
C_SINGLE_L = "#ff9896"
C_TWIN_L = "#9ecae1"

# ============================================================
# 2. 基础工具
# ============================================================
def odd_window(n: int, w: int, p: int = POLYORDER) -> int:
    if n <= p + 2:
        return 0
    w = min(w, n)
    if w % 2 == 0:
        w -= 1
    if w <= p:
        w = p + 3 + ((p + 3) % 2 == 0)
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return w if w > p and w >= 3 else 0


def smooth(y, w, p=POLYORDER):
    y = np.asarray(y, float)
    ww = odd_window(len(y), w, p)
    return y.copy() if ww == 0 else savgol_filter(y, ww, p, mode="interp")


def slope(x, y, w):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ww = odd_window(len(y), w, 1)
    if ww == 0 or len(y) < 3:
        return np.gradient(y, x)
    h, out = ww // 2, np.empty(len(y), float)
    for i in range(len(y)):
        lo, hi = max(0, i - h), min(len(y), i + h + 1)
        xs, ys = x[lo:hi], y[lo:hi]
        out[i] = np.nan if len(xs) < 2 else np.polyfit(xs - xs.mean(), ys, 1)[0]
    return out


def rms_run(y, w):
    y = np.asarray(y, float)
    ww = odd_window(len(y), w, 1)
    if ww == 0:
        return np.abs(y)
    h, out = ww // 2, np.empty(len(y), float)
    for i in range(len(y)):
        lo, hi = max(0, i - h), min(len(y), i + h + 1)
        out[i] = np.sqrt(np.mean(y[lo:hi] ** 2))
    return out


def cumtrapz0(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    out = np.zeros_like(y)
    if len(y) >= 2:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * np.diff(x))
    return out


def norm_to_final(y):
    y = np.asarray(y, float)
    if len(y) == 0:
        return y
    f = y[-1]
    return np.full_like(y, np.nan) if (not np.isfinite(f) or abs(f) < 1e-30) else y / f


def onset(x, y_norm, alpha=0.10):
    idx = np.where(np.asarray(y_norm, float) >= alpha)[0]
    return float(np.asarray(x, float)[idx[0]]) if len(idx) else np.nan


def mad_scale(y):
    y = np.asarray(y, float)
    med = np.nanmedian(y)
    mad = np.nanmedian(np.abs(y - med))
    s = 1.4826 * mad
    return float(s) if np.isfinite(s) and s > EPS else float(np.nanstd(y) + EPS)


def first_col(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def clean_name(s):
    s = re.sub(r"\|\s*([^|]+?)\s*\|", lambda m: f"abs{m.group(1).strip()}", str(s).strip())
    for a, b in [("(", ""), (")", ""), ("/", "_"), ("-", "_"), (" ", ""), ("<", ""), (">", "")]:
        s = s.replace(a, b)
    return s


def req(df, cols, name):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"{name}: 缺少必要列 {miss}；可用列为 {list(df.columns)}")


def find_zero_cross_after_min(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    i0 = int(np.nanargmin(y))
    for i in range(i0 + 1, len(y)):
        y0, y1 = y[i - 1], y[i]
        if np.isnan(y0) or np.isnan(y1):
            continue
        if y1 >= 0 and y0 < 0:
            x0, x1 = x[i - 1], x[i]
            if abs(y1 - y0) < EPS:
                return float(x1)
            frac = -y0 / (y1 - y0)
            return float(x0 + frac * (x1 - x0))
    return np.nan


def classify_energy_jump(dpe, threshold):
    if not np.isfinite(dpe):
        return "neutral"
    if dpe <= -threshold:
        return "release"
    if dpe >= threshold:
        return "storage"
    return "neutral"


def event_energy_threshold(vals):
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return EPS
    frac_thr = EVENT_ENERGY_FRAC * np.nanmax(np.abs(vals))
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    robust_thr = 0.5 * 1.4826 * mad
    return max(EPS, frac_thr, robust_thr)

# ============================================================
# 3. 读入
# ============================================================
def read_ws(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep=r"\s+", comment="#", engine="python")


def read_vars(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.readline()
    if first.strip().startswith("VARIABLES"):
        cols = [clean_name(x) for x in re.findall(r'"([^"]+)"', first)]
        df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=None, names=cols, engine="python")
    else:
        df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
        df.columns = [clean_name(c) for c in df.columns]
    return df.rename(columns={
        "absPxabs_mean": "absPx_mean",
        "absPyabs_mean": "absPy_mean",
        "absPzabs_mean": "absPz_mean",
        "absPabs_mean": "absP_mean",
    })


def read_energy(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    tcol = first_col(df, TIME_ENE)
    if tcol is None:
        raise KeyError(f"{path.name}: 能量文件未找到 TimeStep/Time 列")
    df = df.rename(columns={tcol: "Time"})
    req(df, ["PE_total", "KE_total", "ETOT"], path.name)
    # 强制将所有数值列转为 float 以避免合并时的 MergeError
    for c in ["Time", "PE_total", "KE_total", "ETOT"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    df = df.dropna(subset=["Time", "PE_total", "KE_total", "ETOT"]).copy()
    df = df.drop_duplicates(subset=["Time"], keep="last").sort_values("Time").reset_index(drop=True)
    return df


def parse_in_metadata(path: Path) -> dict:
    out = {
        "Ptarget": np.nan,
        "Nload": np.nan,
        "epsY": np.nan,
        "has_volume_output": False,
        "has_energy_exchange_output": False,
        "protocol_note": "",
    }
    if not path.exists():
        out["protocol_note"] = "in.txt not found"
        return out
    txt = path.read_text(encoding="utf-8", errors="ignore")
    def grab(var):
        m = re.search(rf"variable\s+{var}\s+equal\s+([-+0-9.eE]+)", txt)
        return float(m.group(1)) if m else np.nan
    out["Ptarget"] = grab("Ptarget")
    out["Nload"] = grab("Nload")
    out["epsY"] = grab("epsY")
    out["has_volume_output"] = all(k in txt for k in ["lx", "ly", "lz"]) and ("OUTINST" in txt and "lx ly lz" in txt)
    out["has_energy_exchange_output"] = ("ecouple" in txt) or ("econserve" in txt) or ("f_npt" in txt)
    if ("fix             npt_dyn" in txt or "fix             npt_load" in txt) and ("fix             ndef" in txt or "fix             ndef all deform" in txt):
        out["protocol_note"] = "x/z npt + y deform protocol; mechanical work is only available as proxy from stress-strain unless volume/work channels are exported"
    else:
        out["protocol_note"] = "protocol not fully identified"
    return out

# ============================================================
# 4. 清洗 / 对齐
# ============================================================
def clean_mech(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    tcol = first_col(df, TIME_MECH)
    if tcol is None:
        warnings.warn(f"{name}: 未找到 Time/TimeStep，退化为按行序对齐。")
        df["_row_id_"] = np.arange(len(df))
    else:
        df = df.rename(columns={tcol: "Time"})
        df["Time"] = df["Time"].astype(float)
        df = df.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    req(df, [STRAIN_COL, STRESS_COL], name)
    out = pd.DataFrame({
        "strain": np.abs(pd.to_numeric(df[STRAIN_COL], errors="coerce").to_numpy(float)),
        "stress_raw": np.abs(pd.to_numeric(df[STRESS_COL], errors="coerce").to_numpy(float)),
    })
    if "Time" in df.columns:
        out["Time"] = df["Time"].to_numpy(float)
    else:
        out["_row_id_"] = df["_row_id_"].to_numpy(int)
    out = out[np.isfinite(out["strain"]) & np.isfinite(out["stress_raw"])].copy()
    return out.sort_values("Time" if "Time" in out.columns else "strain").reset_index(drop=True)


def clean_pol(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = df.dropna(how="all").copy()
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    tcol = first_col(df, TIME_POL)
    if tcol is None:
        raise KeyError(f"{name}: 极化文件未找到 Time 或 TimeStep 列")
    df = df.rename(columns={tcol: "Time"})
    df["Time"] = df["Time"].astype(float)
    return df.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)


def align_mech_pol(mech: pd.DataFrame, pol: pd.DataFrame) -> pd.DataFrame:
    if "Time" in mech.columns and "Time" in pol.columns:
        # 强制转换类型解决 MergeError: incompatible merge keys
        pol = pol.copy()
        mech = mech.copy()
        pol["Time"] = pol["Time"].astype(float)
        mech["Time"] = mech["Time"].astype(float)
        out = pd.merge_asof(pol.sort_values("Time"), mech.sort_values("Time"), on="Time", direction="nearest")
        out["align_method"] = "time"
        return out
    if len(mech) == len(pol):
        out = pol.copy()
        for c in mech.columns:
            if c not in out.columns:
                out[c] = mech[c].to_numpy()
        out["align_method"] = "row"
        return out
    u0, u1 = np.linspace(0, 1, len(mech)), np.linspace(0, 1, len(pol))
    out = pol.copy()
    for c in mech.columns:
        if c not in ["Time", "_row_id_"]:
            out[c] = np.interp(u1, u0, mech[c].to_numpy(float))
    out["align_method"] = "progress"
    return out


def align_state_energy(state: pd.DataFrame, ene: pd.DataFrame) -> pd.DataFrame:
    if "Time" in state.columns and "Time" in ene.columns:
        # 强制转换类型解决 MergeError: incompatible merge keys
        state = state.copy()
        ene = ene.copy()
        state["Time"] = state["Time"].astype(float)
        ene["Time"] = ene["Time"].astype(float)
        out = pd.merge_asof(state.sort_values("Time"), ene.sort_values("Time"), on="Time", direction="nearest")
        out["energy_align_method"] = "time"
        return out
    if len(state) == len(ene):
        out = state.copy()
        for c in ["PE_total", "KE_total", "ETOT"]:
            out[c] = ene[c].to_numpy(float)
        out["energy_align_method"] = "row"
        return out
    u0, u1 = np.linspace(0, 1, len(ene)), np.linspace(0, 1, len(state))
    out = state.copy()
    for c in ["PE_total", "KE_total", "ETOT"]:
        out[c] = np.interp(u1, u0, ene[c].to_numpy(float))
    out["energy_align_method"] = "progress"
    return out

# ============================================================
# 5. 机械 / 极化 / 能量状态量
# ============================================================
def add_mech_metrics(mech: pd.DataFrame) -> pd.DataFrame:
    out = mech.copy()
    e = out["strain"].to_numpy(float)
    s = out["stress_raw"].to_numpy(float)
    out["stress_sm"] = smooth(s, STRESS_SMOOTH_WINDOW)
    out["stress_trend"] = smooth(s, STRESS_TREND_WINDOW)
    out["delta_sigma_serr"] = s - out["stress_trend"].to_numpy(float)
    out["delta_sigma_serr_sm"] = smooth(out["delta_sigma_serr"].to_numpy(float), EVENT_SIGNAL_WINDOW)
    out["S_rms"] = rms_run(out["delta_sigma_serr"].to_numpy(float), SERRATION_RMS_WINDOW)
    out["A_sigma"] = cumtrapz0(e, out["S_rms"].to_numpy(float))
    out["A_sigma_norm"] = norm_to_final(out["A_sigma"].to_numpy(float))
    out["onset_sigma"] = onset(e, out["A_sigma_norm"].to_numpy(float))
    out["W_mech_proxy"] = cumtrapz0(e, out["stress_sm"].to_numpy(float))
    return out


def add_pol_metrics(state: pd.DataFrame, name: str) -> pd.DataFrame:
    out = state.copy()
    req(out, ["Px_mean", "Py_mean", "Pz_mean", "absPx_mean", "absPy_mean", "absPz_mean", "absP_mean"], f"{name} overall")
    out["P_parallel"] = out["Py_mean"]
    out["O_parallel"] = out["absPy_mean"]
    out["Pmag_mean"] = out["absP_mean"]
    out["Pvec_norm"] = np.sqrt(out["Px_mean"] ** 2 + out["Py_mean"] ** 2 + out["Pz_mean"] ** 2)
    out["chi_parallel"] = np.abs(out["P_parallel"]) / (out["O_parallel"] + EPS)
    out["chi_vec"] = out["Pvec_norm"] / (out["Pmag_mean"] + EPS)
    denom = out["absPx_mean"] + out["absPy_mean"] + out["absPz_mean"] + EPS
    out["eta_perp"] = (out["absPx_mean"] + out["absPz_mean"]) / denom

    for c in ["P_parallel", "O_parallel", "chi_parallel", "chi_vec", "eta_perp"]:
        out[f"{c}_sm"] = smooth(out[c].to_numpy(float), GLOBAL_SMOOTH_WINDOW)

    out["P_parallel_abs_sm"] = smooth(np.abs(out["P_parallel"].to_numpy(float)), GLOBAL_SMOOTH_WINDOW)

    # O_parallel 做相对归一化是稳的；chi_parallel 用原始平滑量更稳，避免初值过小造成 ratio 爆大。
    out["O_parallel_rel"] = out["O_parallel_sm"] / (float(out["O_parallel_sm"].iloc[0]) + EPS)
    out["chi_parallel_rel"] = out["chi_parallel_sm"] / (float(out["chi_parallel_sm"].iloc[0]) + EPS)
    out["chi_parallel_delta"] = out["chi_parallel_sm"] - float(out["chi_parallel_sm"].iloc[0])
    return out


def add_energy_metrics(state: pd.DataFrame, name: str) -> pd.DataFrame:
    out = state.copy()
    req(out, ["PE_total", "KE_total", "ETOT"], f"{name} energy")
    e = out["strain"].to_numpy(float)

    pe = out["PE_total"].to_numpy(float)
    ke = out["KE_total"].to_numpy(float)
    et = out["ETOT"].to_numpy(float)

    out["dPE"] = pe - pe[0]
    out["dKE"] = ke - ke[0]
    out["dETOT"] = et - et[0]

    for c in ["dPE", "dKE", "dETOT"]:
        out[f"{c}_sm"] = smooth(out[c].to_numpy(float), ENERGY_SMOOTH_WINDOW)
        out[f"d{c}_de"] = slope(e, out[f"{c}_sm"].to_numpy(float), ENERGY_RATE_WINDOW)

    # 路径几何：累计“能量下探”与“能量回升/重储存”
    for base in ["dPE", "dETOT"]:
        rate = out[f"d{base}_de"].to_numpy(float)
        down = np.clip(-rate, 0.0, None)
        up = np.clip(rate, 0.0, None)
        out[f"A_{base}_down"] = cumtrapz0(e, down)
        out[f"A_{base}_up"] = cumtrapz0(e, up)
        out[f"A_{base}_down_norm"] = norm_to_final(out[f"A_{base}_down"].to_numpy(float))
        out[f"A_{base}_up_norm"] = norm_to_final(out[f"A_{base}_up"].to_numpy(float))
        out[f"onset_{base}_down"] = onset(e, out[f"A_{base}_down_norm"].to_numpy(float))
        out[f"onset_{base}_up"] = onset(e, out[f"A_{base}_up_norm"].to_numpy(float))

    out["e_min_dPE"] = float(e[int(np.nanargmin(out["dPE"].to_numpy(float)))])
    out["e_min_dETOT"] = float(e[int(np.nanargmin(out["dETOT"].to_numpy(float)))])
    out["dPE_min"] = float(np.nanmin(out["dPE"].to_numpy(float)))
    out["dETOT_min"] = float(np.nanmin(out["dETOT"].to_numpy(float)))
    out["e_zero_dPE"] = find_zero_cross_after_min(e, out["dPE"].to_numpy(float))
    out["e_zero_dETOT"] = find_zero_cross_after_min(e, out["dETOT"].to_numpy(float))
    return out

# ============================================================
# 6. 事件识别与能量条件化分析
# ============================================================
def event_windows(state: pd.DataFrame) -> list[tuple[int, int]]:
    sig = np.abs(state["delta_sigma_serr_sm"].to_numpy(float))
    scale = mad_scale(sig)
    height = max(EVENT_HEIGHT_SCALE * scale, EPS)
    prom = max(EVENT_PROM_SCALE * scale, EPS)
    peaks, _ = find_peaks(sig, height=height, prominence=prom, distance=EVENT_MIN_DISTANCE)
    if len(peaks) == 0:
        return []
    widths, _, left_ips, right_ips = peak_widths(sig, peaks, rel_height=EVENT_WIDTH_REL_HEIGHT)
    segs = []
    n = len(sig)
    for l, r in zip(left_ips, right_ips):
        lo = max(0, int(np.floor(l)))
        hi = min(n - 1, int(np.ceil(r)))
        if hi - lo + 1 >= EVENT_MIN_WIDTH:
            segs.append((lo, hi))
    if not segs:
        return []
    segs.sort()
    merged = [segs[0]]
    for a, b in segs[1:]:
        la, lb = merged[-1]
        if a <= lb + EVENT_MERGE_GAP:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def build_events_energy(state: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = [
        "dataset", "event_id", "e_start", "e_end", "duration_e", "s_onset", "s_pre", "s_post",
        "Delta_s_drop", "W_sigma", "W_mech_proxy", "Delta_PE", "Delta_KE", "Delta_ETOT",
        "Delta_P_parallel", "Delta_O_parallel", "Delta_chi_parallel", "energy_class"
    ]
    segs = event_windows(state)
    if not segs:
        return pd.DataFrame(columns=cols)

    e = state["strain"].to_numpy(float)
    s = state["stress_sm"].to_numpy(float)
    ds = state["delta_sigma_serr_sm"].to_numpy(float)
    dpe = state["dPE_sm"].to_numpy(float)
    dke = state["dKE_sm"].to_numpy(float)
    det = state["dETOT_sm"].to_numpy(float)

    def pick(c, i):
        return float(state[c].iloc[i]) if c in state.columns else np.nan

    tmp_dpe = []
    raw_rows = []
    for k, (i0, i1) in enumerate(segs, 1):
        ipre, ipost = max(0, i0 - EVENT_PREPOST_PAD), min(len(state) - 1, i1 + EVENT_PREPOST_PAD)
        sl = slice(i0, i1 + 1)
        es, ss, dss = e[sl], s[sl], ds[sl]
        imax = int(np.argmax(ss))
        spre = float(ss[imax])
        spost = float(np.nanmin(ss[imax:])) if imax < len(ss) else float(ss[-1])
        dpe_i = float(dpe[ipost] - dpe[ipre])
        tmp_dpe.append(abs(dpe_i))
        raw_rows.append({
            "dataset": name,
            "event_id": k,
            "e_start": float(e[i0]),
            "e_end": float(e[i1]),
            "duration_e": float(e[i1] - e[i0]),
            "s_onset": float(s[i0]),
            "s_pre": spre,
            "s_post": spost,
            "Delta_s_drop": max(0.0, spre - spost),
            "W_sigma": float(np.trapz(np.abs(dss), es)) if len(es) >= 2 else 0.0,
            "W_mech_proxy": float(np.trapz(ss, es)) if len(es) >= 2 else 0.0,
            "Delta_PE": dpe_i,
            "Delta_KE": float(dke[ipost] - dke[ipre]),
            "Delta_ETOT": float(det[ipost] - det[ipre]),
            "Delta_P_parallel": pick("P_parallel_sm", ipost) - pick("P_parallel_sm", ipre),
            "Delta_O_parallel": pick("O_parallel_sm", ipost) - pick("O_parallel_sm", ipre),
            "Delta_chi_parallel": pick("chi_parallel_sm", ipost) - pick("chi_parallel_sm", ipre),
        })
    thr = event_energy_threshold(tmp_dpe) if len(tmp_dpe) else EPS
    rows = []
    for r in raw_rows:
        r["energy_class"] = classify_energy_jump(r["Delta_PE"], thr)
        rows.append(r)
    return pd.DataFrame(rows, columns=cols)


def build_threshold_spectrum(events: pd.DataFrame, name: str, label: str, weight_col: str) -> pd.DataFrame:
    cols = ["dataset", "class", "s", "G_N", "G_W"]
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=cols)
    ev = events.sort_values("s_onset").reset_index(drop=True)
    s = ev["s_onset"].to_numpy(float)
    w = ev[weight_col].to_numpy(float)
    if len(s) == 0:
        return pd.DataFrame(columns=cols)
    grid = np.unique(s)
    return pd.DataFrame({
        "dataset": name,
        "class": label,
        "s": grid,
        "G_N": [np.mean(s <= x) for x in grid],
        "G_W": [np.sum(w[s <= x]) / (np.sum(w) + EPS) if np.sum(w) > 0 else np.nan for x in grid],
    })


def build_energy_conditioned_spectra(events: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = ["dataset", "class", "s", "G_N", "G_W"]
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=cols)
    pieces = []
    for label, sub in {
        "release": events[events["energy_class"] == "release"].copy(),
        "storage": events[events["energy_class"] == "storage"].copy(),
    }.items():
        if len(sub) == 0:
            continue
        sub = sub.copy()
        sub["PE_weight"] = np.abs(sub["Delta_PE"].to_numpy(float))
        pieces.append(build_threshold_spectrum(sub, name, label, "PE_weight"))
    return pd.concat(pieces, axis=0, ignore_index=True) if pieces else pd.DataFrame(columns=cols)

# ============================================================
# 7. 主流程
# ============================================================
def prepare_dataset(name: str, cfg: dict) -> dict:
    mech = add_mech_metrics(clean_mech(read_ws(cfg["ss"]), f"{name} mechanical"))
    pol = clean_pol(read_vars(cfg["overall"]), f"{name} overall")
    ene = read_energy(cfg["energy"])

    state0 = add_pol_metrics(align_mech_pol(mech, pol), name)
    state = add_energy_metrics(align_state_energy(state0, ene), name)
    state["dataset"] = name

    events = build_events_energy(state, name)
    ene_spec = build_energy_conditioned_spectra(events, name)

    return {
        "name": name,
        "mech": mech,
        "pol": pol,
        "ene": ene,
        "state": state,
        "events": events,
        "ene_spec": ene_spec,
    }

# ============================================================
# 8. 作图
# ============================================================
def fig1_energy_paths(single: dict, twin: dict):
    s, t = single["state"], twin["state"]
    fig, ax = plt.subplots(3, 1, figsize=(8.4, 9.2), sharex=True, gridspec_kw={"hspace": 0.05}, constrained_layout=True)

    ax[0].plot(s["strain"], s["dPE_sm"], color=C_SINGLE, lw=2.0, label="Single ΔPE")
    ax[0].plot(t["strain"], t["dPE_sm"], color=C_TWIN, lw=2.0, label="Twin ΔPE")
    ax[0].axhline(0.0, color="black", lw=0.8, ls=":")
    ax[0].set_ylabel("ΔPE (eV)")
    ax[0].set_title("Figure 4a. Potential-energy pathway")
    ax[0].legend(fontsize=9, ncol=2)

    ax[1].plot(s["strain"], s["dETOT_sm"], color=C_SINGLE, lw=2.0, label="Single ΔETOT")
    ax[1].plot(t["strain"], t["dETOT_sm"], color=C_TWIN, lw=2.0, label="Twin ΔETOT")
    ax[1].plot(s["strain"], s["dKE_sm"], color=C_SINGLE_L, lw=1.2, ls="--", label="Single ΔKE")
    ax[1].plot(t["strain"], t["dKE_sm"], color=C_TWIN_L, lw=1.2, ls="--", label="Twin ΔKE")
    ax[1].axhline(0.0, color="black", lw=0.8, ls=":")
    ax[1].set_ylabel("ΔETOT / ΔKE (eV)")
    ax[1].set_title("Figure 4b. Total- and kinetic-energy pathway")
    ax[1].legend(fontsize=8.8, ncol=2)

    ax[2].plot(s["strain"], s["W_mech_proxy"], color=C_SINGLE, lw=2.0, label="Single W_mech_proxy")
    ax[2].plot(t["strain"], t["W_mech_proxy"], color=C_TWIN, lw=2.0, label="Twin W_mech_proxy")
    ax[2].plot(s["strain"], s["A_sigma_norm"] * float(np.nanmax(s["W_mech_proxy"])), color=C_SINGLE_L, lw=1.1, ls=":", label="Single A_sigma_norm (scaled)")
    ax[2].plot(t["strain"], t["A_sigma_norm"] * float(np.nanmax(t["W_mech_proxy"])), color=C_TWIN_L, lw=1.1, ls=":", label="Twin A_sigma_norm (scaled)")
    ax[2].set_xlabel("Compressive strain e")
    ax[2].set_ylabel("Mechanical proxy")
    ax[2].set_title("Figure 4c. Mechanical input proxy (not unit-matched to eV)")
    ax[2].legend(fontsize=8.2, ncol=2)

    fig.savefig(OUT_DIR / "Figure4_protocol_energy_paths.png", bbox_inches="tight")


def fig2_path_geometry(single: dict, twin: dict):
    s, t = single["state"], twin["state"]
    fig, ax = plt.subplots(3, 1, figsize=(8.4, 9.2), sharex=True, gridspec_kw={"hspace": 0.05}, constrained_layout=True)

    ax[0].plot(s["strain"], s["A_dPE_down_norm"], color=C_SINGLE, lw=2.0, label="Single A_PE^-")
    ax[0].plot(t["strain"], t["A_dPE_down_norm"], color=C_TWIN, lw=2.0, label="Twin A_PE^-")
    ax[0].plot(s["strain"], s["A_dPE_up_norm"], color=C_SINGLE, lw=1.3, ls="--", label="Single A_PE^+")
    ax[0].plot(t["strain"], t["A_dPE_up_norm"], color=C_TWIN, lw=1.3, ls="--", label="Twin A_PE^+")
    ax[0].set_ylabel("Normalized cumulative PE geometry")
    ax[0].set_title("Figure 5a. Potential-energy descent vs storage progression")
    ax[0].legend(fontsize=8.6, ncol=2)

    ax[1].plot(s["strain"], s["A_dETOT_down_norm"], color=C_SINGLE, lw=2.0, label="Single A_E^-")
    ax[1].plot(t["strain"], t["A_dETOT_down_norm"], color=C_TWIN, lw=2.0, label="Twin A_E^-")
    ax[1].plot(s["strain"], s["A_dETOT_up_norm"], color=C_SINGLE, lw=1.3, ls="--", label="Single A_E^+")
    ax[1].plot(t["strain"], t["A_dETOT_up_norm"], color=C_TWIN, lw=1.3, ls="--", label="Twin A_E^+")
    ax[1].set_ylabel("Normalized cumulative ETOT geometry")
    ax[1].set_title("Figure 5b. Total-energy descent vs storage progression")
    ax[1].legend(fontsize=8.6, ncol=2)

    ax[2].plot(s["strain"], s["O_parallel_rel"], color=C_SINGLE, lw=1.8, label="Single O_parallel/O0")
    ax[2].plot(s["strain"], s["chi_parallel_sm"], color=C_SINGLE, lw=1.1, ls="--", label="Single chi_parallel")
    ax[2].plot(t["strain"], t["O_parallel_rel"], color=C_TWIN, lw=1.8, label="Twin O_parallel/O0")
    ax[2].plot(t["strain"], t["chi_parallel_sm"], color=C_TWIN, lw=1.1, ls="--", label="Twin chi_parallel")
    ax[2].set_xlabel("Compressive strain e")
    ax[2].set_ylabel("Organization metrics")
    ax[2].set_title("Figure 5c. Occupancy pathway and bounded coherence")
    ax[2].legend(fontsize=8.4, ncol=2)

    fig.savefig(OUT_DIR / "Figure5_path_geometry.png", bbox_inches="tight")


def fig3_event_energy(single: dict, twin: dict):
    es, et = single["events"], twin["events"]
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 7.8), constrained_layout=True)

    for ev, c, name in [(es, C_SINGLE, "Single"), (et, C_TWIN, "Twin")]:
        if len(ev) == 0:
            continue
        rel = ev[ev["energy_class"] == "release"].copy()
        ax[0, 0].scatter(ev["s_onset"], ev["Delta_PE"], s=24, color=c, alpha=0.8, label=name)
        ax[0, 1].scatter(ev["W_sigma"], ev["Delta_PE"], s=24, color=c, alpha=0.8, label=name)
        if len(rel):
            ax[1, 0].scatter(np.abs(rel["Delta_chi_parallel"]), -rel["Delta_PE"], s=24, color=c, alpha=0.8, label=f"{name} release")
            ax[1, 1].scatter(np.abs(rel["Delta_O_parallel"]), -rel["Delta_PE"], s=24, color=c, alpha=0.8, label=f"{name} release")

    ax[0, 0].axhline(0.0, color="black", lw=0.8, ls=":")
    ax[0, 0].set_xlabel("Event onset stress")
    ax[0, 0].set_ylabel("ΔPE_i (eV)")
    ax[0, 0].set_title("Figure 6a. Event onset stress vs signed ΔPE_i")
    ax[0, 0].legend(fontsize=8.5)

    ax[0, 1].axhline(0.0, color="black", lw=0.8, ls=":")
    ax[0, 1].set_xlabel("W_sigma")
    ax[0, 1].set_ylabel("ΔPE_i (eV)")
    ax[0, 1].set_title("Figure 6b. Mechanical burst proxy vs signed ΔPE_i")
    ax[0, 1].legend(fontsize=8.5)

    ax[1, 0].set_xlabel("|Δchi_parallel_i|")
    ax[1, 0].set_ylabel("-ΔPE_i (release only)")
    ax[1, 0].set_title("Figure 6c. Coherence jump vs release magnitude")
    ax[1, 0].legend(fontsize=8.5)

    ax[1, 1].set_xlabel("|ΔO_parallel_i|")
    ax[1, 1].set_ylabel("-ΔPE_i (release only)")
    ax[1, 1].set_title("Figure 6d. Occupancy jump vs release magnitude")
    ax[1, 1].legend(fontsize=8.5)

    fig.savefig(OUT_DIR / "Figure6_event_energy_coupling.png", bbox_inches="tight")


def fig4_energy_conditioned_spectra(single: dict, twin: dict):
    ss, st = single["ene_spec"], twin["ene_spec"]
    es, et = single["events"], twin["events"]
    fig, ax = plt.subplots(2, 1, figsize=(8.4, 7.8), sharex=True, gridspec_kw={"hspace": 0.08}, constrained_layout=True)

    for spec, ev, c, name in [(ss, es, C_SINGLE, "Single"), (st, et, C_TWIN, "Twin")]:
        if len(spec) == 0:
            continue
        rel = spec[spec["class"] == "release"]
        sto = spec[spec["class"] == "storage"]
        nrel = int(np.sum(ev["energy_class"] == "release")) if len(ev) else 0
        nsto = int(np.sum(ev["energy_class"] == "storage")) if len(ev) else 0
        if len(rel):
            ax[0].plot(rel["s"], rel["G_N"], color=c, lw=2.0, marker="o", ms=3, label=f"{name} release G_N (N={nrel})")
            ax[0].plot(rel["s"], rel["G_W"], color=c, lw=1.2, ls="--", marker="o", ms=3, label=f"{name} release G_W")
        if len(sto):
            ax[1].plot(sto["s"], sto["G_N"], color=c, lw=2.0, marker="o", ms=3, label=f"{name} storage G_N (N={nsto})")
            ax[1].plot(sto["s"], sto["G_W"], color=c, lw=1.2, ls="--", marker="o", ms=3, label=f"{name} storage G_W")

    ax[0].set_ylabel("Cumulative release spectrum")
    ax[0].set_title("Figure 7a. PE-release-conditioned threshold spectrum")
    ax[0].legend(fontsize=8.2, ncol=2)

    ax[1].set_xlabel("Onset stress")
    ax[1].set_ylabel("Cumulative storage spectrum")
    ax[1].set_title("Figure 7b. PE-storage-conditioned threshold spectrum")
    ax[1].legend(fontsize=8.2, ncol=2)

    fig.savefig(OUT_DIR / "Figure7_energy_conditioned_spectra.png", bbox_inches="tight")

# ============================================================
# 9. 汇总与报告
# ============================================================
def spearman_proxy(x, y):
    x = pd.Series(np.asarray(x, float))
    y = pd.Series(np.asarray(y, float))
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    return float(x[mask].corr(y[mask], method="spearman"))


def summarize(ds: dict, meta: dict) -> dict:
    st, ev, name = ds["state"], ds["events"], ds["name"]
    out = {
        "dataset": name,
        "protocol_note": meta.get("protocol_note", ""),
        "Ptarget_bar": meta.get("Ptarget", np.nan),
        "Nload": meta.get("Nload", np.nan),
        "epsY": meta.get("epsY", np.nan),
        "has_volume_output": meta.get("has_volume_output", False),
        "has_energy_exchange_output": meta.get("has_energy_exchange_output", False),
        "dPE_min": float(np.nanmin(st["dPE"].to_numpy(float))),
        "e_min_dPE": float(st["e_min_dPE"].iloc[0]),
        "e_zero_dPE": float(st["e_zero_dPE"].iloc[0]) if np.isfinite(st["e_zero_dPE"].iloc[0]) else np.nan,
        "dETOT_min": float(np.nanmin(st["dETOT"].to_numpy(float))),
        "e_min_dETOT": float(st["e_min_dETOT"].iloc[0]),
        "e_zero_dETOT": float(st["e_zero_dETOT"].iloc[0]) if np.isfinite(st["e_zero_dETOT"].iloc[0]) else np.nan,
        "A_dPE_down_end": float(st["A_dPE_down"].iloc[-1]),
        "A_dPE_up_end": float(st["A_dPE_up"].iloc[-1]),
        "A_dETOT_down_end": float(st["A_dETOT_down"].iloc[-1]),
        "A_dETOT_up_end": float(st["A_dETOT_up"].iloc[-1]),
        "W_mech_proxy_end": float(st["W_mech_proxy"].iloc[-1]),
        "N_events": int(len(ev)),
        "N_release": int(np.sum(ev["energy_class"] == "release")) if len(ev) else 0,
        "N_storage": int(np.sum(ev["energy_class"] == "storage")) if len(ev) else 0,
        "median_Delta_PE_release": float(np.nanmedian(ev.loc[ev["energy_class"] == "release", "Delta_PE"])) if len(ev) and np.any(ev["energy_class"] == "release") else np.nan,
        "median_Delta_PE_storage": float(np.nanmedian(ev.loc[ev["energy_class"] == "storage", "Delta_PE"])) if len(ev) and np.any(ev["energy_class"] == "storage") else np.nan,
        "rho_Wsigma_signedDeltaPE": spearman_proxy(ev["W_sigma"] if len(ev) else [], ev["Delta_PE"] if len(ev) else []),
        "rho_Wsigma_releaseMag": spearman_proxy(ev.loc[ev["energy_class"] == "release", "W_sigma"] if len(ev) else [], -ev.loc[ev["energy_class"] == "release", "Delta_PE"] if len(ev) else []),
        "rho_chi_releaseMag": spearman_proxy(np.abs(ev.loc[ev["energy_class"] == "release", "Delta_chi_parallel"]) if len(ev) else [], -ev.loc[ev["energy_class"] == "release", "Delta_PE"] if len(ev) else []),
        "rho_O_releaseMag": spearman_proxy(np.abs(ev.loc[ev["energy_class"] == "release", "Delta_O_parallel"]) if len(ev) else [], -ev.loc[ev["energy_class"] == "release", "Delta_PE"] if len(ev) else []),
    }
    return out


def write_report(single: dict, twin: dict, meta: dict):
    summary = pd.DataFrame([summarize(single, meta), summarize(twin, meta)])
    summary.to_csv(OUT_DIR / "summary_energy.csv", index=False)

    lines = []
    lines += ["# Protocol energy report\n\n"]
    lines += ["- This script adds a protocol-level energy layer on top of test1-3.\n"]
    lines += ["- It uses ΔPE, ΔKE, ΔETOT, cumulative descent/storage geometry, and event-level energy-organization coupling.\n"]
    lines += ["- It does NOT claim full thermodynamic closure.\n"]
    if not meta.get("has_volume_output", False):
        lines += ["- Mechanical file does not export volume; therefore W_mech_proxy cannot be converted into eV or directly subtracted from ΔETOT.\n"]
    if not meta.get("has_energy_exchange_output", False):
        lines += ["- No explicit thermostat/barostat energy exchange was exported; protocol residual is intentionally not constructed.\n"]
    lines += ["- Recommended main reading: deeper early PE descent in Twin vs stronger later PE recovery/storage in Single.\n"]
    (OUT_DIR / "report_energy.md").write_text("".join(lines), encoding="utf-8")

# ============================================================
# 10. 主程序
# ============================================================
def main():
    print("=" * 72)
    print(">>> BaTiO3 protocol energy analysis (test4)")
    print("=" * 72)
    print(f"工作目录: {BASE_DIR}")
    print(f"输出目录: {OUT_DIR}\n")

    meta = parse_in_metadata(IN_FILE)
    print("协议识别:")
    print(f"  Ptarget = {meta['Ptarget']} bar")
    print(f"  Nload   = {meta['Nload']}")
    print(f"  epsY    = {meta['epsY']}")
    print(f"  has_volume_output = {meta['has_volume_output']}")
    print(f"  has_energy_exchange_output = {meta['has_energy_exchange_output']}")
    print(f"  note: {meta['protocol_note']}\n")

    single = prepare_dataset("Single", FILES["Single"])
    twin = prepare_dataset("Twin", FILES["Twin"])

    single["state"].to_csv(OUT_DIR / "state_energy_single.csv", index=False)
    twin["state"].to_csv(OUT_DIR / "state_energy_twin.csv", index=False)
    single["events"].to_csv(OUT_DIR / "events_energy_single.csv", index=False)
    twin["events"].to_csv(OUT_DIR / "events_energy_twin.csv", index=False)
    single["ene_spec"].to_csv(OUT_DIR / "energy_conditioned_spectrum_single.csv", index=False)
    twin["ene_spec"].to_csv(OUT_DIR / "energy_conditioned_spectrum_twin.csv", index=False)

    fig1_energy_paths(single, twin)
    fig2_path_geometry(single, twin)
    fig3_event_energy(single, twin)
    fig4_energy_conditioned_spectra(single, twin)
    write_report(single, twin, meta)

    print("运行完成：")
    print("- state_energy_single/twin.csv")
    print("- events_energy_single/twin.csv")
    print("- energy_conditioned_spectrum_single/twin.csv")
    print("- Figure4~7")
    print("- summary_energy.csv / report_energy.md")
    plt.show()


if __name__ == "__main__":
    main()