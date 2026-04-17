# -*- coding: utf-8 -*-
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, peak_widths

# ===================== 配置 =====================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

FILES = {
    "Single": {
        "ss": BASE_DIR / "single_ss_y_inst_2pct_cont.txt",
        "overall": BASE_DIR / "Pol_Pro_stats_overall_single.dat",
    },
    "Twin": {
        "ss": BASE_DIR / "twin_ss_y_inst_2pct_cont.txt",
        "overall": BASE_DIR / "Pol_Pro_stats_overall_twin.dat",
    },
}
OUT_DIR = BASE_DIR / "results_energy_event_minimal"
OUT_DIR.mkdir(exist_ok=True)

STRAIN_COL, STRESS_COL = "s2", "p2"
TIME_MECH = ["TimeStep", "Time", "Step", "step"]
TIME_POL = ["TimeStep", "Time", "time", "step"]
EPS = 1e-12

POLYORDER = 2
STRESS_SMOOTH_WINDOW = 9
STRESS_TREND_WINDOW = 41
SERRATION_RMS_WINDOW = 21
GLOBAL_SMOOTH_WINDOW = 11
GLOBAL_RATE_WINDOW = 21
RATIO_MIN_FRAC_OF_FINAL = 0.05

EVENT_SIGNAL_WINDOW = 7
EVENT_HEIGHT_SCALE = 0.80
EVENT_PROM_SCALE = 0.50
EVENT_MIN_DISTANCE = 4
EVENT_MIN_WIDTH = 2
EVENT_WIDTH_REL_HEIGHT = 0.65
EVENT_MERGE_GAP = 0
EVENT_PREPOST_PAD = 2

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
        "Arial Unicode MS", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "PingFang SC", "Heiti SC", "Arial", "Helvetica", "DejaVu Sans"
    ],
})
C_SINGLE, C_TWIN = "#d62728", "#1f77b4"

# ===================== 工具 =====================
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


def stable_ratio(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    dmax = np.nanmax(den) if np.isfinite(np.nanmax(den)) else np.nan
    floor = max(EPS, RATIO_MIN_FRAC_OF_FINAL * dmax) if np.isfinite(dmax) and dmax > 0 else EPS
    out = np.full_like(num, np.nan)
    mask = den > floor
    out[mask] = num[mask] / den[mask]
    return out


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

# ===================== 读入 =====================
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

# ===================== 计算 =====================
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
    out["W_tot"] = cumtrapz0(e, out["stress_sm"].to_numpy(float))
    out["W_serr"] = cumtrapz0(e, np.abs(out["delta_sigma_serr_sm"].to_numpy(float)))
    out["phi_serr"] = stable_ratio(out["W_serr"].to_numpy(float), out["W_tot"].to_numpy(float))
    return out


def align_mech_pol(mech: pd.DataFrame, pol: pd.DataFrame) -> pd.DataFrame:
    if "Time" in mech.columns and "Time" in pol.columns:
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


def add_pol_metrics(state: pd.DataFrame, name: str) -> pd.DataFrame:
    out = state.copy()
    req(out, ["Px_mean", "Py_mean", "Pz_mean", "absPx_mean", "absPy_mean", "absPz_mean", "absP_mean"], f"{name} overall")
    out["P_parallel"] = out["Py_mean"]
    out["O_parallel"] = out["absPy_mean"]
    out["Pmag_mean"] = out["absP_mean"]
    out["Pvec_norm"] = np.sqrt(out["Px_mean"] ** 2 + out["Py_mean"] ** 2 + out["Pz_mean"] ** 2)
    out["chi_parallel"] = np.abs(out["P_parallel"]) / (out["O_parallel"] + EPS)
    out["chi_vec"] = out["Pvec_norm"] / (out["Pmag_mean"] + EPS)
    denom_comp = out["absPx_mean"] + out["absPy_mean"] + out["absPz_mean"] + EPS
    out["eta_perp"] = (out["absPx_mean"] + out["absPz_mean"]) / denom_comp

    for c in ["P_parallel", "O_parallel", "chi_parallel", "chi_vec", "eta_perp"]:
        out[f"{c}_sm"] = smooth(out[c].to_numpy(float), GLOBAL_SMOOTH_WINDOW)

    out["P_parallel_abs_sm"] = smooth(np.abs(out["P_parallel"].to_numpy(float)), GLOBAL_SMOOTH_WINDOW)
    out["O_parallel_rel"] = out["O_parallel_sm"] / (float(out["O_parallel_sm"].iloc[0]) + EPS)
    out["chi_parallel_rel"] = out["chi_parallel_sm"] / (float(out["chi_parallel_sm"].iloc[0]) + EPS)
    return out


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


def build_events(state: pd.DataFrame, name: str) -> pd.DataFrame:
    cols = [
        "dataset", "event_id", "e_start", "e_end", "duration_e", "s_onset", "s_pre", "s_post",
        "Delta_s_drop", "W_sigma", "Delta_P_parallel", "Delta_O_parallel", "Delta_chi_parallel"
    ]
    segs = event_windows(state)
    if not segs:
        return pd.DataFrame(columns=cols)

    e = state["strain"].to_numpy(float)
    s = state["stress_sm"].to_numpy(float)
    ds = state["delta_sigma_serr_sm"].to_numpy(float)

    def pick(c, i):
        return float(state[c].iloc[i]) if c in state.columns else np.nan

    rows = []
    for k, (i0, i1) in enumerate(segs, 1):
        ipre, ipost = max(0, i0 - EVENT_PREPOST_PAD), min(len(state) - 1, i1 + EVENT_PREPOST_PAD)
        sl = slice(i0, i1 + 1)
        es, ss, dss = e[sl], s[sl], ds[sl]
        imax = int(np.argmax(ss))
        spre = float(ss[imax])
        spost = float(np.nanmin(ss[imax:])) if imax < len(ss) else float(ss[-1])
        rows.append({
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
            "Delta_P_parallel": pick("P_parallel_sm", ipost) - pick("P_parallel_sm", ipre),
            "Delta_O_parallel": pick("O_parallel_sm", ipost) - pick("O_parallel_sm", ipre),
            "Delta_chi_parallel": pick("chi_parallel_sm", ipost) - pick("chi_parallel_sm", ipre),
        })
    return pd.DataFrame(rows)


def build_spectrum(events: pd.DataFrame, name: str) -> pd.DataFrame:
    if events is None or len(events) == 0:
        return pd.DataFrame(columns=["dataset", "s", "G_N", "G_W"])
    ev = events.sort_values("s_onset").reset_index(drop=True)
    s = np.sort(ev["s_onset"].to_numpy(float))
    w = np.where(ev["W_sigma"].to_numpy(float) > 0, ev["W_sigma"].to_numpy(float), 1.0)
    grid = np.unique(s)
    return pd.DataFrame({
        "dataset": name,
        "s": grid,
        "G_N": [np.mean(s <= x) for x in grid],
        "G_W": [np.sum(w[s <= x]) / (np.sum(w) + EPS) for x in grid],
    })

# ===================== 流程 =====================
def prepare_dataset(name: str, cfg: dict) -> dict:
    mech = add_mech_metrics(clean_mech(read_ws(cfg["ss"]), f"{name} mechanical"))
    overall = clean_pol(read_vars(cfg["overall"]), f"{name} overall")
    state = add_pol_metrics(align_mech_pol(mech, overall), name)
    state["dataset"] = name
    events = build_events(state, name)
    spectrum = build_spectrum(events, name)
    return {"name": name, "state": state, "events": events, "spectrum": spectrum}

# ===================== 作图 =====================
def fig1(single: dict, twin: dict):
    s, t = single["state"], twin["state"]
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 9.0), sharex=True, gridspec_kw={"hspace": 0.05}, constrained_layout=True)
    for d, c, name in [(s, C_SINGLE, "Single"), (t, C_TWIN, "Twin")]:
        ax[0].plot(d["strain"], d["W_tot"], color=c, lw=2.0, label=f"{name} W_tot")
        ax[0].plot(d["strain"], d["W_serr"], color=c, lw=1.4, ls="--", label=f"{name} W_serr")
        ax[1].plot(d["strain"], d["phi_serr"], color=c, lw=2.0, label=f"{name} phi_serr")
        ax[2].plot(d["strain"], d["S_rms"], color=c, lw=1.6, label=f"{name} S_rms")
        ax[2].plot(d["strain"], d["A_sigma_norm"], color=c, lw=1.2, ls="--", label=f"{name} A_sigma_norm")
    ax[0].set_ylabel("Work-like density")
    ax[0].set_title("Figure 1a. Robust work-like evidence")
    ax[0].legend(fontsize=8.5, ncol=2)
    ax[1].set_ylim(bottom=0)
    ax[1].set_ylabel("W_serr / W_tot")
    ax[1].set_title("Figure 1b. Serration-work fraction")
    ax[1].legend(fontsize=8.5, ncol=2)
    ax[2].set_ylim(bottom=0)
    ax[2].set_xlabel("Compressive strain e")
    ax[2].set_ylabel("Activity proxy")
    ax[2].set_title("Figure 1c. S_rms and cumulative activity")
    ax[2].legend(fontsize=8.5, ncol=2)
    fig.savefig(OUT_DIR / "Figure1_mech_robust.png", bbox_inches="tight")


def fig2(single: dict, twin: dict):
    s, t = single["state"], twin["state"]
    fig, ax = plt.subplots(3, 1, figsize=(8.2, 9.0), sharex=True, gridspec_kw={"hspace": 0.05}, constrained_layout=True)
    for d, c, name in [(s, C_SINGLE, "Single"), (t, C_TWIN, "Twin")]:
        ax[0].plot(d["strain"], d["P_parallel_abs_sm"], color=c, lw=2.0, label=f"{name} |P_parallel|")
        ax[0].plot(d["strain"], d["O_parallel_sm"], color=c, lw=1.2, ls="--", label=f"{name} O_parallel")
        ax[0].plot(d["strain"], d["chi_parallel_sm"], color=c, lw=1.2, ls=":", label=f"{name} chi_parallel")
        ax[1].plot(d["strain"], d["O_parallel_rel"], color=c, lw=1.8, label=f"{name} O_parallel / O0")
        ax[1].plot(d["strain"], d["chi_parallel_rel"], color=c, lw=1.2, ls="--", label=f"{name} chi_parallel / chi0")
        ax[2].plot(d["strain"], d["chi_vec_sm"], color=c, lw=2.0, label=f"{name} chi_vec")
        ax[2].plot(d["strain"], d["eta_perp_sm"], color=c, lw=1.4, ls="--", label=f"{name} eta_perp")
    ax[0].set_ylabel("Parallel-channel metrics")
    ax[0].set_title("Figure 2a. |P_parallel| = O_parallel * chi_parallel")
    ax[0].legend(fontsize=8.2, ncol=2)
    ax[1].set_ylabel("Relative organization")
    ax[1].set_title("Figure 2b. Occupancy and coherence evolution")
    ax[1].legend(fontsize=8.2, ncol=2)
    ax[2].set_xlabel("Compressive strain e")
    ax[2].set_ylabel("Organization metrics")
    ax[2].set_title("Figure 2c. Vector coherence and transverse redistribution")
    ax[2].legend(fontsize=8.2, ncol=2)
    fig.savefig(OUT_DIR / "Figure2_polarization_organization.png", bbox_inches="tight")


def fig3(single: dict, twin: dict):
    es, et = single["events"], twin["events"]
    ss, st = single["spectrum"], twin["spectrum"]
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 7.2), gridspec_kw={"hspace": 0.10}, constrained_layout=True)
    if len(ss):
        ax[0].plot(ss["s"], ss["G_N"], color=C_SINGLE, lw=2.0, label="Single G_N")
        ax[0].plot(ss["s"], ss["G_W"], color=C_SINGLE, lw=1.4, ls="--", label="Single G_W")
    if len(st):
        ax[0].plot(st["s"], st["G_N"], color=C_TWIN, lw=2.0, label="Twin G_N")
        ax[0].plot(st["s"], st["G_W"], color=C_TWIN, lw=1.4, ls="--", label="Twin G_W")
    ax[0].set_ylabel("Cumulative unlocking")
    ax[0].set_title("Figure 3a. Fast-serration threshold spectrum")
    ax[0].legend(fontsize=9, ncol=2)

    def ccdf(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0)]
        if len(v) <= 1:
            return np.array([]), np.array([])
        v = np.sort(v)
        return v, 1.0 - np.arange(1, len(v) + 1) / len(v)

    xs, ys = ccdf(es["W_sigma"].to_numpy(float) if len(es) else np.array([]))
    xt, yt = ccdf(et["W_sigma"].to_numpy(float) if len(et) else np.array([]))
    if len(xs):
        ax[1].plot(xs, ys, color=C_SINGLE, lw=2.0, label="Single CCDF(W_sigma)")
    if len(xt):
        ax[1].plot(xt, yt, color=C_TWIN, lw=2.0, label="Twin CCDF(W_sigma)")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("Event work proxy W_sigma")
    ax[1].set_ylabel("Pr(W_sigma > W)")
    ax[1].set_title("Figure 3b. Event-work CCDF")
    ax[1].legend(fontsize=9)
    fig.savefig(OUT_DIR / "Figure3_fast_event_spectra.png", bbox_inches="tight")

# ===================== 汇总 =====================
def summarize(ds: dict) -> dict:
    st, ev, sp, name = ds["state"], ds["events"], ds["spectrum"], ds["name"]
    out = {
        "dataset": name,
        "onset_sigma": float(st["onset_sigma"].iloc[0]) if "onset_sigma" in st.columns else np.nan,
        "W_tot_end": float(st["W_tot"].iloc[-1]),
        "W_serr_end": float(st["W_serr"].iloc[-1]),
        "phi_serr_end": float(np.nan_to_num(st["phi_serr"].iloc[-1], nan=np.nan)),
        "N_events": int(len(ev)),
        "median_W_sigma": float(np.nanmedian(ev["W_sigma"])) if len(ev) else np.nan,
        "median_onset_stress": float(np.nanmedian(ev["s_onset"])) if len(ev) else np.nan,
        "max_W_sigma": float(np.nanmax(ev["W_sigma"])) if len(ev) else np.nan,
        "s10": np.nan,
        "s50": np.nan,
        "s90": np.nan,
        "B_s": np.nan,
    }
    if len(sp):
        x, y = sp["s"].to_numpy(float), sp["G_N"].to_numpy(float)
        for q, key in [(0.10, "s10"), (0.50, "s50"), (0.90, "s90")]:
            idx = np.where(y >= q)[0]
            out[key] = float(x[idx[0]]) if len(idx) else np.nan
        if np.isfinite(out["s10"]) and np.isfinite(out["s90"]):
            out["B_s"] = out["s90"] - out["s10"]
    return out


def write_report(single: dict, twin: dict):
    pd.DataFrame([summarize(single), summarize(twin)]).to_csv(OUT_DIR / "summary_minimal.csv", index=False)
    txt = []
    txt += ["# Minimal robust report\n\n"]
    txt += ["- Main mechanical evidence: W_tot, W_serr, phi_serr, S_rms, A_sigma_norm\n"]
    txt += ["- Main polarization evidence: |P_parallel|, O_parallel, chi_parallel, O_parallel/O0, chi_parallel/chi0, chi_vec, eta_perp\n"]
    txt += ["- Main event evidence: fast-serration threshold spectrum G_N/G_W and CCDF(W_sigma)\n"]
    txt += ["- Not used as main evidence: W_acc, a_acc, e_acc, H_sec, derivative contribution plots, event-wise scatter plots\n"]
    (OUT_DIR / "report_minimal.md").write_text("".join(txt), encoding="utf-8")

# ===================== 主程序 =====================
def main():
    print("=" * 72)
    print(">>> BaTiO3 minimal robust analysis")
    print("=" * 72)
    print(f"工作目录: {BASE_DIR}")
    print(f"输出目录: {OUT_DIR}\n")

    single = prepare_dataset("Single", FILES["Single"])
    twin = prepare_dataset("Twin", FILES["Twin"])

    single["state"].to_csv(OUT_DIR / "state_single.csv", index=False)
    twin["state"].to_csv(OUT_DIR / "state_twin.csv", index=False)
    single["events"].to_csv(OUT_DIR / "events_single.csv", index=False)
    twin["events"].to_csv(OUT_DIR / "events_twin.csv", index=False)
    single["spectrum"].to_csv(OUT_DIR / "threshold_spectrum_single.csv", index=False)
    twin["spectrum"].to_csv(OUT_DIR / "threshold_spectrum_twin.csv", index=False)

    fig1(single, twin)
    fig2(single, twin)
    fig3(single, twin)
    write_report(single, twin)

    print("运行完成：Figure1~3、state/events/spectrum CSV、summary_minimal.csv、report_minimal.md 已输出。")
    plt.show()


if __name__ == "__main__":
    main()
