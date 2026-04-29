#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaTiO3 Ref. / 90° DW 压缩响应 Figure 4 与 SI 验证脚本
====================================================

用途
----
本脚本是在旧版 M2y/m_y 分析脚本基础上重写的 Figure 4 专用版本。
它保留原始数据读取、力学-极化数据合并、M2y/m_y 计算逻辑，
但重写了拟合和输出部分，用于支持正文 Figure 4 与 Supplementary validation。

核心写作定位
------------
这不是完整热力学方程或可迁移本构模型，而是一个 semi-quantitative
field-to-stress consistency test：

    local Py reorganization
        -> loading-axis polarization strength trajectory
        -> dominant stress trend

主要比较三类模型：

Model I: strain-only baseline
    s(e) ≈ A_e e
    s(e) ≈ A_e e + B_e e^2

Model II: common-coefficient reduced relation
    Ref. 和 90° DW 共用同一组参数：
    s(e) ≈ A_e e - A_M [1 - m_y(e)]

Model III: structure-dependent reduced relation
    Ref. 和 90° DW 使用相同形式，但允许 effective coefficients 不同：
    s_i(e) ≈ A_{e,i} e - A_{M,i} [1 - m_{y,i}(e)]

注意
----
1. A_e 不应写成本征弹性模量；建议写为 effective mechanical projection coefficient。
2. A_M 不应写成本征电致伸缩系数或 Q coefficient；建议写为
   effective polarization-strength-to-stress projection coefficient。
3. reduced relation 只能写成 consistency test / trend-level relation，
   不能写成 universal constitutive law。
4. 输出中的 single/twin 只作为 legacy source labels；论文写作使用 Ref. / 90° DW。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1.0e-15

# =============================================================================
# 用户配置区
# =============================================================================

# 原始数据文件夹。脚本默认放在项目根目录下运行，并读取 ./compute/ 下的数据。
DATA_FOLDER_NAME = "compute"

# 输出文件夹名称会自动使用该后缀。
OUTPUT_SUFFIX = "figure4_SI"

# 是否显示图像。Spyder 中建议 True；服务器批处理可改为 False。
SHOW_FIGURES = True

# 拟合默认不加截距。若初始应变处存在显著非零残余应力，可以改为 True 做敏感性检查。
USE_INTERCEPT = False

# 论文命名与旧文件命名的映射。
# 文件仍然是 single/twin，但论文统一写 Ref. / 90° DW。
CASE_CONFIG = {
    "Ref": {
        "legacy": "single",
        "mech_file": "single_ss_y_inst_2pct_cont.txt",
        "overall_file": "Pol_Pro_stats_overall_single.dat",
        "sections_file": "Pol_Pro_stats_sections_single.dat",
    },
    "90DW": {
        "legacy": "twin",
        "mech_file": "twin_ss_y_inst_2pct_cont.txt",
        "overall_file": "Pol_Pro_stats_overall_twin.dat",
        "sections_file": "Pol_Pro_stats_sections_twin.dat",
    },
}

CASE_ORDER = ["Ref", "90DW"]
CASE_LABEL = {"Ref": "Ref.", "90DW": "90° DW"}


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class CasePaths:
    case: str
    legacy: str
    mech: Path
    overall: Path
    sections: Path


@dataclass
class CaseResult:
    case: str
    legacy: str
    merged: pd.DataFrame
    sections: pd.DataFrame
    summary: Dict[str, float]


# =============================================================================
# 基础读取函数
# =============================================================================

def clean_var_name(name: str) -> str:
    """清理并标准化列名中的特殊字符。"""
    name = str(name).strip()
    name = name.replace("|Px|", "absPx")
    name = name.replace("|Py|", "absPy")
    name = name.replace("|Pz|", "absPz")
    name = name.replace("|P|", "absP")
    name = name.replace(" ", "")
    name = name.replace("/", "_per_")
    name = name.replace("%", "pct")
    name = name.replace("-", "_")
    return name


def read_mech_txt(path: Path) -> pd.DataFrame:
    """读取宏观力学 dump 数据文件。"""
    df = pd.read_csv(path, sep=r"\s+", comment="#", engine="python")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "TimeStep" not in df.columns:
        raise ValueError(f"{path} 缺少 TimeStep 列")
    df = (
        df.dropna(subset=["TimeStep"])
          .drop_duplicates(subset=["TimeStep"], keep="first")
          .sort_values("TimeStep")
          .reset_index(drop=True)
    )
    return df


def read_stats_table(path: Path) -> pd.DataFrame:
    """读取极化统计数据文件，兼容 Tecplot VARIABLES/ZONE 与普通表头文本。"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header: List[str] = []
    data_lines: List[List[str]] = []

    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        var_match = re.match(r"^VARIABLES\s*=\s*(.*)", s, re.IGNORECASE)
        if var_match:
            var_str = var_match.group(1)
            if '"' in var_str:
                header = re.findall(r'"([^"]+)"', var_str)
            else:
                header = var_str.split()
            continue

        if s.upper().startswith("ZONE"):
            continue

        if not header and not data_lines:
            header = [x.strip('"') for x in s.split()]
            continue

        data_lines.append(s.split())

    if not header:
        raise ValueError(f"{path} 文件为空或无法识别表头")

    clean_data: List[List[object]] = []
    for row in data_lines:
        if len(row) >= len(header):
            clean_data.append(row[:len(header)])
        else:
            clean_data.append(row + [np.nan] * (len(header) - len(row)))

    df = pd.DataFrame(clean_data, columns=header)

    new_cols: List[str] = []
    seen = set()
    for col in df.columns:
        clean_col = clean_var_name(col)
        base_col = clean_col
        i = 1
        while clean_col in seen:
            clean_col = f"{base_col}_{i}"
            i += 1
        seen.add(clean_col)
        new_cols.append(clean_col)
    df.columns = new_cols

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    time_col = "Time" if "Time" in df.columns else ("TimeStep" if "TimeStep" in df.columns else None)
    if time_col is None:
        raise ValueError(f"{path} 缺少 Time 或 TimeStep 列。解析出的表头为: {list(df.columns)}")

    df = (
        df.dropna(subset=[time_col])
          .drop_duplicates(subset=[time_col], keep="first")
          .sort_values(time_col)
          .reset_index(drop=True)
    )

    if time_col != "TimeStep":
        df = df.rename(columns={time_col: "TimeStep"})

    return df


# =============================================================================
# 简易信号处理与评价指标
# =============================================================================

def odd_window(n: int, wanted: int) -> int:
    if n < 3:
        return 1
    w = max(3, min(wanted, n if n % 2 == 1 else n - 1))
    if w % 2 == 0:
        w -= 1
    return max(3, w)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    w = odd_window(len(x), window)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def running_rms(x: np.ndarray, window: int) -> np.ndarray:
    return np.sqrt(np.maximum(moving_average(np.asarray(x, dtype=float) ** 2, window), 0.0))


def safe_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    if len(y) < 2:
        return float("nan")
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / denom)


def regression_metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    """返回 RMSE/MAE/R²/归一化误差等指标。"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat)
    y = y[mask]
    yhat = yhat[mask]
    if len(y) == 0:
        return {
            "n": 0,
            "rmse": np.nan,
            "mae": np.nan,
            "max_abs_error": np.nan,
            "r2": np.nan,
            "nrmse_range": np.nan,
            "nrmse_mean_abs": np.nan,
        }
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    max_abs = float(np.max(np.abs(resid)))
    y_range = float(np.max(y) - np.min(y))
    mean_abs = float(np.mean(np.abs(y)))
    return {
        "n": int(len(y)),
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "r2": safe_r2(y, yhat),
        "nrmse_range": rmse / y_range if y_range > EPS else np.nan,
        "nrmse_mean_abs": rmse / mean_abs if mean_abs > EPS else np.nan,
    }


def design_with_optional_intercept(X: np.ndarray) -> np.ndarray:
    """按全局开关给设计矩阵添加截距列。"""
    X = np.asarray(X, dtype=float)
    if USE_INTERCEPT:
        X = np.column_stack([np.ones(len(X)), X])
    return X


def fit_ols(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """最小二乘拟合。"""
    X = design_with_optional_intercept(X)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return coef, yhat


def predict_ols(coef: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = design_with_optional_intercept(X)
    return X @ coef


# =============================================================================
# 数据构建
# =============================================================================

def discover_paths(root: Path, folder_name: str = DATA_FOLDER_NAME) -> Dict[str, CasePaths]:
    compute_dir = root / folder_name
    paths: Dict[str, CasePaths] = {}
    missing: List[str] = []

    for case, cfg in CASE_CONFIG.items():
        cp = CasePaths(
            case=case,
            legacy=cfg["legacy"],
            mech=compute_dir / cfg["mech_file"],
            overall=compute_dir / cfg["overall_file"],
            sections=compute_dir / cfg["sections_file"],
        )
        paths[case] = cp
        for label in ("mech", "overall", "sections"):
            p = getattr(cp, label)
            if not p.exists():
                missing.append(f"{case}:{label}:{p}")

    if missing:
        raise FileNotFoundError("未找到必须的数据文件:\n  - " + "\n  - ".join(missing))
    return paths


def build_case(paths: CasePaths) -> CaseResult:
    """读取并合并单个案例的数据，计算 M2y/m_y 与加载段标记。"""
    mech = read_mech_txt(paths.mech)
    overall = read_stats_table(paths.overall)
    sections = read_stats_table(paths.sections)

    required_overall = ["Py_mean", "Py_var", "absPy_mean"]
    missing_overall = [c for c in required_overall if c not in overall.columns]
    if missing_overall:
        raise ValueError(f"{paths.overall} 缺少以下列: {missing_overall}")

    merged = mech.merge(overall, on="TimeStep", how="inner", validate="one_to_one")
    if len(merged) < 20:
        raise ValueError(f"{paths.case} 案例的对齐数据行数过少: {len(merged)}")

    if "s2" not in merged.columns or "p2" not in merged.columns:
        raise ValueError(f"{paths.case} 缺少 s2 或 p2 列，无法得到 y 轴应变/应力。")
    merged["strain_y"] = merged["s2"].abs()
    merged["stress_y"] = merged["p2"].abs()

    merged["M2y"] = merged["Py_var"] + merged["Py_mean"] ** 2
    merged["M2y_sqrt"] = np.sqrt(np.maximum(merged["M2y"], 0.0))
    M2y0 = float(merged["M2y"].iloc[0])
    merged["m_y"] = merged["M2y"] / max(M2y0, EPS)
    merged["progress_M2y"] = 1.0 - merged["m_y"]

    merged["absPy_mean_norm"] = merged["absPy_mean"] / max(float(merged["absPy_mean"].iloc[0]), EPS)
    merged["Py_mean_norm"] = merged["Py_mean"] / max(abs(float(merged["Py_mean"].iloc[0])), EPS)

    merged["stress_smooth"] = moving_average(merged["stress_y"].to_numpy(), 11)
    merged["delta_sigma_serr"] = merged["stress_y"] - merged["stress_smooth"]
    merged["S_rms"] = running_rms(merged["delta_sigma_serr"].to_numpy(), 11)

    # section-wise M2 heterogeneity 只保留为可选诊断输出；不生成 Fig.3 SI maps。
    for sec in ("xy", "xz", "yz"):
        req = [f"Py_mean_{sec}", f"Py_var_{sec}"]
        missing = [c for c in req if c not in sections.columns]
        if missing:
            raise ValueError(f"{paths.sections} 缺少截面列 {missing}")
        sections[f"M2y_{sec}"] = sections[f"Py_var_{sec}"] + sections[f"Py_mean_{sec}"] ** 2

    section_m2_cols = [f"M2y_{sec}" for sec in ("xy", "xz", "yz")]
    sections["H_M2_sections"] = sections[section_m2_cols].std(axis=1) / np.maximum(
        sections[section_m2_cols].mean(axis=1), EPS
    )
    merged = merged.merge(
        sections[["TimeStep", "H_M2_sections"]],
        on="TimeStep",
        how="left",
        validate="one_to_one",
    )

    idx_max_pos = int(np.argmax(merged["strain_y"].to_numpy()))
    merged["is_loading"] = False
    merged.loc[:idx_max_pos, "is_loading"] = True

    summary = {
        "n_mech": float(len(mech)),
        "n_overall": float(len(overall)),
        "n_sections": float(len(sections)),
        "n_merged": float(len(merged)),
        "n_loading": float(int(merged["is_loading"].sum())),
        "strain_start": float(merged["strain_y"].iloc[0]),
        "strain_max": float(merged["strain_y"].max()),
        "stress_start": float(merged["stress_y"].iloc[0]),
        "stress_max_loading": float(merged.loc[merged["is_loading"], "stress_y"].max()),
        "M2y_start": float(M2y0),
        "m_y_end_loading": float(merged.loc[merged["is_loading"], "m_y"].iloc[-1]),
        "H_M2_start": float(merged["H_M2_sections"].iloc[0]),
        "H_M2_end": float(merged["H_M2_sections"].iloc[-1]),
    }

    return CaseResult(case=paths.case, legacy=paths.legacy, merged=merged, sections=sections, summary=summary)


def get_loading_arrays(results: Dict[str, CaseResult]) -> Dict[str, Dict[str, np.ndarray]]:
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for case in CASE_ORDER:
        df = results[case].merged.loc[results[case].merged["is_loading"]].copy()
        data[case] = {
            "e": df["strain_y"].to_numpy(dtype=float),
            "s": df["stress_y"].to_numpy(dtype=float),
            "p": df["progress_M2y"].to_numpy(dtype=float),
            "m": df["m_y"].to_numpy(dtype=float),
        }
    return data


# =============================================================================
# Figure 4 / SI 模型比较
# =============================================================================

def add_prediction_column(results: Dict[str, CaseResult], col: str, case: str, values: np.ndarray) -> None:
    results[case].merged[col] = values
    results[case].merged[f"resid_{col}"] = results[case].merged["stress_y"] - results[case].merged[col]


def add_metric_row(rows: List[Dict[str, object]], model: str, case: str, y: np.ndarray, yhat: np.ndarray) -> None:
    rows.append({"model": model, "case": case, **regression_metrics(y, yhat)})


def add_coeff_row(rows: List[Dict[str, object]], model: str, case: str, name: str, value: float) -> None:
    rows.append({"model": model, "case": case, "coefficient": name, "value": float(value)})


def collect_overall_metric(results: Dict[str, CaseResult], pred_col: str) -> Tuple[np.ndarray, np.ndarray]:
    y_all: List[np.ndarray] = []
    yh_all: List[np.ndarray] = []
    for case in CASE_ORDER:
        df = results[case].merged.loc[results[case].merged["is_loading"]]
        y_all.append(df["stress_y"].to_numpy(dtype=float))
        yh_all.append(df[pred_col].to_numpy(dtype=float))
    return np.concatenate(y_all), np.concatenate(yh_all)


def fit_all_models(results: Dict[str, CaseResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """执行所有 Figure 4/SI 模型比较，并把预测列写回 results[case].merged。"""
    data = get_loading_arrays(results)
    coeff_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []

    # Model I-a: linear strain-only baseline, separate by case: s = Ae * e
    model = "M1_linear_strain_only_separate"
    for case in CASE_ORDER:
        e, s = data[case]["e"], data[case]["s"]
        coef, _ = fit_ols(s, e.reshape(-1, 1))
        Ae = coef[-1]
        pred_all = predict_ols(coef, results[case].merged["strain_y"].to_numpy().reshape(-1, 1))
        add_prediction_column(results, "pred_M1_linear_sep", case, pred_all)
        add_coeff_row(coeff_rows, model, case, "Ae_effective", Ae)
        df_load = results[case].merged.loc[results[case].merged["is_loading"]]
        add_metric_row(metric_rows, model, case, df_load["stress_y"], df_load["pred_M1_linear_sep"])
    y, yh = collect_overall_metric(results, "pred_M1_linear_sep")
    add_metric_row(metric_rows, model, "pooled", y, yh)

    # Model I-b: quadratic strain-only baseline, separate by case: s = Ae * e + Be * e^2
    model = "M1_quadratic_strain_only_separate"
    for case in CASE_ORDER:
        e, s = data[case]["e"], data[case]["s"]
        X = np.column_stack([e, e ** 2])
        coef, _ = fit_ols(s, X)
        Ae, Be = coef[-2], coef[-1]
        e_all = results[case].merged["strain_y"].to_numpy()
        X_all = np.column_stack([e_all, e_all ** 2])
        pred_all = predict_ols(coef, X_all)
        add_prediction_column(results, "pred_M1_quadratic_sep", case, pred_all)
        add_coeff_row(coeff_rows, model, case, "Ae_effective", Ae)
        add_coeff_row(coeff_rows, model, case, "Be_effective", Be)
        df_load = results[case].merged.loc[results[case].merged["is_loading"]]
        add_metric_row(metric_rows, model, case, df_load["stress_y"], df_load["pred_M1_quadratic_sep"])
    y, yh = collect_overall_metric(results, "pred_M1_quadratic_sep")
    add_metric_row(metric_rows, model, "pooled", y, yh)

    # Model II: common-coefficient reduced relation: s = Ae * e - AM * progress
    model = "M2_common_coeff_reduced"
    e_all = np.concatenate([data[c]["e"] for c in CASE_ORDER])
    p_all = np.concatenate([data[c]["p"] for c in CASE_ORDER])
    s_all = np.concatenate([data[c]["s"] for c in CASE_ORDER])
    X = np.column_stack([e_all, -p_all])
    coef, _ = fit_ols(s_all, X)
    Ae_common, AM_common = coef[-2], coef[-1]
    add_coeff_row(coeff_rows, model, "pooled", "Ae_common_effective", Ae_common)
    add_coeff_row(coeff_rows, model, "pooled", "AM_common_effective", AM_common)
    for case in CASE_ORDER:
        df = results[case].merged
        X_all = np.column_stack([df["strain_y"].to_numpy(), -df["progress_M2y"].to_numpy()])
        pred_all = predict_ols(coef, X_all)
        add_prediction_column(results, "pred_M2_common", case, pred_all)
        df_load = df.loc[df["is_loading"]]
        add_metric_row(metric_rows, model, case, df_load["stress_y"], df_load["pred_M2_common"])
    y, yh = collect_overall_metric(results, "pred_M2_common")
    add_metric_row(metric_rows, model, "pooled", y, yh)

    # Model III: structure-dependent reduced relation: s_i = Ae_i * e - AM_i * progress
    model = "M3_structure_dependent_reduced"
    separate_coeffs: Dict[str, Tuple[float, float]] = {}
    for case in CASE_ORDER:
        e, p, s = data[case]["e"], data[case]["p"], data[case]["s"]
        X = np.column_stack([e, -p])
        coef, _ = fit_ols(s, X)
        Ae_i, AM_i = coef[-2], coef[-1]
        separate_coeffs[case] = (Ae_i, AM_i)
        add_coeff_row(coeff_rows, model, case, "Ae_structure_effective", Ae_i)
        add_coeff_row(coeff_rows, model, case, "AM_structure_effective", AM_i)
        df = results[case].merged
        X_all = np.column_stack([df["strain_y"].to_numpy(), -df["progress_M2y"].to_numpy()])
        pred_all = predict_ols(coef, X_all)
        add_prediction_column(results, "pred_M3_structure", case, pred_all)
        df_load = df.loc[df["is_loading"]]
        add_metric_row(metric_rows, model, case, df_load["stress_y"], df_load["pred_M3_structure"])
    y, yh = collect_overall_metric(results, "pred_M3_structure")
    add_metric_row(metric_rows, model, "pooled", y, yh)

    # Constrained fit A: common Ae, separate AM
    model = "M4_constrained_common_Ae_separate_AM"
    rows_X: List[List[float]] = []
    rows_y: List[float] = []
    for case in CASE_ORDER:
        e, p, s = data[case]["e"], data[case]["p"], data[case]["s"]
        for ei, pi, si in zip(e, p, s):
            rows_X.append([ei, -pi if case == "Ref" else 0.0, -pi if case == "90DW" else 0.0])
            rows_y.append(si)
    coef, _ = fit_ols(np.asarray(rows_y), np.asarray(rows_X))
    Ae_c, AM_ref, AM_90 = coef[-3], coef[-2], coef[-1]
    add_coeff_row(coeff_rows, model, "pooled", "Ae_common_effective", Ae_c)
    add_coeff_row(coeff_rows, model, "Ref", "AM_Ref_effective", AM_ref)
    add_coeff_row(coeff_rows, model, "90DW", "AM_90DW_effective", AM_90)
    for case in CASE_ORDER:
        df = results[case].merged
        e = df["strain_y"].to_numpy()
        p = df["progress_M2y"].to_numpy()
        if case == "Ref":
            X_all = np.column_stack([e, -p, np.zeros_like(p)])
        else:
            X_all = np.column_stack([e, np.zeros_like(p), -p])
        pred_all = predict_ols(coef, X_all)
        add_prediction_column(results, "pred_M4_common_Ae", case, pred_all)
        df_load = df.loc[df["is_loading"]]
        add_metric_row(metric_rows, model, case, df_load["stress_y"], df_load["pred_M4_common_Ae"])
    y, yh = collect_overall_metric(results, "pred_M4_common_Ae")
    add_metric_row(metric_rows, model, "pooled", y, yh)

    # Constrained fit B: separate Ae, common AM
    model = "M5_constrained_separate_Ae_common_AM"
    rows_X = []
    rows_y = []
    for case in CASE_ORDER:
        e, p, s = data[case]["e"], data[case]["p"], data[case]["s"]
        for ei, pi, si in zip(e, p, s):
            rows_X.append([ei if case == "Ref" else 0.0, ei if case == "90DW" else 0.0, -pi])
            rows_y.append(si)
    coef, _ = fit_ols(np.asarray(rows_y), np.asarray(rows_X))
    Ae_ref, Ae_90, AM_c = coef[-3], coef[-2], coef[-1]
    add_coeff_row(coeff_rows, model, "Ref", "Ae_Ref_effective", Ae_ref)
    add_coeff_row(coeff_rows, model, "90DW", "Ae_90DW_effective", Ae_90)
    add_coeff_row(coeff_rows, model, "pooled", "AM_common_effective", AM_c)
    for case in CASE_ORDER:
        df = results[case].merged
        e = df["strain_y"].to_numpy()
        p = df["progress_M2y"].to_numpy()
        if case == "Ref":
            X_all = np.column_stack([e, np.zeros_like(e), -p])
        else:
            X_all = np.column_stack([np.zeros_like(e), e, -p])
        pred_all = predict_ols(coef, X_all)
        add_prediction_column(results, "pred_M5_common_AM", case, pred_all)
        df_load = df.loc[df["is_loading"]]
        add_metric_row(metric_rows, model, case, df_load["stress_y"], df_load["pred_M5_common_AM"])
    y, yh = collect_overall_metric(results, "pred_M5_common_AM")
    add_metric_row(metric_rows, model, "pooled", y, yh)

    # Cross-prediction: 用一个结构的 Model III 参数预测另一个结构。
    model = "M6_cross_prediction_using_other_structure_coeffs"
    for target_case in CASE_ORDER:
        source_case = "90DW" if target_case == "Ref" else "Ref"
        Ae_s, AM_s = separate_coeffs[source_case]
        df = results[target_case].merged
        pred_all = Ae_s * df["strain_y"].to_numpy() - AM_s * df["progress_M2y"].to_numpy()
        col = f"pred_M6_cross_from_{source_case}"
        add_prediction_column(results, col, target_case, pred_all)
        df_load = df.loc[df["is_loading"]]
        add_metric_row(metric_rows, model, target_case, df_load["stress_y"], df_load[col])
        add_coeff_row(coeff_rows, model, target_case, f"Ae_from_{source_case}", Ae_s)
        add_coeff_row(coeff_rows, model, target_case, f"AM_from_{source_case}", AM_s)

    return pd.DataFrame(coeff_rows), pd.DataFrame(metric_rows)


# =============================================================================
# 绘图函数
# =============================================================================

def save_or_show(fig: plt.Figure, outpath: Path) -> None:
    fig.tight_layout()
    fig.savefig(outpath, dpi=240, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def plot_main_figure4_candidate(results: Dict[str, CaseResult], outpath: Path) -> None:
    """正文 Figure 4 候选：m_y trajectory + common/structure-dependent consistency test。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    for case in CASE_ORDER:
        df = results[case].merged.loc[results[case].merged["is_loading"]]
        ax.plot(df["strain_y"], df["m_y"], label=CASE_LABEL[case])
    ax.set_xlabel(r"$|\epsilon_y|$")
    ax.set_ylabel(r"$m_y=M_{2,y}/M_{2,y}(0)$")
    ax.set_title("(a) Loading-axis polarization strength")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    for case in CASE_ORDER:
        df = results[case].merged.loc[results[case].merged["is_loading"]]
        ax.plot(df["strain_y"], df["stress_y"], label=f"{CASE_LABEL[case]} measured")
        ax.plot(df["strain_y"], df["pred_M2_common"], linestyle="--", label=f"{CASE_LABEL[case]} common")
        ax.plot(df["strain_y"], df["pred_M3_structure"], linestyle=":", label=f"{CASE_LABEL[case]} struct.-dep.")
    ax.set_xlabel(r"$|\epsilon_y|$")
    ax.set_ylabel(r"$|\sigma_y|$")
    ax.set_title("(b) Field-to-stress consistency test")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    save_or_show(fig, outpath)


def plot_si_model_comparison(results: Dict[str, CaseResult], outpath: Path) -> None:
    """SI: 三类模型对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, case in zip(axes, CASE_ORDER):
        df = results[case].merged.loc[results[case].merged["is_loading"]]
        ax.plot(df["strain_y"], df["stress_y"], label="measured")
        ax.plot(df["strain_y"], df["pred_M1_linear_sep"], linestyle="--", label="linear strain-only")
        ax.plot(df["strain_y"], df["pred_M1_quadratic_sep"], linestyle="--", label="quadratic strain-only")
        ax.plot(df["strain_y"], df["pred_M2_common"], linestyle="-.", label="common reduced")
        ax.plot(df["strain_y"], df["pred_M3_structure"], linestyle=":", label="structure-dependent reduced")
        ax.set_title(CASE_LABEL[case])
        ax.set_xlabel(r"$|\epsilon_y|$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$|\sigma_y|$")
    axes[0].legend(frameon=False, fontsize=8)
    save_or_show(fig, outpath)


def plot_si_residuals(results: Dict[str, CaseResult], outpath: Path) -> None:
    """SI: residual curves。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    pred_cols = [
        ("pred_M1_linear_sep", "linear strain-only"),
        ("pred_M1_quadratic_sep", "quadratic strain-only"),
        ("pred_M2_common", "common reduced"),
        ("pred_M3_structure", "structure-dependent reduced"),
    ]

    for ax, case in zip(axes, CASE_ORDER):
        df = results[case].merged.loc[results[case].merged["is_loading"]]
        for col, label in pred_cols:
            ax.plot(df["strain_y"], df["stress_y"] - df[col], label=label)
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title(CASE_LABEL[case])
        ax.set_xlabel(r"$|\epsilon_y|$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("residual")
    axes[0].legend(frameon=False, fontsize=8)
    save_or_show(fig, outpath)


def plot_si_constrained_and_cross(results: Dict[str, CaseResult], outpath: Path) -> None:
    """SI: constrained fits 与 cross-prediction。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, case in zip(axes, CASE_ORDER):
        df = results[case].merged.loc[results[case].merged["is_loading"]]
        cross_col = "pred_M6_cross_from_90DW" if case == "Ref" else "pred_M6_cross_from_Ref"
        ax.plot(df["strain_y"], df["stress_y"], label="measured")
        ax.plot(df["strain_y"], df["pred_M4_common_Ae"], linestyle="--", label="common Ae")
        ax.plot(df["strain_y"], df["pred_M5_common_AM"], linestyle="-.", label="common AM")
        ax.plot(df["strain_y"], df[cross_col], linestyle=":", label="cross prediction")
        ax.set_title(CASE_LABEL[case])
        ax.set_xlabel(r"$|\epsilon_y|$")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$|\sigma_y|$")
    axes[0].legend(frameon=False, fontsize=8)
    save_or_show(fig, outpath)


def plot_si_polarization_metrics(results: Dict[str, CaseResult], outpath: Path) -> None:
    """SI 可选：比较 <Py>, <|Py|>, M2y/m_y。"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for case in CASE_ORDER:
        df = results[case].merged.loc[results[case].merged["is_loading"]]
        axes[0].plot(df["strain_y"], df["Py_mean"], label=CASE_LABEL[case])
        axes[1].plot(df["strain_y"], df["absPy_mean_norm"], label=CASE_LABEL[case])
        axes[2].plot(df["strain_y"], df["m_y"], label=CASE_LABEL[case])

    axes[0].set_title(r"signed $\langle P_y\rangle$")
    axes[1].set_title(r"normalized $\langle |P_y|\rangle$")
    axes[2].set_title(r"$m_y=M_{2,y}/M_{2,y}(0)$")
    for ax in axes:
        ax.set_xlabel(r"$|\epsilon_y|$")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
    save_or_show(fig, outpath)


# =============================================================================
# 输出工具
# =============================================================================

def write_predictions(results: Dict[str, CaseResult], outdir: Path) -> None:
    """输出每个 case 的 processed_state，并汇总 loading predictions。"""
    all_rows = []
    loading_rows = []
    for case in CASE_ORDER:
        res = results[case]
        df = res.merged.copy()
        df.insert(0, "case", case)
        df.insert(1, "legacy_case", res.legacy)
        df.to_csv(outdir / f"processed_state_{case}.csv", index=False)
        df.to_csv(outdir / f"processed_state_{res.legacy}.csv", index=False)  # 旧名兼容
        res.sections.to_csv(outdir / f"processed_sections_{case}.csv", index=False)
        all_rows.append(df)
        loading_rows.append(df.loc[df["is_loading"]].copy())

    pd.concat(all_rows, ignore_index=True).to_csv(outdir / "fit_predictions_all.csv", index=False)
    pd.concat(loading_rows, ignore_index=True).to_csv(outdir / "fit_predictions_loading.csv", index=False)


def write_summary_text(
    results: Dict[str, CaseResult],
    coeff_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    outpath: Path,
) -> None:
    """输出文本总结，便于快速复制到 SI 草稿。"""
    lines: List[str] = []
    lines.append("BaTiO3 Ref. / 90° DW Figure 4 + SI model comparison summary")
    lines.append("=" * 78)
    lines.append("")
    lines.append("Main interpretation:")
    lines.append("  The reduced relation is treated as a semi-quantitative field-to-stress")
    lines.append("  consistency test, not as a universal constitutive model.")
    lines.append("  Fitted coefficients are effective projection coefficients, not intrinsic")
    lines.append("  elastic or electrostrictive constants.")
    lines.append("")

    for case in CASE_ORDER:
        s = results[case].summary
        lines.append(f"[{CASE_LABEL[case]} / legacy={results[case].legacy}]")
        lines.append(f"  n_merged        : {int(s['n_merged'])}")
        lines.append(f"  n_loading       : {int(s['n_loading'])}")
        lines.append(f"  strain_max      : {s['strain_max']:.8f}")
        lines.append(f"  stress_max_load : {s['stress_max_loading']:.8f}")
        lines.append(f"  m_y_end_loading : {s['m_y_end_loading']:.8f}")
        lines.append("")

    lines.append("Model metrics on loading segment:")
    lines.append(metrics_df.to_string(index=False))
    lines.append("")
    lines.append("Fit coefficients:")
    lines.append(coeff_df.to_string(index=False))
    lines.append("")
    lines.append("Suggested SI wording:")
    lines.append("  The common-coefficient reduced relation captures the dominant stress trend")
    lines.append("  in both domain structures, while structure-dependent coefficients refine")
    lines.append("  the quantitative agreement. The coefficients are therefore interpreted as")
    lines.append("  effective projection measures rather than intrinsic material constants.")

    outpath.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 主程序
# =============================================================================

def main() -> int:
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path(".").resolve()

    root = script_dir
    outdir = script_dir / f"analysis_{DATA_FOLDER_NAME}_{OUTPUT_SUFFIX}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Discovering input files...")
    paths = discover_paths(root)

    print("[2/5] Reading and merging data...")
    results = {case: build_case(paths[case]) for case in CASE_ORDER}

    print("[3/5] Fitting Figure 4 / SI model comparisons...")
    coeff_df, metrics_df = fit_all_models(results)

    print("[4/5] Writing CSV outputs...")
    write_predictions(results, outdir)
    coeff_df.to_csv(outdir / "fit_coefficients_all.csv", index=False)
    metrics_df.to_csv(outdir / "fit_metrics_all.csv", index=False)
    summary_df = pd.DataFrame([{ "case": case, "legacy_case": results[case].legacy, **results[case].summary } for case in CASE_ORDER])
    summary_df.to_csv(outdir / "summary_cases.csv", index=False)
    write_summary_text(results, coeff_df, metrics_df, outdir / "summary_figure4_SI.txt")

    print("[5/5] Plotting figures...")
    plot_main_figure4_candidate(results, outdir / "figure4_main_candidate.png")
    plot_si_model_comparison(results, outdir / "figureS_model_comparison.png")
    plot_si_residuals(results, outdir / "figureS_residuals.png")
    plot_si_constrained_and_cross(results, outdir / "figureS_constrained_cross_prediction.png")
    plot_si_polarization_metrics(results, outdir / "figureS_polarization_metrics.png")

    print("\n[成功] Figure 4 / SI 分析完成")
    print(f"输出目录: {outdir}")
    print("\n核心输出:")
    print("  - fit_metrics_all.csv")
    print("  - fit_coefficients_all.csv")
    print("  - fit_predictions_loading.csv")
    print("  - figure4_main_candidate.png")
    print("  - figureS_model_comparison.png")
    print("  - figureS_residuals.png")
    print("  - figureS_constrained_cross_prediction.png")
    print("  - summary_figure4_SI.txt")

    pooled_metrics = metrics_df[metrics_df["case"] == "pooled"].copy()
    if len(pooled_metrics):
        print("\nPooled model metrics:")
        cols = ["model", "rmse", "mae", "r2", "nrmse_range"]
        print(pooled_metrics[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    # 不使用 raise SystemExit，以免 Spyder 中图像窗口被提前关闭。
    main()
