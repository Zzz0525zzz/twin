#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaTiO3 孪晶(Twin) / 单畴(Single) 在 y 轴单轴压缩下的极化演化与力学分析
最小可靠机制分析 (单脚本独立运行版本)

本脚本的核心物理逻辑
--------------------------
主要状态变量 (Primary state variable):
    M2y = <Py^2> = Py_var + Py_mean^2

本脚本直接读取文本格式的原始/半原始 MD 输出数据:
    - dumps/*_ss_y_inst_2pct_cont.txt
    - tecplot/Pol_*/Pol_Pro_stats_overall.dat
    - tecplot/Pol_*/Pol_Pro_stats_sections_*.dat
并生成最多三张核心图表。

本脚本使用的拟合闭合关系是有意设计的基于“可观测量”的极简模型:
    sigma(e) ≈ C_lin * e - K_M * (1 - M2y/M2y0)
这是一个受朗道-金兹堡-德文希尔(GLD)理论/电致伸缩效应启发的唯象本构方程。
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EPS = 1.0e-15

# =============================================================================
# 用户配置区 (User Configuration)
# =============================================================================
# 定义存放原始数据文件的文件夹名称 (默认: "compute2")
# 如果后续更换了数据文件夹(如 "compute3")，直接在这里修改即可。
DATA_FOLDER_NAME = "compute"


def clean_var_name(name: str) -> str:
    """清理并标准化列名中的特殊字符"""
    name = name.strip()
    name = name.replace("|Px|", "absPx")
    name = name.replace("|Py|", "absPy")
    name = name.replace("|Pz|", "absPz")
    name = name.replace("|P|", "absP")
    name = name.replace(" ", "")
    name = name.replace("/", "_per_")
    name = name.replace("%", "pct")
    name = name.replace("-", "_")
    return name


@dataclass
class CasePaths:
    """存储单个模拟案例(单畴或孪晶)所需的所有文件路径"""
    mech: Path
    overall: Path
    sections: Path


@dataclass
class CaseResult:
    """存储单个模拟案例的完整分析结果"""
    case: str
    merged: pd.DataFrame
    sections: pd.DataFrame
    summary: Dict[str, float]
    fit_table: pd.DataFrame


# -----------------------------------------------------------------------------
# 读取器 (Readers)
# -----------------------------------------------------------------------------

def read_mech_txt(path: Path) -> pd.DataFrame:
    """读取宏观力学 dump 数据文件"""
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
    """读取极化统计数据文件 (智能兼容带表头的纯文本及Tecplot格式)"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
    header = []
    data_lines = []
    
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
            
        # 识别 Tecplot 的 VARIABLES 行
        var_match = re.match(r'^VARIABLES\s*=\s*(.*)', s, re.IGNORECASE)
        if var_match:
            var_str = var_match.group(1)
            # 提取带引号的列名或空格分隔的列名
            if '"' in var_str:
                header = re.findall(r'"([^"]+)"', var_str)
            else:
                header = var_str.split()
            continue
            
        # 忽略 Tecplot 的 ZONE 行
        if s.upper().startswith("ZONE"):
            continue
            
        # 如果没有 VARIABLES 行，把第一行有效数据作为表头 (针对纯文本表格)
        if not header and not data_lines:
            header = [x.strip('"') for x in s.split()]
            continue
            
        # 追加数据行
        data_lines.append(s.split())
        
    if not header:
        raise ValueError(f"{path} 文件为空或无法识别表头")
        
    # 确保数据列数对齐 (针对文件末尾可能有不完整截断的情况)
    clean_data = []
    for row in data_lines:
        if len(row) >= len(header):
            clean_data.append(row[:len(header)])
        else:
            clean_data.append(row + [np.nan] * (len(header) - len(row)))
            
    df = pd.DataFrame(clean_data, columns=header)
    
    # 将所有数据列强制转为浮点型数值
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 清理并标准化列名 (将 |Py|_mean 转换为 absPy_mean 等)
    df.columns = [clean_var_name(col) for col in df.columns]
    
    time_col = "Time" if "Time" in df.columns else ("TimeStep" if "TimeStep" in df.columns else None)
    if time_col is None:
        raise ValueError(f"{path} 缺少 Time 或 TimeStep 列。解析出的表头为: {list(df.columns)}")
        
    df = (
        df.dropna(subset=[time_col])
          .drop_duplicates(subset=[time_col], keep="first")
          .sort_values(time_col)
          .reset_index(drop=True)
    )
    
    # 统一将时间列命名为 TimeStep，方便后续横向对其合并
    if time_col != "TimeStep":
        df = df.rename(columns={time_col: "TimeStep"})
        
    return df


# -----------------------------------------------------------------------------
# 简易信号处理辅助函数 (无 scipy 依赖版)
# -----------------------------------------------------------------------------

def odd_window(n: int, wanted: int) -> int:
    """确保滑动窗口大小为奇数，且不超过数据长度"""
    if n < 3:
        return 1
    w = max(3, min(wanted, n if n % 2 == 1 else n - 1))
    if w % 2 == 0:
        w -= 1
    return max(3, w)


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """一维数组的平滑滑动平均"""
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    w = odd_window(len(x), window)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(xp, kernel, mode="valid")


def running_rms(x: np.ndarray, window: int) -> np.ndarray:
    """计算滑动均方根 (用于提取应力锯齿雪崩的强度)"""
    return np.sqrt(np.maximum(moving_average(np.asarray(x, dtype=float) ** 2, window), 0.0))


def smooth_gradient(y: np.ndarray, x: np.ndarray, window: int = 9) -> np.ndarray:
    """计算平滑后的梯度 (用于提取切线模量 E_t)"""
    y = moving_average(np.asarray(y, dtype=float), window)
    x = moving_average(np.asarray(x, dtype=float), window)
    if len(y) < 3:
        return np.full_like(y, np.nan)
    dy = np.gradient(y)
    dx = np.gradient(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = dy / dx
    g[~np.isfinite(g)] = np.nan
    return g


def cumulative_abs_onset(x: np.ndarray, y: np.ndarray, frac: float = 0.10) -> float:
    """提取累积绝对变差达到阈值时的临界应变 (Onset point)"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return float("nan")
    dy = np.diff(y)
    cum = np.concatenate([[0.0], np.cumsum(np.abs(dy))])
    total = float(cum[-1])
    if total <= 0.0:
        return float("nan")
    idx = int(np.searchsorted(cum, frac * total))
    idx = min(idx, len(x) - 1)
    return float(x[idx])


def safe_r2(y: np.ndarray, yhat: np.ndarray) -> float:
    """安全计算决定系数 R^2"""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / denom)


def fit_no_intercept(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """无截距的多元线性最小二乘法拟合"""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ coef
    return coef, yhat, safe_r2(y, yhat)


# -----------------------------------------------------------------------------
# 数据构建与核心逻辑分析
# -----------------------------------------------------------------------------

def build_case(case: str, paths: CasePaths) -> CaseResult:
    """构建单个模拟案例的数据帧，提取微观特征并完成本构模型的拟合"""
    mech = read_mech_txt(paths.mech)
    overall = read_stats_table(paths.overall)
    sections = read_stats_table(paths.sections)

    required_overall = ["Py_mean", "Py_var", "absPy_mean"]
    missing_overall = [c for c in required_overall if c not in overall.columns]
    if missing_overall:
        raise ValueError(f"{paths.overall} 缺少以下列: {missing_overall}")

    # 按时间步对齐力学态与极化态
    merged = mech.merge(overall, on="TimeStep", how="inner", validate="one_to_one")
    if len(merged) < 20:
        raise ValueError(f"{case} 案例的对齐数据行数过少: {len(merged)}")

    # 宏观力学：统一转换为绝对正值（压缩量化）
    merged["strain_y"] = merged["s2"].abs()
    merged["stress_y"] = merged["p2"].abs()

    # 构建核心状态变量
    merged["M2y"] = merged["Py_var"] + merged["Py_mean"] ** 2
    merged["M2y_sqrt"] = np.sqrt(np.maximum(merged["M2y"], 0.0))

    # 归一化初始状态（获取“保留率/退化率”）
    M2y0 = float(merged["M2y"].iloc[0])
    merged["m_y"] = merged["M2y"] / max(M2y0, EPS)
    
    # 极化退化进度 (1 - 保留率)
    merged["progress_M2y"] = 1.0 - merged["m_y"]

    # 非线性力学诊断量的提取
    merged["stress_smooth"] = moving_average(merged["stress_y"].to_numpy(), 11)
    merged["delta_sigma_serr"] = merged["stress_y"] - merged["stress_smooth"]
    merged["S_rms"] = running_rms(merged["delta_sigma_serr"].to_numpy(), 11)  # 锯齿雪崩的RMS

    # 计算截面级别上的空间异质性（Heterogeneity）
    for sec in ("xy", "xz", "yz"):
        req = [f"Py_mean_{sec}", f"Py_var_{sec}", f"absPy_mean_{sec}"]
        missing = [c for c in req if c not in sections.columns]
        if missing:
            raise ValueError(f"{paths.sections} 缺少截面列 {missing}")
        sections[f"M2y_{sec}"] = sections[f"Py_var_{sec}"] + sections[f"Py_mean_{sec}"] ** 2

    section_m2_cols = [f"M2y_{sec}" for sec in ("xy", "xz", "yz")]
    sections["H_M2_sections"] = sections[section_m2_cols].std(axis=1) / np.maximum(sections[section_m2_cols].mean(axis=1), EPS)

    merged = merged.merge(
        sections[["TimeStep", "H_M2_sections"]],
        on="TimeStep",
        how="left",
        validate="one_to_one",
    )

    # =============== 分割加载段与计算依赖加载段的参量 ===============
    # 自动识别加载段与卸载段的转折点 (最大应变处)
    idx_max_strain = merged["strain_y"].idxmax()
    # 提取纯加载段数据 (用于精准拟合本构参数和提取Onset)
    merged_loading = merged.iloc[:idx_max_strain+1].copy()

    # 切线模量仅在加载段计算具有明确的"屈服"诊断意义，避免卸载折返导致图表混乱
    merged_loading["E_t"] = smooth_gradient(
        merged_loading["stress_y"].to_numpy(),
        merged_loading["strain_y"].to_numpy(),
        window=9,
    )
    # 将 E_t 塞进 merged 表供后续作图调取
    merged["E_t"] = np.nan
    merged.loc[:idx_max_strain, "E_t"] = merged_loading["E_t"]

    # =============== 物理约束的边界拟合法 (Bounded Fitting) ===============
    # 允许 C_lin 在用户指定的 [170, 220] GPa 范围内浮动寻找全局最优解，
    # 同时计算对应的极化松弛参数 K_M，避免无约束最小二乘法导致的符号反转和共线性发散。
    
    e_loading = merged_loading["strain_y"].to_numpy()
    y_loading = merged_loading["stress_y"].to_numpy()
    prog_M2_loading = merged_loading["progress_M2y"].to_numpy()

    C_candidates = np.linspace(170.0, 220.0, 501)
    best_sse = float('inf')
    best_C = 170.0
    best_K = 0.0

    denom = np.sum(prog_M2_loading ** 2)
    denom = denom if denom > EPS else EPS

    for C_test in C_candidates:
        # 理论纯弹性应力 与 真实应力 的差值 (即极化耗散导致的“应力亏损”)
        stress_deficit = C_test * e_loading - y_loading
        
        # 在给定 C_test 下，使残差最小的最优 K_M
        K_test = np.sum(prog_M2_loading * stress_deficit) / denom
        
        # 计算当前组合的误差 SSE
        y_pred = C_test * e_loading - K_test * prog_M2_loading
        sse = np.sum((y_loading - y_pred) ** 2)
        
        if sse < best_sse:
            best_sse = sse
            best_C = C_test
            best_K = K_test

    C_lin = best_C
    K_M = best_K

    # 将物理参数代入全过程(包含卸载段)进行验证
    # 预测公式: sigma = C_lin * e - K_M * (1 - m_y)
    merged["stress_pred_M2"] = C_lin * merged["strain_y"] - K_M * merged["progress_M2y"]
    merged["resid_M2"] = merged["stress_y"] - merged["stress_pred_M2"]

    # 计算加载段拟合优度 R2
    yhat_loading = C_lin * e_loading - K_M * prog_M2_loading
    r2_M2 = safe_r2(y_loading, yhat_loading)

    # 将拟合参数打包输出为表格
    fit_table = pd.DataFrame(
        [
            {
                "case": case,
                "model": "stress ~ C*strain - K*(1-m_y)",
                "coef_1": float(C_lin),
                "coef_2": float(K_M),
                "coef_3": float("nan"),
                "r2": float(r2_M2),
            }
        ]
    )

    # 提取全局概览指标 (注意：Onset的提取现在只基于加载段，防止卸载波动干扰)
    summary: Dict[str, float] = {
        "n_mech": float(len(mech)),
        "n_overall": float(len(overall)),
        "n_sections": float(len(sections)),
        "n_merged": float(len(merged)),
        "strain_start": float(merged["strain_y"].iloc[0]),
        "strain_max": float(merged["strain_y"].max()),
        "stress_start": float(merged["stress_y"].iloc[0]),
        "M2y_start": float(merged["M2y"].iloc[0]),
        "corr_stress_vs_M2": float(np.corrcoef(merged_loading["stress_y"], merged_loading["M2y"])[0, 1]),
        "onset10_M2y": cumulative_abs_onset(merged_loading["strain_y"], merged_loading["M2y"], frac=0.10),
        "onset10_S_rms": cumulative_abs_onset(merged_loading["strain_y"], merged_loading["S_rms"], frac=0.10),
        "H_M2_start": float(merged["H_M2_sections"].iloc[0]),
        "H_M2_end": float(merged["H_M2_sections"].iloc[-1]),
        "fit_r2_M2": float(r2_M2),
        "fit_C_M2": float(C_lin),
        "fit_K_M2": float(K_M),
    }

    return CaseResult(
        case=case,
        merged=merged,
        sections=sections,
        summary=summary,
        fit_table=fit_table,
    )


# -----------------------------------------------------------------------------
# 图表绘制 (Plotting)
# (注: 为兼容未配置中文字体的运行环境，图表坐标轴保持英文标签)
# -----------------------------------------------------------------------------

def make_figure_1(results: Dict[str, CaseResult], outpath: Path) -> None:
    """生成 Figure 1: 宏观力学响应与微观极化演化的轨迹对比"""
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True)

    ax = axes[0]
    for case in ("single", "twin"):
        df = results[case].merged
        ax.plot(df["strain_y"], df["stress_y"], label=f"{case}: stress")
    ax.set_ylabel("|sigma_y|")
    ax.set_title("Figure 1. Mechanics and state variables")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    for case in ("single", "twin"):
        df = results[case].merged
        ax.plot(df["strain_y"], df["m_y"], label=f"{case}: m_y = M2y/M2y0")
    ax.set_xlabel("|epsilon_y|")
    ax.set_ylabel("normalized state m_y")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.show()


def make_figure_2(results: Dict[str, CaseResult], outpath: Path) -> None:
    """生成 Figure 2: 可观测量闭合模型的预测准确率与残差对比"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), sharex="col")

    for col, case in enumerate(("single", "twin")):
        df = results[case].merged
        s = results[case].summary
        
        ax = axes[0, col]
        ax.plot(df["strain_y"], df["stress_y"], label="measured")
        ax.plot(df["strain_y"], df["stress_pred_M2"], label="fit with m_y")
        ax.set_title(f"Figure 2. Observable closure ({case})")
        ax.set_ylabel("|sigma_y|")
        ax.grid(True, alpha=0.3)
        # 将图例显式固定在左上角
        ax.legend(loc="upper left", frameon=False)
        
        # 在图上添加拟合参数文本
        c_lin_val = s['fit_C_M2']
        k_m_val = s['fit_K_M2']
        ax.text(0.05, 0.70, f"$C_{{lin}}$ = {c_lin_val:.1f}\n$K_M$ = {k_m_val:.1f}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax = axes[1, col]
        ax.plot(df["strain_y"], df["resid_M2"], label="residual: m_y fit")
        ax.set_xlabel("|epsilon_y|")
        ax.set_ylabel("residual")
        ax.grid(True, alpha=0.3)
        # 统一规范残差图的图例位置
        ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.show()


def make_figure_3(results: Dict[str, CaseResult], outpath: Path) -> None:
    """生成 Figure 3: 切线模量演化"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))

    for case in ("single", "twin"):
        df = results[case].merged
        ax.plot(df["strain_y"], df["E_t"], label=f"{case}: E_t")
    ax.set_title("Figure 3. Tangent modulus (diagnostic only)")
    ax.set_xlabel("|epsilon_y|")
    ax.set_ylabel("E_t = d|sigma|/d|epsilon|")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.show()


# -----------------------------------------------------------------------------
# 实用工具与输出
# -----------------------------------------------------------------------------

def discover_paths(root: Path, folder_name: str = DATA_FOLDER_NAME) -> Dict[str, CasePaths]:
    """遍历搜索默认目录结构中的所需数据文件"""
    compute_dir = root / folder_name
    paths = {
        "single": CasePaths(
            mech=compute_dir / "single_ss_y_inst_2pct_cont.txt",
            overall=compute_dir / "Pol_Pro_stats_overall_single.dat",
            sections=compute_dir / "Pol_Pro_stats_sections_single.dat",
        ),
        "twin": CasePaths(
            mech=compute_dir / "twin_ss_y_inst_2pct_cont.txt",
            overall=compute_dir / "Pol_Pro_stats_overall_twin.dat",
            sections=compute_dir / "Pol_Pro_stats_sections_twin.dat",
        ),
    }
    missing: List[str] = []
    for case, cp in paths.items():
        for label in ("mech", "overall", "sections"):
            p = getattr(cp, label)
            if not p.exists():
                missing.append(f"{case}:{label}:{p}")
    if missing:
        raise FileNotFoundError("未找到必须的数据文件:\n  - " + "\n  - ".join(missing))
    return paths


def write_summary_text(results: Dict[str, CaseResult], outpath: Path) -> None:
    """输出文本格式的简要摘要报告"""
    lines: List[str] = []
    lines.append("BaTiO3 孪晶 / 单畴极简演化机制总结报告")
    lines.append("=")
    lines.append("")
    lines.append("当前分析采用的主状态变量: M2y = <Py^2>")
    lines.append("测试的可观测量闭合公式: sigma ≈ C*strain - K*(1 - M2y/M2y0)")
    lines.append("注: C_lin 在 170~220 GPa 范围进行边界搜索提取，全程统一。")
    lines.append("")
    for case in ("single", "twin"):
        s = results[case].summary
        lines.append(f"[{case} 模拟案例]")
        lines.append(f"  成功对齐并合并的数据行数: {int(s['n_merged'])}")
        lines.append(f"  加载段最大应变 (strain_max): {s['strain_max']:.6f}")
        lines.append(f"  等效本底弹性模量 (C_lin): {s['fit_C_M2']:.1f} GPa")
        lines.append(f"  极化背应力系数 (K_M): {s['fit_K_M2']:.1f} GPa")
        lines.append(f"  加载段拟合决定系数 (R2):    {s['fit_r2_M2']:.6f}")
        lines.append(f"  加载段 10% 临界启动应变 (M2y / S_rms): {s['onset10_M2y']:.6f}, {s['onset10_S_rms']:.6f}")
        lines.append(f"  截面 M2 异质性指数变化:  {s['H_M2_start']:.6f} -> {s['H_M2_end']:.6f}")
        lines.append("")
    outpath.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# 主函数入口 (Main)
# -----------------------------------------------------------------------------

def main() -> int:
    """脚本执行入口点"""
    # 自动获取当前 Python 脚本所在的物理文件夹路径
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        # 兼容在极少数纯交互式终端(未保存成文件)中运行的情况
        script_dir = Path(".").resolve()

    # 根目录设定为脚本所在文件夹，输出文件夹自动跟随输入文件夹名称联动
    root = script_dir
    outdir = script_dir / f"analysis_{DATA_FOLDER_NAME}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 扫描文件路径并构建数据
    paths = discover_paths(root)
    results = {
        case: build_case(case, cp)
        for case, cp in paths.items()
    }

    # 2. 导出所有核心数据为 CSV 表格 (这一步产出了用于画图和提取公式常数的底层数据)
    for case, res in results.items():
        res.merged.to_csv(outdir / f"processed_state_{case}.csv", index=False)
        res.sections.to_csv(outdir / f"processed_sections_{case}.csv", index=False)
        res.fit_table.to_csv(outdir / f"fit_table_{case}.csv", index=False)

    summary_df = pd.DataFrame(
        [{"case": case, **res.summary} for case, res in results.items()]
    )
    summary_df.to_csv(outdir / "summary_mechanism.csv", index=False)
    write_summary_text(results, outdir / "summary_mechanism.txt")

    # 3. 生成并保存三组学术核心图表
    make_figure_1(results, outdir / "figure1_mechanics_and_state.png")
    make_figure_2(results, outdir / "figure2_observable_closure.png")
    make_figure_3(results, outdir / "figure3_diagnostics_onsets.png")

    # 4. 终端提示输出
    print(f"[成功] 所有处理已完成，结果输出至: {outdir.name}")
    print("各机制诊断速览：")
    for case in ("single", "twin"):
        s = results[case].summary
        print(
            f" - {case.upper()} 案例: C_lin={s['fit_C_M2']:.1f} | K_M={s['fit_K_M2']:.1f} | R2={s['fit_r2_M2']:.6f}"
        )
    return 0


if __name__ == "__main__":
    # 移除了 raise SystemExit() 强制退出机制，防止 Spyder 来不及渲染图像
    main()