#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""polarization_atomic_twin_pro.py

静态投影网格 + 动态 5 帧邻居重构计算 (The Ultimate Hybrid - Refactored Edition)
-------------------------------------------------------------------------
核心机制：
1. 【网格锁死】：首帧探测真实晶轴并展开，输出给 Tecplot 的 I x J 网格永远锁死，杜绝锯齿与飞线。
2. 【计算动态】：每隔 5 帧重构 KDTree 刷新邻居，利用张量并行计算，获取实时演化的极化矢量。
3. 【自动寻层】：刀刃自动吸附物理原子层，彻底告别空截面。
4. 【专业架构】：高度模块化解耦，集成 Argparse 命令行控制，具备工程级稳定性与可读性。
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:
    print("\n[致命错误] 本程序需要 scipy 库进行极速空间寻址。")
    print("请在终端运行: pip install scipy --user\n")
    import sys
    sys.exit(1)


# ==============================================================================
# 模块 1：全局常数与数据结构 (Constants & Data Structures)
# ==============================================================================

DEFAULT_DUMP_PATTERN = "single_YCompressCont.*.dump"
DEFAULT_SECTION_THICKNESS = 1.5
DEFAULT_MAX_KNN_DIST = 8.0
DEFAULT_REBUILD_FREQ = 5
DEFAULT_OUT_DIR_BASE = "Pol_out_Pro"
OUT_PREFIX = "Pol_Pro"
DEFAULT_SHOW_PROGRESS = True
PROGRESS_BAR_LEN = 30

CHARGE_BASE: Dict[int, float] = {
    1: 4.784, 2: -2.948, 3: 4.488, 4: -1.614, 5: 1.039, 6: -2.609,
}
DEFAULT_CHARGE: Dict[int, float] = {t: q * 1.0 for t, q in CHARGE_BASE.items()}

DEFAULT_SHARE: Dict[int, float] = {
    1: 1.0 / 8.0, 2: 1.0 / 8.0, 3: 1.0, 4: 1.0, 5: 1.0 / 2.0, 6: 1.0 / 2.0,
}

KNN_EXPECTED: Dict[int, int] = {
    1: 8, 2: 8, 4: 1, 5: 6, 6: 6,
}

DEFAULT_CENTER_TYPE = 3
ELEM_CHARGE = 1.602176634e-19
ANG2M = 1e-10
V_CUBE = (4.0 ** 3) * (ANG2M ** 3)
CONV_UC_CM2 = 1602.176634  # (μC/cm^2) per (e·Å / Å^3)

@dataclass
class Config:
    pattern: str
    out_prefix: str
    section_thickness: float
    max_knn_dist: float
    rebuild_freq: int

@dataclass
class RefSystem:
    boxlen: np.ndarray       
    origin: np.ndarray       
    ids: np.ndarray          
    types: np.ndarray        
    pos: np.ndarray          
    index_by_id: np.ndarray  


# ==============================================================================
# 模块 2：通用几何与数学工具 (Math & Geometry Utilities)
# ==============================================================================

def _fmt_hms(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h:d}:{m:02d}:{sec:02d}" if h > 0 else f"{m:02d}:{sec:02d}"

def natural_sort(files: Sequence[str]) -> List[str]:
    return sorted(files, key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)])

def mic_delta(delta: np.ndarray, boxlen: np.ndarray) -> np.ndarray:
    return delta - boxlen * np.rint(delta / boxlen)

def safe_stats(x: np.ndarray) -> Tuple[float, float, float, float]:
    if x.size == 0: return (np.nan, np.nan, np.nan, np.nan)
    if x.size == 1: return (float(x.min()), float(x.max()), 0.0, 0.0)
    return (float(x.min()), float(x.max()), float(x.std(ddof=1)), float(x.var(ddof=1)))


# ==============================================================================
# 模块 3：数据解析引擎 (Data Parsing Engine)
# ==============================================================================

def choose_coord_columns(hdr: List[str]) -> Tuple[str, str, str]:
    for m in [("xu", "yu", "zu"), ("x", "y", "z"), ("xs", "ys", "zs")]:
        if all(c in hdr for c in m): return m
    raise ValueError("找不到坐标列 (xu/x/xs)，请检查 dump 文件格式。")

def parse_dump_as_ref(path: str) -> RefSystem:
    """提取第一帧（或 atomic.data）作为建立绝对拓扑网格的参考系"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in range(3): f.readline()
        n = int(f.readline().strip())
        f.readline()
        xlo, xhi, *_ = map(float, f.readline().split())
        ylo, yhi, *_ = map(float, f.readline().split())
        zlo, zhi, *_ = map(float, f.readline().split())
        boxlen = np.array([xhi - xlo, yhi - ylo, zhi - zlo], dtype=float)
        origin = np.array([xlo, ylo, zlo], dtype=float)

        hdr = f.readline().split()[2:]
        cx, cy, cz = choose_coord_columns(hdr)
        idx_id, idx_type = hdr.index("id"), hdr.index("type")
        idx_x, idx_y, idx_z = hdr.index(cx), hdr.index(cy), hdr.index(cz)

        ids, types, pos = np.empty(n, dtype=int), np.empty(n, dtype=int), np.empty((n, 3), dtype=float)
        for i in range(n):
            sp = f.readline().split()
            ids[i], types[i] = int(sp[idx_id]), int(sp[idx_type])
            pos[i, 0], pos[i, 1], pos[i, 2] = float(sp[idx_x]), float(sp[idx_y]), float(sp[idx_z])

    if cx == "xs":
        pos[:, 0], pos[:, 1], pos[:, 2] = xlo + pos[:, 0] * boxlen[0], ylo + pos[:, 1] * boxlen[1], zlo + pos[:, 2] * boxlen[2]

    index_by_id = np.full(int(ids.max()) + 1, -1, dtype=int)
    index_by_id[ids] = np.arange(n, dtype=int)
    return RefSystem(boxlen=boxlen, origin=origin, ids=ids, types=types, pos=pos, index_by_id=index_by_id)

def parse_dump_coords(path: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, Tuple[float, float, float, float, float, float]]:
    """极速提取演化帧中的坐标用于实时极化计算"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline(); step = int(f.readline().strip())
        f.readline(); n = int(f.readline().strip())
        f.readline()
        xlo, xhi, *_ = map(float, f.readline().split())
        ylo, yhi, *_ = map(float, f.readline().split())
        zlo, zhi, *_ = map(float, f.readline().split())
        bounds = (xlo, xhi, ylo, yhi, zlo, zhi)
        hdr = f.readline().split()[2:]

        cx, cy, cz = choose_coord_columns(hdr)
        idx_id, idx_type = hdr.index("id"), hdr.index("type")
        idx_x, idx_y, idx_z = hdr.index(cx), hdr.index(cy), hdr.index(cz)

        ids, types, r = np.empty(n, dtype=int), np.empty(n, dtype=int), np.empty((n, 3), dtype=float)
        for i in range(n):
            sp = f.readline().split()
            ids[i], types[i] = int(sp[idx_id]), int(sp[idx_type])
            r[i, 0], r[i, 1], r[i, 2] = float(sp[idx_x]), float(sp[idx_y]), float(sp[idx_z])

    if cx == "xs":
        Lx, Ly, Lz = xhi - xlo, yhi - ylo, zhi - zlo
        r[:, 0], r[:, 1], r[:, 2] = xlo + r[:, 0] * Lx, ylo + r[:, 1] * Ly, zlo + r[:, 2] * Lz

    return step, ids, types, r, bounds


# ==============================================================================
# 模块 4：拓扑锁定与网格重组引擎 (Topology & Grid Engine)
# ==============================================================================

def detect_lattice_axes(coords: np.ndarray, is_2d: bool) -> List[np.ndarray]:
    """智能探测真实晶轴（免疫孪晶或 45° 旋转）"""
    if len(coords) < 10:
        return [np.array([1.0, 0.0]), np.array([0.0, 1.0])] if is_2d else [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    
    centroid = np.mean(coords, axis=0)
    center_idx = np.argmin(np.sum((coords - centroid)**2, axis=1))
    
    tree = cKDTree(coords)
    dists, idxs = tree.query(coords[center_idx], k=min(30, len(coords)))
    valid = (dists > 3.0) & (dists < 5.0) 
    neighbors = coords[idxs[valid]]
    
    axes = []
    if len(neighbors) > 0:
        vecs = neighbors - coords[center_idx]
        axes.append(vecs[0] / np.linalg.norm(vecs[0]))
        for v in vecs[1:]:
            v_norm = v / np.linalg.norm(v)
            if all(np.abs(np.dot(v_norm, ax)) < 0.5 for ax in axes): 
                axes.append(v_norm)
                if (is_2d and len(axes) == 2) or (not is_2d and len(axes) == 3): break
                    
    ortho_axes = []
    if len(axes) > 0:
        ortho_axes.append(axes[0])
        for i in range(1, len(axes)):
            v = axes[i]
            for ax in ortho_axes: v = v - np.dot(v, ax) * ax
            norm = np.linalg.norm(v)
            if norm > 1e-3: ortho_axes.append(v / norm)
                
    if is_2d:
        if len(ortho_axes) == 0: ortho_axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        elif len(ortho_axes) == 1: ortho_axes.append(np.array([-ortho_axes[0][1], ortho_axes[0][0]]))
    else:
        if len(ortho_axes) == 0: ortho_axes = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        elif len(ortho_axes) == 1:
            v1 = ortho_axes[0]
            temp = np.array([0.0, 1.0, 0.0]) if np.abs(v1[0]) > 0.5 else np.array([1.0, 0.0, 0.0])
            v2 = temp - np.dot(temp, v1) * v1
            ortho_axes.extend([v2 / np.linalg.norm(v2), np.cross(v1, v2 / np.linalg.norm(v2))])
        elif len(ortho_axes) == 2:
            ortho_axes.append(np.cross(ortho_axes[0], ortho_axes[1]))
            
    return [ax / np.linalg.norm(ax) for ax in ortho_axes]

def assign_layers(coords_1d: np.ndarray, tol: float = 2.0) -> np.ndarray:
    if coords_1d.size == 0: return np.array([], dtype=int)
    sort_idx = np.argsort(coords_1d)
    sorted_vals = coords_1d[sort_idx]
    
    diffs = np.diff(sorted_vals) > tol
    labels_sorted = np.insert(np.cumsum(diffs), 0, 0)
    
    labels = np.zeros_like(coords_1d, dtype=int)
    labels[sort_idx] = labels_sorted
    return labels

def create_unwrapped_grid(ref: RefSystem, ti_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int, int, bool, np.ndarray]:
    """获取全相连的 3D 绝对连续正交网格，并将质心死死钉在盒子中央"""
    r_ti = ref.pos[ref.index_by_id[ti_ids]]
    boxlen = ref.boxlen
    n = len(r_ti)
    
    r_w = ref.origin + np.mod(r_ti - ref.origin, boxlen)
    dist_threshold2 = 4.8 ** 2
    adj = [[] for _ in range(n)]
    chunk_size = 1000
    for i in range(0, n, chunk_size):
        end = min(n, i + chunk_size)
        chunk = r_w[i:end]
        d = mic_delta(chunk[:, np.newaxis, :] - r_w[np.newaxis, :, :], boxlen)
        dist2 = np.sum(d**2, axis=-1)
        for k in range(end - i):
            neighbors = np.where(dist2[k] < dist_threshold2)[0]
            adj[i+k].extend([nxt for nxt in neighbors if nxt != (i+k)])
            
    r_uw = np.copy(r_w)
    visited = np.zeros(n, dtype=bool)
    
    for i in range(n):
        if not visited[i]:
            visited[i] = True
            queue = deque([i])
            while queue:
                curr = queue.popleft()
                for nxt in adj[curr]:
                    if not visited[nxt]:
                        visited[nxt] = True
                        r_uw[nxt] = r_uw[curr] + mic_delta(r_w[nxt] - r_w[curr], boxlen)
                        queue.append(nxt)
                        
    # 绝对锚定
    box_mid = ref.origin + boxlen / 2.0
    centroid = np.mean(r_uw, axis=0)
    r_uw -= np.round((centroid - box_mid) / boxlen) * boxlen
                        
    axes = detect_lattice_axes(r_uw, is_2d=False)
    proj_u, proj_v, proj_w = np.dot(r_uw, axes[0]), np.dot(r_uw, axes[1]), np.dot(r_uw, axes[2])
    
    ix, iy, iz = assign_layers(proj_u, tol=2.0), assign_layers(proj_v, tol=2.0), assign_layers(proj_w, tol=2.0)
    sort_idx = np.lexsort((ix, iy, iz))
    
    nx, ny, nz = len(np.unique(ix)), len(np.unique(iy)), len(np.unique(iz))
    is_struct = (nx * ny * nz == n)
    
    return r_uw[sort_idx], r_w[sort_idx], nx, ny, nz, is_struct, sort_idx

def setup_section_smart(coords_sec: np.ndarray, section_name: str) -> Tuple[bool, int, int, np.ndarray]:
    """将截面坐标投影到晶轴并重整为 I x J 阵列"""
    n = len(coords_sec)
    if n == 0: return False, 1, 1, np.arange(0)

    idx1, idx2 = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[section_name]
    c2d = coords_sec[:, [idx1, idx2]]

    axes = detect_lattice_axes(c2d, is_2d=True)
    proj_c2d = np.empty_like(c2d)
    proj_c2d[:, 0] = np.dot(c2d, axes[0])
    proj_c2d[:, 1] = np.dot(c2d, axes[1])

    factors = [(i, n//i) for i in range(2, int(np.sqrt(n)) + 1) if n % i == 0]
    factors += [(J, I) for I, J in factors if I != J]
    if not factors: return False, n, 1, np.arange(n)

    best_energy, best_I, best_J, best_sort, found = float('inf'), n, 1, np.arange(n), False

    for primary_axis, secondary_axis in [(1, 0), (0, 1)]:
        for I, J in factors:
            sort1 = np.argsort(proj_c2d[:, primary_axis])
            try:
                grid_idx = sort1.reshape((J, I))
                real_grid_c = c2d[sort1].reshape((J, I, 2))
            except ValueError: continue

            for j in range(J):
                row_sort = np.argsort(proj_c2d[grid_idx[j], secondary_axis])
                grid_idx[j] = grid_idx[j, row_sort]
                real_grid_c[j] = real_grid_c[j, row_sort]

            energy = np.sum((real_grid_c[:, 1:, :] - real_grid_c[:, :-1, :])**2) + np.sum((real_grid_c[1:, :, :] - real_grid_c[:-1, :, :])**2)

            if energy < best_energy:
                best_energy, best_I, best_J, best_sort, found = energy, I, J, grid_idx.flatten(), True

    if found and (best_energy / max(1, (best_I * (best_J - 1) + best_J * (best_I - 1)))) > 50.0: found = False
    return (True, best_I, best_J, best_sort) if found else (False, n, 1, np.arange(n))

def find_actual_atomic_layer(coords_1d: np.ndarray, math_midpoint: float, tol: float = 1.0) -> float:
    if coords_1d.size == 0: return math_midpoint
    sorted_vals = np.sort(coords_1d)
    diffs = np.diff(sorted_vals) > tol
    split_indices = np.where(diffs)[0] + 1
    if len(split_indices) == 0: return float(np.mean(sorted_vals))
    layers = np.array([np.mean(arr) for arr in np.split(sorted_vals, split_indices) if len(arr) > 0])
    if len(layers) == 0: return math_midpoint
    return float(layers[np.argmin(np.abs(layers - math_midpoint))])

def section_mask(coords: np.ndarray, section_name: str, plane_value: float, thickness: float) -> np.ndarray:
    idx = {"XY": 2, "XZ": 1, "YZ": 0}[section_name]
    return np.abs(coords[:, idx] - plane_value) <= 0.5 * thickness


# ==============================================================================
# 模块 5：极化物理运算核 (Polarization Physics Engine)
# ==============================================================================

def update_knn_cache(
    cpos: np.ndarray, types: np.ndarray, r: np.ndarray, ids: np.ndarray, 
    origin: np.ndarray, boxlen: np.ndarray, max_knn_dist: float
) -> Dict[int, np.ndarray]:
    """极速重建 K-NN 邻居网络缓存"""
    cached_neighbor_ids = {}
    c_shift = np.mod(cpos - origin, boxlen)
    max_all_id = int(ids.max()) if len(ids) > 0 else 0
    
    for t, k_val in KNN_EXPECTED.items():
        t_mask = (types == t)
        t_r, t_ids = r[t_mask], ids[t_mask]
        if t_r.size > 0:
            tree = cKDTree(np.mod(t_r - origin, boxlen), boxsize=boxlen)
            _, idxs = tree.query(c_shift, k=k_val, distance_upper_bound=max_knn_dist)
            if k_val == 1: idxs = idxs.reshape(-1, 1)
            valid_query = idxs < len(t_ids)
            safe_idxs = np.where(valid_query, idxs, 0)
            ids_matrix = t_ids[safe_idxs]
            ids_matrix[~valid_query] = -1
            cached_neighbor_ids[t] = ids_matrix
        else:
            cached_neighbor_ids[t] = np.full((len(cpos), k_val), -1, dtype=int)
            
    return cached_neighbor_ids

def compute_polarization_tensors(
    n_ti: int, cpos: np.ndarray, valid_ti: np.ndarray, center_ids: np.ndarray, 
    types: np.ndarray, r: np.ndarray, ids: np.ndarray, id_to_cur_row: np.ndarray,
    origin: np.ndarray, boxlen: np.ndarray, cached_neighbor_ids: Dict[int, np.ndarray], 
    charge_lut: np.ndarray, share_lut: np.ndarray, max_knn_dist: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """纯 NumPy 张量归约计算，榨干 CPU 浮点性能"""
    wq_center = float(share_lut[DEFAULT_CENTER_TYPE] * charge_lut[DEFAULT_CENTER_TYPE])
    sum_rq_pos, sum_rq_neg = np.zeros((n_ti, 3)), np.zeros((n_ti, 3))
    total_q_pos, total_q_neg = np.full(n_ti, wq_center), np.zeros(n_ti)
    
    c_pos_expand = cpos[:, None, :]
    
    for t, k_val in KNN_EXPECTED.items():
        n_ids = cached_neighbor_ids[t]
        valid_id_mask = (n_ids != -1)
        safe_ids = np.where(valid_id_mask, n_ids, 0)
        out_of_bounds = safe_ids >= len(id_to_cur_row)
        safe_ids[out_of_bounds] = 0
        
        rows = id_to_cur_row[safe_ids]
        valid_row_mask = (rows != -1) & valid_id_mask & (~out_of_bounds)
        
        target_pos = np.zeros((n_ti, k_val, 3))
        target_pos[valid_row_mask] = r[rows[valid_row_mask]]
        
        d_vec = mic_delta(target_pos - c_pos_expand, boxlen)
        final_valid_mask = valid_row_mask & (np.linalg.norm(d_vec, axis=-1) <= max_knn_dist) & valid_ti[:, None]
        
        wq = share_lut[t] * charge_lut[t]
        if wq > 0:
            sum_rq_pos += np.sum(d_vec * final_valid_mask[:, :, None] * wq, axis=1)
            total_q_pos += np.sum(final_valid_mask * wq, axis=1)
        else:
            sum_rq_neg += np.sum(d_vec * final_valid_mask[:, :, None] * abs(wq), axis=1)
            total_q_neg += np.sum(final_valid_mask * abs(wq), axis=1)
            
    safe_pos_div, safe_neg_div = np.where(total_q_pos > 0, total_q_pos, 1.0), np.where(total_q_neg > 0, total_q_neg, 1.0)
    r_pos_local = np.where(total_q_pos[:, None] > 0, sum_rq_pos / safe_pos_div[:, None], 0.0)
    r_neg_local = np.where(total_q_neg[:, None] > 0, sum_rq_neg / safe_neg_div[:, None], 0.0)
    
    P_cells = np.zeros((n_ti, 3))
    P_cells[valid_ti] = (total_q_pos[valid_ti, None] * ELEM_CHARGE) * (r_pos_local[valid_ti] - r_neg_local[valid_ti]) * ANG2M / V_CUBE
    Pmag = np.linalg.norm(P_cells, axis=1)
    
    qdiff_absmax = float(np.max(np.abs((total_q_pos - total_q_neg)[valid_ti]))) if np.any(valid_ti) else 0.0
    return P_cells, Pmag, qdiff_absmax


# ==============================================================================
# 模块 6：输出排版系统 (Data Export System)
# ==============================================================================

def write_tecplot_zone_3d(fh, ts: int, coords: np.ndarray, P_all: np.ndarray, Pmag: np.ndarray, nx: int, ny: int, nz: int, is_struct: bool):
    n = int(coords.shape[0])
    if n == 0: return
    if is_struct:
        fh.write(f'ZONE T="ALL_TS{ts}", I={nx}, J={ny}, K={nz}, DATAPACKING=POINT, STRANDID=1, SOLUTIONTIME={float(ts):.6f}\n')
    else:
        fh.write(f'ZONE T="ALL_TS{ts}", I={n}, J=1, K=1, DATAPACKING=POINT, STRANDID=1, SOLUTIONTIME={float(ts):.6f}\n')
        
    buf = [f"{coords[i,0]:.5f} {coords[i,1]:.5f} {coords[i,2]:.5f} {P_all[i,0]:.5e} {P_all[i,1]:.5e} {P_all[i,2]:.5e} {Pmag[i]:.5e}\n" for i in range(n)]
    fh.write("".join(buf))

def write_tecplot_zone_section(fh, ts: int, section_name: str, coords_raw: np.ndarray, P_raw: np.ndarray, Pmag_raw: np.ndarray, plane_value: float, thickness: float, is_grid: bool, I: int, J: int, sort_idx: np.ndarray):
    n = int(coords_raw.shape[0])
    if n == 0: return
    coords, P_all, Pmag = coords_raw[sort_idx], P_raw[sort_idx], Pmag_raw[sort_idx]

    fh.write(f'ZONE T="{section_name}_mid_T{thickness:.3f}_C{plane_value:.3f}_TS{ts}", I={I if is_grid else n}, J={J if is_grid else 1}, K=1, DATAPACKING=POINT, STRANDID=1, SOLUTIONTIME={float(ts):.6f}\n')
    buf = []
    for i in range(n):
        x, y, z = coords[i]
        px, py, pz, pm = P_all[i,0], P_all[i,1], P_all[i,2], Pmag[i]
        if section_name == "XY": buf.append(f"{x:.5f} {y:.5f} {plane_value:.5f} {px:.5e} {py:.5e} {pz:.5e} {px:.5e} {py:.5e} {pz:.5e} {pm:.5e}\n")
        elif section_name == "XZ": buf.append(f"{x:.5f} {z:.5f} {plane_value:.5f} {px:.5e} {py:.5e} {pz:.5e} {px:.5e} {pz:.5e} {py:.5e} {pm:.5e}\n")
        elif section_name == "YZ": buf.append(f"{y:.5f} {z:.5f} {plane_value:.5f} {px:.5e} {py:.5e} {pz:.5e} {py:.5e} {pz:.5e} {px:.5e} {pm:.5e}\n")
    fh.write("".join(buf))

def section_stats_row(ts: int, Psec: np.ndarray) -> List[float]:
    if Psec.size == 0: return [float(ts), 0.0] + [np.nan]*19
    Px, Py, Pz, Pmag = Psec[:, 0], Psec[:, 1], Psec[:, 2], np.linalg.norm(Psec, axis=1)
    return [float(ts), float(Psec.shape[0]), float(Px.mean()), float(Py.mean()), float(Pz.mean()), *safe_stats(Px), *safe_stats(Py), *safe_stats(Pz), float(np.abs(Px).mean()), float(np.abs(Py).mean()), float(np.abs(Pz).mean()), float(Pmag.mean())]


# ==============================================================================
# 模块 7：主控制循环 (Main Controller)
# ==============================================================================

def process_trajectory(files: List[str], conf: Config) -> None:
    files = natural_sort(files)
    os.makedirs(DEFAULT_OUT_DIR_BASE, exist_ok=True)
    
    # 初始化静态首帧拓扑基准
    ref_path = "atomic.data" if os.path.exists("atomic.data") else files[0]
    ref = parse_dump_as_ref(ref_path)
    
    ti_ids = ref.ids[ref.types == DEFAULT_CENTER_TYPE]
    n_ti = len(ti_ids)
    
    r_uw_sorted, r_w_sorted, nx, ny, nz, is_struct, sort_idx_3d = create_unwrapped_grid(ref, ti_ids)
    ti_ids_sorted = ti_ids[sort_idx_3d]
    
    z_mid, y_mid, x_mid = ref.origin[2] + ref.boxlen[2]/2, ref.origin[1] + ref.boxlen[1]/2, ref.origin[0] + ref.boxlen[0]/2
    z_xy = find_actual_atomic_layer(r_w_sorted[:, 2], z_mid)
    y_xz = find_actual_atomic_layer(r_w_sorted[:, 1], y_mid)
    x_yz = find_actual_atomic_layer(r_w_sorted[:, 0], x_mid)
    
    m_xy = section_mask(r_w_sorted, "XY", z_xy, conf.section_thickness)
    m_xz = section_mask(r_w_sorted, "XZ", y_xz, conf.section_thickness)
    m_yz = section_mask(r_w_sorted, "YZ", x_yz, conf.section_thickness)

    is_g_xy, nx_xy, ny_xy, sort_xy = setup_section_smart(r_uw_sorted[m_xy], "XY")
    is_g_xz, nx_xz, nz_xz, sort_xz = setup_section_smart(r_uw_sorted[m_xz], "XZ")
    is_g_yz, nx_yz, ny_yz, sort_yz = setup_section_smart(r_uw_sorted[m_yz], "YZ")

    print(f"\n================ [ 晶轴投影网格重组报告 ] ================")
    print(f" 3D 全局网格: {'[展开平整]' if is_struct else '[安全散列]'} (I={nx}, J={ny}, K={nz})")
    print(f" XY 面(Z={z_xy:.2f}): {'[自适应编织]' if is_g_xy else '[散点]'} (I={nx_xy}, J={ny_xy})")
    print(f" XZ 面(Y={y_xz:.2f}): {'[自适应编织]' if is_g_xz else '[散点]'} (I={nx_xz}, J={nz_xz})")
    print(f" YZ 面(X={x_yz:.2f}): {'[自适应编织]' if is_g_yz else '[散点]'} (I={nx_yz}, J={ny_yz})")
    print(f"==========================================================\n")

    charge_lut, share_lut = np.zeros(10, dtype=float), np.zeros(10, dtype=float)
    for k, v in DEFAULT_CHARGE.items(): charge_lut[k] = v
    for k, v in DEFAULT_SHARE.items(): share_lut[k] = v

    cached_neighbor_ids = {}
    
    out_paths = {
        "ALL": os.path.join(DEFAULT_OUT_DIR_BASE, f"{conf.out_prefix}_ALL.plt"),
        "XY": os.path.join(DEFAULT_OUT_DIR_BASE, f"{conf.out_prefix}_XY_mid.plt"),
        "XZ": os.path.join(DEFAULT_OUT_DIR_BASE, f"{conf.out_prefix}_XZ_mid.plt"),
        "YZ": os.path.join(DEFAULT_OUT_DIR_BASE, f"{conf.out_prefix}_YZ_mid.plt"),
        "STAT_ALL": os.path.join(DEFAULT_OUT_DIR_BASE, f"{conf.out_prefix}_stats_overall.dat"),
        "STAT_SEC": os.path.join(DEFAULT_OUT_DIR_BASE, f"{conf.out_prefix}_stats_sections.dat")
    }

    with open(out_paths["ALL"], "w") as fall, open(out_paths["XY"], "w") as fxy, open(out_paths["XZ"], "w") as fxz, open(out_paths["YZ"], "w") as fyz, open(out_paths["STAT_ALL"], "w") as fs_all, open(out_paths["STAT_SEC"], "w") as fs_sec:

        fall.write('VARIABLES= "X" "Y" "Z" "Px" "Py" "Pz" "Pmag"\n')
        fxy.write('VARIABLES= "X" "Y" "Zsec" "Px" "Py" "Pz" "U" "V" "W" "Pmag"\n')
        fxz.write('VARIABLES= "X" "Z" "Ysec" "Px" "Py" "Pz" "U" "V" "W" "Pmag"\n')
        fyz.write('VARIABLES= "Y" "Z" "Xsec" "Px" "Py" "Pz" "U" "V" "W" "Pmag"\n')
        fs_all.write('VARIABLES= "Time" "Px_mean" "Py_mean" "Pz_mean" "Px_min" "Px_max" "Px_std" "Px_var" "Py_min" "Py_max" "Py_std" "Py_var" "Pz_min" "Pz_max" "Pz_std" "Pz_var" "|Px|_mean" "|Py|_mean" "|Pz|_mean" "|P|_mean" "qdiff_absmax"\n')
        fs_sec.write('VARIABLES= "Time" "N_all" "Px_mean_all" "Py_mean_all" "Pz_mean_all" "Px_min_all" "Px_max_all" "Px_std_all" "Px_var_all" "Py_min_all" "Py_max_all" "Py_std_all" "Py_var_all" "Pz_min_all" "Pz_max_all" "Pz_std_all" "Pz_var_all" "|Px|_mean_all" "|Py|_mean_all" "|Pz|_mean_all" "Pmag_mean_all" "N_xy"  "Px_mean_xy"  "Py_mean_xy"  "Pz_mean_xy"  "Px_min_xy"  "Px_max_xy"  "Px_std_xy"  "Px_var_xy"  "Py_min_xy"  "Py_max_xy"  "Py_std_xy"  "Py_var_xy"  "Pz_min_xy"  "Pz_max_xy"  "Pz_std_xy"  "Pz_var_xy"  "|Px|_mean_xy" "|Py|_mean_xy" "|Pz|_mean_xy" "Pmag_mean_xy" "N_xz"  "Px_mean_xz"  "Py_mean_xz"  "Pz_mean_xz"  "Px_min_xz"  "Px_max_xz"  "Px_std_xz"  "Px_var_xz"  "Py_min_xz"  "Py_max_xz"  "Py_std_xz"  "Py_var_xz"  "Pz_min_xz"  "Pz_max_xz"  "Pz_std_xz"  "Pz_var_xz"  "|Px|_mean_xz" "|Py|_mean_xz" "|Pz|_mean_xz" "Pmag_mean_xz" "N_yz"  "Px_mean_yz"  "Py_mean_yz"  "Pz_mean_yz"  "Px_min_yz"  "Px_max_yz"  "Px_std_yz"  "Px_var_yz"  "Py_min_yz"  "Py_max_yz"  "Py_std_yz"  "Py_var_yz"  "Pz_min_yz"  "Pz_max_yz"  "Pz_std_yz"  "Pz_var_yz"  "|Px|_mean_yz" "|Py|_mean_yz" "|Pz|_mean_yz" "Pmag_mean_yz"\n')

        n_files = len(files)
        t_group0 = time.time()
        
        for idx, fp in enumerate(files):
            step, ids, types, r, bounds = parse_dump_coords(fp)
            boxlen = np.array([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]], dtype=np.float64)
            origin = np.array([bounds[0], bounds[2], bounds[4]], dtype=np.float64)
            
            id_to_cur_row = np.full(int(ids.max()) + 1, -1, dtype=int)
            id_to_cur_row[ids] = np.arange(len(ids))
            
            ti_rows = id_to_cur_row[ti_ids_sorted]
            valid_ti = ti_rows != -1
            cpos = np.zeros((n_ti, 3), dtype=np.float64)
            cpos[valid_ti] = r[ti_rows[valid_ti]]
            
            # 定期重构 KNN 邻居
            if (idx % conf.rebuild_freq == 0) or not cached_neighbor_ids:
                cached_neighbor_ids = update_knn_cache(cpos, types, r, ids, origin, boxlen, conf.max_knn_dist)

            # 张量极化计算
            P_cells, Pmag, qdiff = compute_polarization_tensors(
                n_ti, cpos, valid_ti, center_ids=ti_ids_sorted, types=types, r=r, ids=ids, 
                id_to_cur_row=id_to_cur_row, origin=origin, boxlen=boxlen, 
                cached_neighbor_ids=cached_neighbor_ids, charge_lut=charge_lut, 
                share_lut=share_lut, max_knn_dist=conf.max_knn_dist
            )
            
            # 冻结网格坐标：导出时始终使用第一帧的绝对拓扑坐标 (r_uw_sorted)
            # 而极化属性 (P_cells) 依然是使用当前帧真实坐标计算出的准确结果
            write_tecplot_zone_3d(fall, step, r_uw_sorted, P_cells, Pmag, nx, ny, nz, is_struct)
            write_tecplot_zone_section(fxy, step, "XY", r_uw_sorted[m_xy], P_cells[m_xy], Pmag[m_xy], z_xy, conf.section_thickness, is_g_xy, nx_xy, ny_xy, sort_xy)
            write_tecplot_zone_section(fxz, step, "XZ", r_uw_sorted[m_xz], P_cells[m_xz], Pmag[m_xz], y_xz, conf.section_thickness, is_g_xz, nx_xz, nz_xz, sort_xz)
            write_tecplot_zone_section(fyz, step, "YZ", r_uw_sorted[m_yz], P_cells[m_yz], Pmag[m_yz], x_yz, conf.section_thickness, is_g_yz, nx_yz, ny_yz, sort_yz)

            row_all = section_stats_row(step, P_cells[valid_ti])
            row_xy = section_stats_row(step, P_cells[m_xy & valid_ti])
            row_xz = section_stats_row(step, P_cells[m_xz & valid_ti])
            row_yz = section_stats_row(step, P_cells[m_yz & valid_ti])
            
            fs_all.write(" ".join(f"{v:.6e}" for v in row_all[0:1] + row_all[2:]) + f" {qdiff:.3e}\n")
            fs_sec.write(" ".join(f"{v:.6e}" for v in row_all + row_xy[1:] + row_xz[1:] + row_yz[1:]) + "\n")

            if DEFAULT_SHOW_PROGRESS:
                prog = (idx + 1) / n_files
                bar = "#" * int(PROGRESS_BAR_LEN * prog) + "-" * (PROGRESS_BAR_LEN - int(PROGRESS_BAR_LEN * prog))
                elapsed = time.time() - t_group0
                eta = (elapsed / (idx + 1)) * (n_files - idx - 1)
                print(f"\r[{bar}] {idx+1}/{n_files} | TS={step} | 耗时={_fmt_hms(elapsed)} ETA={_fmt_hms(eta)}", end="", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] ✔ 全部分析顺利完成！生成文件目录: {DEFAULT_OUT_DIR_BASE}/")

def main():
    parser = argparse.ArgumentParser(description="Polarization computation with static structured mesh and dynamic KNN.")
    parser.add_argument("--pattern", default=DEFAULT_DUMP_PATTERN, help="Glob pattern for dump files.")
    parser.add_argument("--out-prefix", default=OUT_PREFIX, help="Output file prefix.")
    parser.add_argument("--thickness", type=float, default=DEFAULT_SECTION_THICKNESS, help="Section thickness.")
    parser.add_argument("--max-knn-dist", type=float, default=DEFAULT_MAX_KNN_DIST, help="Max distance for KNN.")
    parser.add_argument("--rebuild-freq", type=int, default=DEFAULT_REBUILD_FREQ, help="Frequency of neighbor rebuild.")
    args = parser.parse_args()

    conf = Config(
        pattern=args.pattern,
        out_prefix=args.out_prefix,
        section_thickness=args.thickness,
        max_knn_dist=args.max_knn_dist,
        rebuild_freq=args.rebuild_freq
    )

    files = glob.glob(conf.pattern)
    if not files:
        print(f"未找到匹配的文件: {conf.pattern}")
        return
        
    process_trajectory(files, conf)

if __name__ == "__main__":
    main()