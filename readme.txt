现在的项目都存放在#main文件夹中，old_case中是之前的方案，已经被修改，现在以main中的信息为准，其中：

Figures中存放结论图片等内容
lammpsfiles中存放模拟的输入模型、data文件、后处理代码等内容
tecplot中存放后处理得到的文件
Theory_compute中存放理论分析代码与文件

# BaTiO3 under y-axis Uniaxial Compression  
## Twinned and Reference Configurations

本项目研究 BaTiO3（钛酸钡）在 y 轴单轴压缩条件下的力学—极化耦合响应，重点比较两类初始构型：**twinned configuration** 与 **reference configuration**。仓库包含模拟输入、活跃计算数据、后处理脚本和分析图，用于研究不同初始构型在相同压缩路径下如何演化，并如何产生不同的宏观力学—极化响应。

## Background

BaTiO3 在机械压缩下的响应并不是单纯的弹性问题，而是 spontaneous polarization、spontaneous strain、domain architecture 和局部重排共同耦合的结果。当前更关注的问题不是“是否存在一个更方便的状态变量”，而是：

**在相同外加载荷下，不同初始构型是否会沿着不同的 strain-accommodation pathway 组织内部状态变化，并由此产生不同的宏观响应。**

## Current Question

本项目当前采用的工作问题是：

**在相同 y 轴单轴压缩下，twinned 与 reference 两类构型是否可以共享部分状态演化骨架，但通过不同的 accommodation pathways 实现应变容纳，并因此产生不同的宏观力学差异？**

当前更倾向的表述是：

**shared state coordinate + distinct accommodation pathways**

而不是更强的 architecture-governs-everything 说法。

## Current Framework

当前使用的一组主要量为：

- \(M_{2,y}=\langle P_y^2\rangle\)：沿 y 方向的极化二阶内容  
- \(m_y(e)=M_{2,y}(e)/M_{2,y}(0)\)：归一化状态进度  
- \(O_{\parallel}=\langle |P_y| \rangle\)：局域 y 向幅值平均  
- \(\chi_{\parallel}=|\langle P_y\rangle|/\langle |P_y|\rangle\)：同号协同 / 正负抵消指标  
- \(E_t\)、\(\delta\sigma_{serr}\)、\(S_{rms}\)：局部活动与动态解锁诊断量  

当前推荐的层级是：

- \(M_{2,y}\)、\(m_y\)：**shared state coordinate / backbone**
- \(\chi_{\parallel}\)：**organization / pathway indicator**
- \(O_{\parallel}\)：**辅助幅值量**
- \(S_{rms}\)、\(E_t\)：**dynamic diagnostics**
- reduced relation：**compact closure**，不作为背景主语

## Current Interpretation

现有分析表明，twinned 与 reference 在相同压缩条件下表现出明显不同的宏观力学响应，但这种差异并不适合简单理解为“主状态变量完全不同”。当前更合理的工作理解是：

- 两者在沿 y 方向的状态演化上可能共享部分骨架；
- 真正拉开差异的，更可能是这些状态变化如何被组织成不同的应变容纳路径；
- reference 更容易表现出显著的组织重排；
- twinned 更容易更早进入局部活动与空间异质化增强阶段。

## Reduced Description

当前保留的简化表达为：

\[
s(e)\approx a_e e + a_M[1-m_y(e)] + r(e)
\]

其中：

- \(s=|\sigma_y|\)
- \(e=|\varepsilon_y|\)

这条式子当前用于 **后半段的 compact closure / reduced summary**，而不是整篇工作的起点。

## Paper-Writing Defaults

为避免后续反复讨论，当前默认采用以下写作前提：

- 论文不从“observable 是否优于 hidden switching fraction”开场；
- 主线优先写成：

  **initial configuration → accommodation pathway → macroscopic response**

- 当前默认术语为：
  - **twinned configuration**
  - **reference configuration**
- `Single` 不再作为论文级默认术语；
- 当前默认 claim 强度为：

  **partly shared state backbone / shared state coordinate + distinct accommodation pathways**

- Introduction 默认四段逻辑为：
  1. domain architecture mediates response  
  2. literature narrows to pathway gap  
  3. twinned vs reference under identical compression  
  4. observables introduced as evidence framework  

## Scope and Boundary

本仓库当前并不试图直接建立一个普适本构，而是提供一个可直接计算、可逐步检验的 reduced description，用于连接：

**initial configuration → state evolution → accommodation pathway → macroscopic response**

当前版本更适合作为：

**当前仓库、当前阶段、当前可读材料下的最小可靠故事框架。**

旧版本中以 hidden switching fraction 为中心的解释已不再作为默认主线。若后续补充更完整的初始构型定量、横向二阶极化项或更强的真实空间结构证据，当前框架仍可继续升级。

## Repository Note

当前有效内容以 `#main` 为准，`old_case` 为旧方案。后续分析、画图和写作默认优先采用 `#main` 中的活跃数据、活跃脚本和活跃分析结果。