现在的项目都存放在#main文件夹中，old_case中是之前的方案，已经被修改，现在以main中的信息为准，其中：

Figures中存放结论图片等内容
lammpsfiles中存放模拟的输入模型、data文件、后处理代码等内容
tecplot中存放后处理得到的文件
Theory_compute中存放理论分析代码与文件

BaTiO3 Twin / Single under Uniaxial Compression

BaTiO3 under y-axis Uniaxial Compression: Twinned vs. Reference Configurations

本项目研究 BaTiO3（钛酸钡）在 y 轴单轴压缩条件下的力学—极化耦合响应，重点比较两类初始构型：twinned configuration 与 reference configuration。仓库包含模拟输入、活跃计算数据、后处理脚本和分析图，用于研究不同初始构型在相同加载路径下如何产生不同的宏观力学—极化演化。

背景
BaTiO3 在机械压缩下的响应并不是单纯的弹性问题，而是 spontaneous polarization、spontaneous strain、domain architecture 与局部重排共同耦合的结果。对这类 ferroelastic/ferroelectric 体系而言，真正重要的问题不是“是否存在某个更方便的状态变量”，而是：在相同外加载荷下，不同初始构型是否会走出不同的 strain-accommodation pathway，并因此表现出不同的应力响应、极化组织和局部活动。

项目问题
本项目当前关注的核心不是再构造一个 hidden variable，而是比较 twinned 与 reference 两类初始构型在相同 y 轴压缩条件下是否共享某种共同的状态演化骨架，同时又通过不同的组织重排与动态解锁路径产生不同的宏观力学差异。换句话说，本项目更关心“共同状态 + 不同路径”，而不是把某个单独指标直接当成整篇工作的起点。

当前分析框架
为描述这条路径，仓库当前使用一组可直接计算的路径坐标：
- M_{2,y} = <P_y^2>：沿 y 方向保留的极化二阶内容；
- m_y(e) = M_{2,y}(e) / M_{2,y}(0)：共享状态进度；
- O_parallel = <|P_y|>：局域 y 向幅值平均；
- chi_parallel = |<P_y>| / <|P_y|>：同号协同 / 正负抵消的组织指标。

在当前框架下，M_{2,y} 与 m_y 更适合作为 shared state coordinate，用来刻画沿加载轴的极化内容如何演化；chi_parallel 更适合作为 pathway indicator，用来区分不同构型在组织重排上的差异；O_parallel 则主要保留为辅助幅值信息。E_t、delta_sigma_serr、S_rms 等量主要用于诊断路径分岔、局部活动和动态解锁过程。

当前理解
现有分析表明，twinned 与 reference 两类构型在相同压缩条件下表现出明显不同的宏观力学响应，但这种差异并不能简单归结为“是否拥有完全不同的主状态变量”。更合理的工作假设是：两者可以在一定程度上共享沿 y 方向的极化状态演化骨架，而真正拉开力学差异的是这些状态变化如何被组织成不同的应变容纳路径。当前结果尤其表明，reference 构型更容易表现出显著的组织重排，而 twinned 构型更容易更早进入局部活动与空间异质化增强阶段。

项目定位
本仓库当前并不试图直接建立一个普适本构，而是提供一个可直接计算、可逐步检验的 reduced description，用来连接：
initial configuration → accommodation pathway → macroscopic mechanical–polarization response

因此，M_{2,y}-based state coordinate、organization indicators 与动力学诊断量的角色，都是服务于这条整体故事，而不是彼此竞争成为“唯一核心变量”。旧版本中以 hidden switching fraction 为中心的解释已不再作为默认主线；后续若补充更完整的初始构型定量、横向二阶极化项或真实空间结构证据，当前框架仍可继续升级。