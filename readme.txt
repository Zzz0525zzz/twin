# BaTiO3 y 轴单轴压缩：Reference 与 90° Domain-Wall 构型

本项目研究 BaTiO3（钛酸钡）在 y 轴单轴压缩条件下的力学—极化耦合响应，重点比较两类初始构型：

- Reference configuration（Ref.）
- 90° domain-wall configuration（90° DW）

当前有效项目文件存放在 #main 文件夹中。old_case 为早期方案和旧分析流程，仅作参考；后续论文写作、结果分析和绘图默认以 #main 中的内容为准。


1. 目录结构

#main/
├── Figures/          # 正文准备使用的主图
├── lammpsfiles/      # LAMMPS 输入文件、data 文件、后处理代码等
├── tecplot/          # Tecplot 可视化和后处理得到的场数据
├── Theory_compute/   # 理论分析、简化模型和拟合相关代码
├── old_case/         # 旧方案与早期测试结果
├── manuscript.txt    # 阶段性手稿文字、caption、写作草稿等
└── readme.txt        # 当前项目说明

其中：

- Figures/：存放当前准备作为正文使用的主图。
- lammpsfiles/：存放模拟输入、模型文件、data 文件、后处理脚本等。
- tecplot/：存放 Tecplot 可视化文件和后处理得到的场数据。
- Theory_compute/：存放理论分析、简化模型和拟合相关代码。
- old_case/：存放旧方案和早期测试结果，不作为当前默认分析依据。
- manuscript.txt：存放阶段性手稿文字，例如 introduction、figure caption、results discussion、投稿版本草稿等。
- readme.txt：记录当前项目主线、变量定义、目录说明和写作边界。


2. 研究目标

本项目关注的问题是：

在相同 y 轴压缩路径下，Reference 与 90° DW 两类初始构型是否会通过不同的局域极化重排和应变容纳路径，产生不同的宏观力学—极化响应。

当前论文主线不是强调某一个单一变量完全决定响应，而是强调：

initial configuration
→ local polarization reorientation
→ accommodation pathway
→ macroscopic stress response
→ reduced state-variable description

即：

初始构型差异会改变局域极化重排路径，从而改变宏观压缩响应；这种响应可进一步用一个归一化极化状态变量进行简化描述。


3. 当前正文图主线

当前 Figures/ 中准备作为正文使用的主图包括：

Figure 1: Figure1.tif
主要内容：模型尺寸、加载方向、初始 Ref. / 90° DW 构型。
作用：定义模拟体系与初始状态。

Figure 2: Figure2.TIF
主要内容：应力—应变曲线与平均极化响应。
作用：展示两种构型的宏观响应差异。

Figure 3: Figure3.TIF
主要内容：压缩过程中的局域 P_y 空间演化。
作用：解释宏观差异对应的局域极化重排。

Figure 4: Figure4.TIF
主要内容：m_y 状态变量与应力拟合。
作用：给出 reduced state-variable description。

当前四图逻辑为：

Figure 1：初始构型与加载路径
Figure 2：宏观应力和极化响应
Figure 3：局域 P_y 重排机制
Figure 4：m_y 状态变量与简化应力描述


4. 主要物理量定义

4.1 局域极化分量

局域极化的 y 分量记为：

P_y

它是当前分析中连接压缩方向响应和局域极化重排的核心分量。


4.2 极化二阶矩

定义 y 方向极化分量的二阶矩为：

M_{2,y}(ε_y) = <P_y^2>

其中 <...> 表示对当前构型中所有局域极化单元或网格点取空间平均。

M_{2,y} 用于表征体系中 P_y 分量的整体强度，而不是简单的平均极化。


4.3 归一化状态变量

定义归一化二阶矩：

m_y(ε_y) = M_{2,y}(ε_y) / M_{2,y}(0)

其中 M_{2,y}(0) 取同一体系中初始状态或最接近零应变状态下的 M_{2,y}。

当前论文中，m_y 被用作主要的 reduced polarization-state variable，即归一化极化状态变量。


4.4 其他辅助量

后处理和补充分析中还可能使用：

O_parallel = <|P_y|>

表示局域 P_y 幅值的平均水平。

chi_parallel = |<P_y>| / <|P_y|>

用于表征 P_y 的同号协同程度或正负抵消程度。

这些量主要作为辅助诊断，不作为当前正文第一主变量。


5. 简化应力描述

在 Figure 4 和相关理论分析中，使用压缩幅值形式：

e = |ε_y|,    s = |σ_y|

其中：

- e：压缩应变幅值。
- s：压缩应力幅值。

当前保留的简化表达为：

s(e) ≈ a_e e + a_M[1 - m_y(e)] + r(e)

其中：

- a_e e：线性弹性贡献。
- a_M[1 - m_y(e)]：与极化状态变化相关的贡献。
- r(e)：残差项。

这条关系当前只作为 compact closure / reduced summary 使用，用于概括当前两类构型和当前加载路径下的主要趋势；它并不被宣称为普适本构模型。


6. 当前物理理解

当前结果支持以下工作理解：

1. Reference 与 90° DW 在相同 y 轴压缩下表现出明显不同的应力响应。
2. 这种差异不仅来自平均极化大小变化，也来自局域极化方向和空间分布的不同重排路径。
3. 90° DW 构型中，畴壁及其邻域会引入更强的局域异质性和应变容纳自由度。
4. M_{2,y} 和 m_y 能够作为简洁的状态变量，概括 P_y 分量在压缩过程中的整体演化。
5. 结合 m_y 的 reduced relation 可以对压缩应力响应进行紧凑描述，但仍需结合 Figure 3 中的空间极化图理解其物理来源。


7. 术语约定

当前论文和后续分析默认使用以下术语：

- Reference configuration
- Ref.
- 90° domain-wall configuration
- 90° DW

早期代码或旧图中可能出现：

- Single
- Twin
- twinned configuration
- 90°畴

这些属于历史命名或内部命名。除非特别说明，论文级表述应优先使用 Ref. 和 90° DW。


8. 手稿文字管理

manuscript.txt 用于保存当前阶段性的论文文字草稿，包括但不限于：

- title / abstract 草稿。
- introduction 段落。
- figure captions。
- results and discussion 草稿。
- conclusion 草稿。
- 投稿期刊版本的写作尝试。
- 对不同版本文字的阶段性整合。

manuscript.txt 不是最终排版稿，而是当前项目写作过程中的活跃文本记录。后续若形成正式 manuscript，可再单独整理为 .docx、.tex 或其他投稿格式。


9. 当前写作边界

当前项目不试图直接建立 BaTiO3 的普适本构模型，也不把 hidden switching fraction 作为默认主线。

当前更稳妥的论文表述是：

initial configuration
→ distinct local polarization-reorientation pathway
→ different macroscopic stress response
→ compact reduced description using m_y

即：

初始构型影响局域极化重排路径，局域重排路径进一步影响宏观力学响应，而 m_y 可作为一个简洁状态变量对该过程进行概括。


10. 使用说明

后续分析默认优先读取和使用：

#main/Figures/
#main/lammpsfiles/
#main/tecplot/
#main/Theory_compute/
#main/manuscript.txt

old_case 中的文件只用于追溯旧方案，不作为当前论文写作和结果判断的默认依据。

如果后续更新正文图、后处理代码、理论分析代码或阶段性手稿文字，应优先同步更新 #main 下的对应文件，并保持 Figure 编号、变量命名和正文术语一致。
