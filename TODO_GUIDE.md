# kernel.cu TODO 中文引导

这份文档用于帮助你按实现顺序完成 `src/kernel.cu` 的 TODO。
组织方式不是按文件出现顺序，而是按学习和编码顺序：先朴素版，再改良版，最后内存友好版。

约定：每一节最后都有“标准答案”，放在最末尾，避免你过早看到结论。

## 第一阶段：朴素版（先跑通）

### 1. TODO-1.2 in computeVelocityChange

#### 你在做什么
这里是 boids 三条规则的核心数学：

- 规则 1（cohesion）：朝邻居质心靠近
- 规则 2（separation）：与太近邻居拉开距离
- 规则 3（alignment）：速度和邻居趋同

它返回的是“速度变化量”，不是最终位置。

#### 实现前要想清楚
每条规则都需要按距离阈值筛邻居，而且三个规则阈值不同。你需要决定：

- 是否分别统计三个规则的邻居计数
- 没有邻居时如何避免除零
- 是否排除自己（必须排除）

这一步只读 `pos`、`vel`，不写全局数组。

#### 标准答案
对每个 boid 扫描全部 boid（排除自己），分别按 `rule1Distance`、`rule2Distance`、`rule3Distance` 累加三条规则贡献，最后按 `rule1Scale`、`rule2Scale`、`rule3Scale` 缩放并相加，返回总速度变化。

### 2. TODO-1.2 in kernUpdateVelocityBruteForce

#### 你在做什么
这个 kernel 是朴素速度更新：每个线程负责一个 boid，调用上面的规则函数得到速度变化，再计算新速度并限速。

#### 实现前要想清楚
这里必须用双缓冲：

- 读旧速度：`vel1`
- 写新速度：`vel2`

不能原地写 `vel1`，否则同一帧内后执行线程会读到已经更新过的速度，造成时间步混乱。

#### 标准答案
每个线程取一个 boid，`newVel = vel1[i] + computeVelocityChange(...)`，若速度长度超过 `maxSpeed` 则归一化并乘回 `maxSpeed`，写入 `vel2[i]`。

### 3. TODO-1.2 in stepSimulationNaive

#### 你在做什么
这是 host 侧每帧调度逻辑：先更新速度，再更新位置，再交换速度缓冲。

#### 实现前要想清楚
顺序不能错：

1. 速度 kernel
2. 位置 kernel
3. ping-pong 指针交换

并且线程块配置一般用 `(N + blockSize - 1) / blockSize`。

#### 标准答案
依次 launch `kernUpdateVelocityBruteForce`、`kernUpdatePos`，然后交换 `dev_vel1` 和 `dev_vel2` 指针。

## 第二阶段：改良版（uniform grid, scattered）

### 4. TODO-2.1 / TODO-2.3 in initSimulation（额外缓冲分配）

#### 你在做什么
为网格检索和排序流程准备辅助数组。

#### 实现前要想清楚
至少需要：

- boid 对应的数组索引（value）
- boid 对应的网格索引（key）
- 每个 cell 的起始下标
- 每个 cell 的结束下标

并且要把 `int*` 包成 thrust 的 `device_ptr<int>`，用于 `sort_by_key`。

#### 标准答案
在初始化时 `cudaMalloc` 分配粒子索引数组、粒子网格键数组、cell 起止数组，并初始化对应 thrust 指针包装；后续 coherent 版本还要再补临时重排缓冲。

### 5. TODO-2.1 in kernComputeIndices

#### 你在做什么
把 boid 从连续空间映射到网格离散坐标，并写出排序所需 key-value。

#### 实现前要想清楚
典型流程：

- `gridPos = (pos[i] - gridMin) * inverseCellWidth`
- 转整型 cell 坐标
- 用 `gridIndex3Dto1D` 压平为 1D key

value 就是 boid 原始下标 `i`。

#### 标准答案
每个 boid 线程计算自身 cell 的 1D 索引写入 `gridIndices[i]`，并把自身数组下标写入 `indices[i]`。

### 6. TODO-2.1 in kernIdentifyCellStartEnd

#### 你在做什么
在“已按 cell key 排序”的数组上，找出每个 cell 对应区间。

#### 实现前要想清楚
这是边界检测：

- 与前一个 key 不同：当前是某 cell 的 start
- 与后一个 key 不同：当前是某 cell 的 end

空 cell 需要保留哨兵值，所以通常在这之前要先 reset。

#### 标准答案
比较 `particleGridIndices` 相邻项，遇到 key 变化就把当前下标写进对应 cell 的 start 或 end 表。

### 7. TODO-2.1 in kernUpdateVelNeighborSearchScattered

#### 你在做什么
利用 uniform grid 只访问邻近 cell，减少邻居检查数量。

#### 实现前要想清楚
与朴素版相比，规则不变，变化的是“候选邻居来源”：

- 先找自己在哪个 cell
- 再枚举可能含邻居的周围 cell
- 通过 `gridCellStartIndices` / `gridCellEndIndices` 取该 cell boid 范围
- 再通过 `particleArrayIndices` 反查原 boid 索引

最后仍然是“算三规则 + 限速 + 写到 vel2”。

#### 标准答案
每个 boid 只遍历与其交互范围重叠的邻近 cell，在这些 cell 的区间内读取候选 boid（经 `particleArrayIndices` 间接索引），计算速度变化并限速后写入 `vel2`。

### 8. TODO-2.1 in stepSimulationScatteredGrid

#### 你在做什么
把 scattered grid 的整条流水线串起来。

#### 实现前要想清楚
这不是一个 kernel，而是一组步骤：

1. 生成 key-value（grid index, particle index）
2. `thrust::sort_by_key`
3. reset start/end
4. 识别 start/end
5. 邻域速度更新
6. 位置更新
7. 速度 ping-pong

#### 标准答案
按上述顺序执行整条流水线，并在每帧结束交换 `dev_vel1` / `dev_vel2`。

## 第三阶段：内存友好版（uniform grid, coherent）

### 9. TODO-2.3（全局变量附近）考虑额外重排缓冲

#### 你在做什么
为“数据按 cell 连续”准备临时数组。

#### 实现前要想清楚
scattered 版本只让“索引连续”，但 `pos`/`vel` 本体仍散乱。coherent 版本要让本体也连续，通常要有：

- 重排后的 `pos` 缓冲
- 重排后的 `vel` 缓冲（至少当前速度）

#### 标准答案
增加用于按排序结果重排 boid 数据的临时位置/速度缓冲，并在初始化与释放阶段成对管理。

### 10. TODO-2.3 in kernUpdateVelNeighborSearchCoherent

#### 你在做什么
逻辑与 scattered 近似，但读取 boid 数据时去掉一层间接索引。

#### 实现前要想清楚
区别点：

- scattered：cell 区间 -> particleArrayIndices -> pos/vel
- coherent：cell 区间 -> 直接 pos/vel

算法结构可以复用，内存访问路径要改。

#### 标准答案
沿用同样的邻居 cell 枚举与规则计算流程，但直接按 cell 区间访问重排后的 `pos`/`vel1`，不再通过 `particleArrayIndices` 间接寻址。

### 11. TODO-2.3 in stepSimulationCoherentGrid

#### 你在做什么
在 scattered 流水线基础上，插入“数据重排”步骤。

#### 实现前要想清楚
核心新增动作：排序后利用粒子索引，把旧 `pos`/`vel` 按 cell 顺序拷贝到 coherent 缓冲，再在 coherent 缓冲上做邻域更新。

因此你需要明确：

- 哪些指针在这一帧代表 coherent 数据
- 位置更新是在 coherent 数组上做，还是写回原数组
- 帧末交换时到底交换哪一对速度指针

#### 标准答案
先复用 scattered 的“建网格 + 排序 + 边界识别”，再按排序索引重排 boid 数据到 coherent 缓冲，在 coherent 数据上执行邻域速度更新和位置更新，最后按 coherent 的数据流交换速度缓冲。

### 12. TODO-2.1 / TODO-2.3 in endSimulation（释放）

#### 你在做什么
关闭时释放你新增的所有设备内存。

#### 实现前要想清楚
`cudaMalloc` 和 `cudaFree` 必须一一对应；缺一个就泄漏，多一个可能崩溃。

#### 标准答案
除原有 `dev_pos`、`dev_vel1`、`dev_vel2` 外，释放所有 grid 与 coherent 阶段新增缓冲。

## 最后：写代码前你必须心里清楚的事

### A. 每一帧的时间一致性

- 同一时间步内，所有 boid 必须读旧速度、写新速度。
- 这就是为什么要 ping-pong。

### B. 三个版本的本质差异

- 朴素版：邻居来源是“全体 boid”。
- 改良版（scattered）：邻居来源是“邻近 cell 内 boid”，但 boid 数据本体还散。
- 内存友好版（coherent）：邻居来源仍是邻近 cell，但 boid 数据按 cell 连续，更利于缓存与访存合并。

### C. 你最容易写错的点

- 没排除自己。
- 邻居计数为 0 时直接相除。
- cell 坐标越界或压平索引错误。
- start/end 表没 reset 就直接写。
- 一帧结束后忘记交换速度指针。
- coherent 版本里数据源和目标数组混用。

### D. 推荐实现顺序（实操）

1. 先完成并验证朴素版能稳定运动。
2. 再上 scattered grid，只改“邻居来源”，别同时改太多。
3. 最后做 coherent，把“重排”作为单独步骤验证。

### E. 自测建议

- 先小规模 boid（例如几十个）做正确性检查。
- 再上大规模 boid 看性能差异。
- 每做完一步就加 `checkCUDAErrorWithLine` 做定位。
