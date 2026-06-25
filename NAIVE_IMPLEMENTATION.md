# Naive 实现说明

这份文档只解释 `src/kernel.cu` 里的朴素版实现，也就是最先跑通的 boids 版本。
它对应的是：每个 boid 直接和所有其他 boid 做比较，不使用 uniform grid。

## 实现了什么

朴素版的核心逻辑分成三层：

1. `computeVelocityChange` 负责计算一个 boid 受到周围 boid 影响后的速度增量。
2. `kernUpdateVelocityBruteForce` 负责给每个 boid 算出新速度，并做限速。
3. `stepSimulationNaive` 负责在主机端把这两个 kernel 按正确顺序串起来，并做 ping-pong。

## 代码位置

- [src/kernel.cu](src/kernel.cu): `computeVelocityChange`
- [src/kernel.cu](src/kernel.cu): `kernUpdateVelocityBruteForce`
- [src/kernel.cu](src/kernel.cu): `stepSimulationNaive`

## 具体实现思路

### 1. `computeVelocityChange`

这个函数是 boids 规则的核心。
我做的是逐个遍历所有 boid，排除自己，然后根据距离阈值分别统计三条规则：

- Rule 1：统计邻居位置，求局部质心，再朝质心方向产生修正
- Rule 2：统计过近的 boid，产生分离推力
- Rule 3：统计邻居速度，求局部平均速度，再对齐速度

最后把三部分速度变化加起来返回。

这一层只负责“算增量”，不负责更新位置，也不负责写回模拟状态。

### 2. `kernUpdateVelocityBruteForce`

这个 kernel 是每帧速度更新的 GPU 实现。
每个线程处理一个 boid，读取 `vel1` 作为旧速度，调用 `computeVelocityChange` 算出修正量，然后得到新速度写入 `vel2`。

为了保证数值稳定，我在写入前做了限速：如果速度长度超过 `maxSpeed`，就把它缩放回最大速度。

这里必须写入 `vel2`，不能写回 `vel1`，因为同一帧里所有 boid 都要基于“旧速度”做计算。

### 3. `stepSimulationNaive`

这个函数是主机侧的调度入口。
我在这里做了三件事：

1. 计算 CUDA launch 配置。
2. 先执行 `kernUpdateVelocityBruteForce`，再执行 `kernUpdatePos`。
3. 在一帧结束后交换 `dev_vel1` 和 `dev_vel2`，把新速度变成下一帧的旧速度。

位置更新使用的是 `kernUpdatePos`，它会根据当前速度推进位置，并把 boid 在空间边界外时绕回场景范围内。

## 为什么要这样写

朴素版的目标不是快，而是先把模拟逻辑跑正确。
它的优点是实现直接、容易验证，缺点是每个 boid 都要检查所有其他 boid，复杂度是 $O(N^2)$，所以 boid 数量大了以后会很慢。

## 这版的验证结果

这版已经可以正常编译并运行。
当前工程构建和启动时，`unitTest` 里的 Thrust 排序输出也能正常显示，说明 CUDA 环境、OpenGL 连接和程序启动流程都没有问题。

## 你后面写改良版时要沿用的约束

- `vel1` 是旧速度，只读。
- `vel2` 是新速度，只写。
- 同一帧里，所有 boid 都必须读取同一份旧状态。
- 速度更新和位置更新必须分成两个步骤。

如果你接下来要写 uniform grid，最重要的是先保证这份朴素版逻辑作为正确性基线稳定工作。
