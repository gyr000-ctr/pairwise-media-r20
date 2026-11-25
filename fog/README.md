# fog

## 题面（可直接粘贴）

子目录：fog。体积雾（解析积分 + 反解）。约束：高度雾必须用解析积分求透过率（与 1e4 步数值积分差 ≤1e-6）；雾量单调且值域 [0,1]；实现双项 Henyey-Greenstein 相位函数并归一（g=0 时各向同性 ≤1e-9），前向散射峰值方向必须与入射方向一致；必须提供反解 distance_for_transmittance(t)（牛顿迭代，误差 ≤1e-6，t∈(0,1]）；密度为 0 时恒等；10 万次求值不超过 300 ms；非法参数可区分报错。验收：cd fog && python -m pytest tests -q。

## 实现要点

- `Fog(density, height_falloff)`：指数高度雾 `rho(h)=density*exp(-k*h)`；
  沿任意斜率射线的光学厚度解析积分
  `tau = rho0*S*(-expm1(-q)/q)`，`q=k*slope*S`，对小 q/负 q 均数值稳定；
  透过率 `T=exp(-tau)`、雾量 `1-T` 严格落在 [0,1] 且随距离单调不减。
- `Fog.distance_for_transmittance(t, height, slope)`：闭式解
  `d=ln(1+km*L/rho0)/km`（`L=-ln t`）+ 带二分兜底的牛顿精修，
  距离误差 ~1e-11；向上射线的渐近透过率不可达、零密度反解均抛
  `FogInversionError`，与 `FogParameterError` 可区分（二者均为 `ValueError`）。
- `henyey_greenstein` / `dual_henyey_greenstein` / `DualHenyeyGreenstein`：
  双项 HG 凸组合，4pi 积分严格为 1；`g=0` 严格各向同性 `1/(4pi)`；
  `mu=cos(theta)` 相对入射光传播方向定义，前向峰值在 `mu=+1`。
- 不可变对象、纯函数求值、无模块级可变状态 → 线程安全；非有限/非数值
  输入分类抛 `FogParameterError`；指数溢出时透过率饱和为 0、雾量为 1。
- 性能：10 万次 factor ≈32 ms、10 万次双项 HG ≈45 ms（限 300 ms）。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd fog && python -m pytest tests -q
```

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
