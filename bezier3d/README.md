# bezier3d

## 题面（可直接粘贴）

子目录：bezier3d。3D 贝塞尔曲线与挠率。约束：曲线通过端点；切线与中心差分一致（≤1e-6）；弧长与稠密采样误差 ≤0.5%；副法线正交（≤1e-9）；退化曲线可区分报错；1 万次采样不超过 500 ms；并发安全。验收：cd bezier3d && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd bezier3d && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
