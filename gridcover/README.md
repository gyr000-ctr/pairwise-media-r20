# gridcover

## 题面（可直接粘贴）

子目录：gridcover。掩体覆盖率评估。约束：覆盖率与逐格视线采样一致（≤1%）；掩体不透明；多掩体叠加不重复计数；非法网格可区分报错；512x512 不超过 600 ms；并发安全。验收：cd gridcover && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd gridcover && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
