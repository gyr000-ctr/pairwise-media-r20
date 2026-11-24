# imgtile

## 题面（可直接粘贴）

子目录：imgtile。图像分块与边界扩展。约束：分块拼接无损还原（≤1e-12）；padding 模式（reflect/replicate）正确；块尺寸不合时报错；100 万像素不超过 400 ms；并发安全。验收：cd imgtile && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd imgtile && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
