# sparse-residual-operator

运行于 Linux CPU、仅依赖 NumPy 的线性算子库：推理时按通道幅值过滤输入，
保留的大通道使用原始权重，被过滤的小通道仅经过截断 SVD 低秩补偿，
两部分合成时每个输入通道只贡献一次，不会重复计算。

## 模型

`y = x @ W.T`，其中 `W` 形状为 `(out_features, in_features)`：

- 校准：`threshold = np.quantile(|X_cal|, q, method="linear")`，对**所有**
  校准样本的**所有元素**的绝对值取同一个标量分位阈值。
- 推理：按各样本各通道的绝对值与**已冻结**的阈值比较，决定保留掩码；
  输出 `y = W[:, kept] @ x[kept] + W_lr[:, filtered] @ x[filtered]`，
  其中 `W_lr` 是 `W` 的 `target_rank` 截断 SVD 近似。实现上计算
  `W_lr @ x + (W - W_lr)[:, kept] @ x[kept]`，数学等价且无重复计数。

## 分位数插值规则（明确记录）

采用 NumPy 默认的 **linear** 方法：`n` 个排序后的值，虚拟下标
`i = q * (n - 1)`，在 `floor(i)` 与 `ceil(i)` 处的值之间线性插值。

## 端点与等值规则

- `q = 0`：全部保留（输出与稠密一致）；`q = 1`：全部过滤（输出即低秩
  近似）。端点规则优先于阈值比较。
- `0 < q < 1`：`|x| >= threshold` 的通道被保留，**等于阈值的通道保留**。
- `target_rank = 0`：补偿输出为零；`target_rank = min(W.shape)`（满秩）
  时结果与稠密计算一致。
- 权重数值秩不足时仍允许合法的目标秩，补偿模型只保留真实非零奇异
  成分（见 `effective_rank`），不会填入虚假成分。

## 结果与误差

`forward` 返回 `ForwardResult`：`output`、`dense_reference`（稠密参考）、
`relative_error`（参考范数为零时为 `None`，此时看 `absolute_error`）、
`absolute_error`、`kept_channels`（实际保留通道数）、`kept_mask`。

## 保存与加载

`op.save(path)` 原子写入（先写临时文件再 `os.replace`，失败不覆盖已有
文件）；`LowRankCompensatedLinear.load(path)` 直接恢复冻结阈值，**不会**
重新估计，无需校准数据。

## 错误处理

空校准集、维度不匹配、越界参数（分位数不在 `[0,1]`、秩不在
`[0, min(W.shape)]`）、非有限数值（NaN/inf）均抛出带明确信息的
`ValueError`；未校准即推理抛 `RuntimeError`。校准或保存失败不会破坏
算子已有状态或已有存档文件。

## 运行

```bash
python3 example.py                          # 小矩阵示例（向量 + 批量 + 保存/加载）
python3 -m unittest discover -s tests -v    # 21 个 unittest 自动化测试
```

## 误差容限

全部计算与测试使用 float64：

- 代数恒等断言（组合输出、零秩、端点、逐行一致性）：`rtol=1e-12, atol=1e-12`；
- 经过 SVD 重建的稠密一致断言（满秩、秩亏权重）：`rtol=1e-10, atol=1e-10`；
- 相对误差本身按 `‖y - y_ref‖ / ‖y_ref‖` 计算并在示例中打印。
