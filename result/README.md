# 给教授展示的研究报告

当前没有正式实验报告。合成测试在临时目录运行，不放入这里，也不作为研究结果展示。

主流程使用论文的冻结 ResNet-152 + XGBoost 架构，在 GRAPE 上分别训练三个进展模型，
进行嵌套患者级内部验证，不称为原诊断模型的外部验证。

每次 `train-grape --run-name <名称>` 创建一个独立目录，不覆盖已有运行。
根目录的 `bash run_grape.sh` 可一次完成特征提取、模型训练、纵向评估与报告生成；可用
`--run-name <名称>` 指定报告目录，未指定则自动生成名称。
完成后 [INDEX.md](INDEX.md) 列出全部完成的报告；根目录 README 展示最近完成的运行，
不自动挑选效果最好的运行。失败的目录保留 `status.json` 和原因，不列为完成报告。

本目录及报告、图表、CSV、JSON 真实保存在 home：
`/users/zeyuhan/charlie_codebase/DA-SPL/result/<名称>/`。
模型和冻结图像特征另存 `/scratch/users/zeyuhan/DA-SPL/artifacts/<名称>/`，
原始数据和缓存也继续保存在 scratch。运行命令不变，模型与报告使用同一运行名对应。

每次运行的主要文件：

- `report.html`：可直接打开、单文件分享的英文报告，图像已嵌入；可由浏览器打印为 PDF。
- `report.md`：报告文本，便于修改、复制教授的队列级总结。
- `figures/`：三种结局的 A/B 比较、配对差值及描述性分数轨迹，含 PNG/PDF/SVG。
- `primary_results.csv`、`supplementary_metrics.csv`：结果表。
- `predictions.csv`、`temporal_features.csv`：逐眼外层折外预测与时间特征。
- `visit_predictions.csv`：按结局及内外层划分保存的进展相关就诊分数。
- `evaluation.json`：完整指标、置信区间、患者划分、训练折标准化与逻辑回归系数。
- `cohort.json`、`exclusions.csv`：队列统计、时间窗口及排除原因。
- `provenance.json`：模型、数据、编码器和环境来源；旧固定诊断模型模式另保存 `visit_scores.csv` 和旁车 JSON。
- `status.json`：运行是否完整完成。

生成的运行目录默认不提交 Git，但会保留在本机。分享完整分析包时复制整个运行目录；
训练模型及冻结特征另存 `artifacts/<同一运行名>/`，完整复核时一起保留。
仅展示报告时可单独分享 HTML。请保留不同运行的研究目的，不能用反复试参数后最好的
结果替代预先规定的主分析。
