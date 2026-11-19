# STM32H743 MCA 与伽马能谱识别

这个仓库从一条 2048 道计数谱出发，提供三个可以分别运行的实验：在 C 中提取峰参数，在 Python 中标定能量并解谱，以及用已知标签比较模板匹配和机器学习。公开版包含可移植算法和主机测试。完整板级工程、采集现场配置和原始测量文件保存在私有档案。

| 要做的事 | 代码或结果 | 输入与输出 |
|---|---|---|
| 验证 MCA 算法 | [C 核心](mca/Core/Src/mca.c) · [主机回归](mca/Tests/test_mca.c) | 2048 道计数 → 总计数、背景、峰质心、FWHM |
| 调试收谱链路 | [MCA1 帧格式](mca/Docs/MCA_PROTOCOL.md) · [PC 解码器](mca/PC_Viewer/protocol.py) | 8224 字节帧 → CRC 校验、曝光时长、直方图 |
| 做 K、eU、eTh 定量 | [解谱与合成演示](mca/PC_Viewer/gamma_cli.py) | 样品、本底、标样与活时间 → 三能窗和全谱非负拟合 |
| 比较核素分类方法 | [生成、训练与评估](ai/classification/scripts/prototype.py) · [数据量实验](ai/classification/experiments/data_scale_20261116/conclusions.txt) | 1024 道模拟谱 → 四种目标核素或本底 |
| 评价小样本定量 | [LaBr₃ 回归](ai/quantification/regression.py) · [折外误差](ai/quantification/runs/labr3-regression-v1/metrics.csv) | 实测谱与活时间 → Ra-226、Th-228、K-40 活度浓度 |
| 比较响应拟合算法 | [合成实验](ai/quantification/synthetic_response.py) · [分组指标](ai/quantification/runs/synthetic-response-v1/metrics.json) | 固定响应与泊松噪声 → K/U/Th 模板系数 |

## 无板运行

PowerShell，在仓库根目录运行 C 回归需要 GCC 和 Python。PC 演示依赖列在 `mca/PC_Viewer/requirements.txt`。

```powershell
.\mca\Tests\run-host-tests.ps1
python -m pip install -r mca/PC_Viewer/requirements.txt
python mca/PC_Viewer/viewer.py --simulate
python mca/PC_Viewer/gamma_cli.py demo demo-spectrum
python mca/PC_Viewer/gamma_cli.py run demo-spectrum/manifest.json --output demo-spectrum/result.json
```

演示目录须新建。合成样品的真值为 K 1%、eU 3 ppm、eTh 5 ppm。实测定量还需要确认能量标定、有效活时间、几何条件、独立本底和线性独立的标样。未标定谱可用 `gamma_cli.py inspect` 检查道址和计数。CPS 的分母应取曝光时长，TCP 收到两帧的时间间隔只反映传输节拍。

## 复现 AI 实验

- **核素识别。** 在 `ai/classification` 安装 `requirements-lock.txt` 的依赖，将 [GammaBench 固定版本](https://github.com/triem1998/GammaBench/tree/444156e4befd3754b101848a16e862e502fbcfd3) 检出到该目录的 `upstream/`。运行 `python scripts/prototype.py all --run runs/public-demo --small --epochs 2` 可检查流程。完整实验使用 18 个训练、6 个验证和 6 个留出钢厚条件。训练谱从 5000 增至 20000 条时，留出条件的平均准确率从 98.444% 升至 98.911%；增至 80000 条时为 99.044%。所有划分共享一种本底形状，跨仪器表现仍待验证。
- **八样品回归。** 在 `ai/quantification` 安装 `requirements.lock.txt` 的依赖，从 [LaBr₃ NORM 数据集](https://doi.org/10.17632/n6yfhrz824.1) 获取工作簿。运行 `python regression.py train --source <工作簿路径> --out runs/public-labr3`。程序核验输入文件哈希，每折只用七个训练样品确定标准化参数。对 K-40，Ridge 和 MLP 的折外 MAE 分别为 165.17 和 379.48 Bq/kg。
- **固定响应实验。** 从 [公开模板数据](https://doi.org/10.5281/zenodo.20032395) 获取脚本指定版本的归档，放在 `ai/quantification/data/next_search/mobile.zip`。在 `ai/quantification` 运行 `python synthetic_response.py train --out runs/public-synthetic`。同一母混合的两种曝光时长进入同一划分。每个模板归一到单位和，因此误差以合成系数为单位。

上游资料的来源与许可见各数据页面。此仓库提供实验实现和汇总指标，训练所需的第三方响应库由读者从相应来源取得。

## K-40 单峰与活度浓度

`gamma_cli.py k40` 读取带能量标定的计数谱，检查 1460.8 keV 附近的峰。加入匹配的独立本底后可以判断样品是否有净超额。再提供实测活时间、质量和同条件下的 K-40 已知活度浓度标样，才输出样品的 Bq/kg。示例采用同一合成器生成的三个输入。

```powershell
python mca/PC_Viewer/gamma_cli.py demo demo-k40
Copy-Item mca/Docs/k40-demo.json demo-k40/k40.json
python mca/PC_Viewer/gamma_cli.py k40 demo-k40/k40.json --output demo-k40/result.json
python -m unittest discover -s mca/PC_Viewer/tests -p test_k40.py
```

合成算例的预设值和回算值都是 313 Bq/kg。参见[输入字段、公开实测数据、复现步骤和下一轮验证](mca/Docs/K40.md)。
