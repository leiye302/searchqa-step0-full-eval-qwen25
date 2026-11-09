# Search-QA: Qwen2.5 1.5B and 3B, update 0

只评测两份原始 Qwen2.5 Instruct 权重。默认依次完成 **1.5B 和 3B 各 51,713 题**；
没有 SFT、GRPO/OPSD 更新、训练断点加载、checkpoint 保存、模型导出或轨迹落盘。
不需要作者本机、私有数据、API key 或外部付费检索。

## 评测口径

- 固定公开模型：`Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；
  `Qwen/Qwen2.5-3B-Instruct` @ `aa8e72537993ba99e69dfaafa59ed015b17504d1`。
- [同口径的 Search-QA 公开配方](https://github.com/leiye302/grpo-searchqa-qwen25-3b/tree/a054b5c69afd2d74b258dc8101682a6fbe2f04d7)
  固定到 `a054b5c6`；原生 [SDAR](https://github.com/ZJU-REAL/SDAR/tree/29dcb12a55d8fc17ba6ae03fd0a69048f43051f3)
  负责提示词、解码、环境动作、检索反馈和计分。这里仅替换评测控制及记录层。
- 使用 [Search-R1 公开 test.parquet](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/tree/b7d80abfee334a7a91cb377544f09180d58b34f6)
  的全部题目，固定顺序，不随机抽样。七个数据集分别为 NQ 3,610、TriviaQA 11,313、PopQA 14,267、
  HotpotQA 7,405、2WikiMultiHopQA 12,576、MuSiQue 2,417、Bamboogle 125。
  **不跑 448 题监控子集**，512 是评测 batch size，不是题数。
- 原生 `<search>` / `<answer>`，greedy，每题 1 条，最多 4 次环境动作，
  history 4，prompt 4096、每次生成 512 tokens；Wiki18 + E5-base-v2 + FAISS，top-3。
- 每题只计一次 native normalized exact match（任一参考答案别名匹配即得 1）。
  **Macro EM** 为七个数据集 EM 的等权均值，**Micro EM** 为总答对数除以 51,713。
  原生按决策行计的指标另外保留，不与题级 EM 混用。

原生启动器要求构造非空的训练数据加载器；这里让它引用同一个 test 文件作为占位，
但评测控制器不会遍历这个训练加载器，不启用训练用的 reference worker，也不会调用训练循环。
题目漏评、顺序不符、检索最终返回明确错误时，退出并保留日志，不发布全量汇总。

## 环境与准备

沿用 4×A100 80GB 配方：Linux，支持 CUDA 12.8 的驱动，建议 32+ CPU 线程、256 GiB 内存。
即使不写 checkpoint，Wiki18 Flat 索引仍需本地空间：仅索引分片和合并副本约 120 GiB，
另有语料、模型、环境和缓存。全新机器建议预留约 200–250 GiB；两尺寸共用一套检索资产。
已有校验通过的原版检索目录可直接复用。

```bash
git clone https://github.com/leiye302/searchqa-step0-full-eval-qwen25.git
cd searchqa-step0-full-eval-qwen25
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python evaluate.py setup
python -m pip install -r .deps/searchqa-grpo/configs/requirements.txt
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
python -m pip install -e .deps/searchqa-grpo/.deps/sdar -c .deps/searchqa-grpo/configs/requirements.txt
python -m pip install -e .deps/searchqa-grpo/.deps/sdar/agent_system/environments/env_package/search/third_party -c .deps/searchqa-grpo/configs/requirements.txt
python -m pip check
# Download the pinned 1.5B/3B bases and prepare only the public full test.
python evaluate.py prepare
# CPU-only config check, no retrieval/weights/rollouts used:
python evaluate.py plan
```

在同一服务器的另一个终端启动原版检索服务（先 `cd` 到本仓库根目录）：

```bash
conda create -n searchqa-retriever python=3.10 -y
conda activate searchqa-retriever
conda install numpy=1.26.4 faiss-gpu=1.8.0 -c pytorch -c nvidia -y
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r .deps/searchqa-grpo/configs/requirements-retriever.txt
python -m pip check
python .deps/searchqa-grpo/src/retrieval.py --directory "$PWD/data/retrieval" --prepare
python .deps/searchqa-grpo/src/retrieval.py --directory "$PWD/data/retrieval" --serve --gpus 0,1,2,3
```

服务使用四张已分配卡的原生 GPU FAISS/E5，不需要配置 API key。不要将服务端口暴露到公网。
如使用其他 GPU 编号，保持服务和评测的 `--gpus` 与实际分配一致；如复用索引，在两个命令中
分别指定同一绝对路径的 `--directory` / `--retrieval-dir`，不得换检索语料或 top-k。

## 启动与结果

检索服务就绪后，在评测终端执行：

```bash
source .venv/bin/activate
python evaluate.py run --gpus 0,1,2,3 --output runs/base-full
# 在另一个终端观察当前进度：
tail -f runs/base-full/1.5b/run.log
```

默认先跑 1.5B，退出其 Ray/vLLM 进程后再跑 3B，避免同时抢显存。也可用 `--size 1.5b`
或 `--size 3b` 单跑，指定一个新的 `--output`；不覆盖原日志。第二个模型中途失败时，
第一个模型的已完成结果仍在原目录，重跑失败的模型即可。未完成的评测不算成功。

- 每尺寸：`run.log`（进度和错误）、`metrics.json`（七组 EM 和汇总、耗时）、
  `binding.json` 和 `resolved_config.yaml`（版本与配置核验）。
- `summary.json`：所有选定尺寸完成后的汇总。**不会保存模型、训练状态或逐条生成轨迹**。
  模型下载和检索索引是可复用的输入缓存，不是评测输出。

测试覆盖原生验证入口的数据边界、完整题集/模型校验和只写日志的约束；
CPU 测试不是 GPU 实跑成绩。上游代码和模型各自的许可证仍有效，请查阅相应仓库。
