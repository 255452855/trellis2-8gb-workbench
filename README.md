# Pixal3D / TRELLIS.2 本地 3D 工作台（8GB 显卡适配版）

把微软 **TRELLIS.2 / Pixal3D** 图生3D 塞进一张 **GTX 1070 8GB（Pascal sm_61）**，
外加一个**文生3D 一键包**（FLUX.2-klein-4B → 图生3D）。

基于 [`microsoft/TRELLIS`](https://github.com/microsoft/TRELLIS) 与
[`IgorAherne/trellis-stable-projectorz`](https://github.com/IgorAherne/trellis-stable-projectorz) 修改，
两者均为 MIT。主要工作是把"跑不起来"改成"跑得完"。

> 本仓库**不含任何模型权重**。所有 checkpoint 由脚本在首次运行时从官方渠道
> （Hugging Face / 魔搭）自动下载。见下方「模型与许可证」。

---

## 这个仓库解决了什么

| 问题 | 处理 |
|---|---|
| Pascal 没有 Tensor Core，PyTorch 在 sm<80 上自动选中 mem-efficient 后端，**慢 14 倍** | 关掉 flash/mem-efficient SDPA，改用 `bmm+softmax`；实测 4096² 自注意力 **825 ms → 58 ms** |
| 显存守卫用**线性**模型估算，10407 token 算出「装得下」，实际注意力矩阵要 9.4 GB → 硬塞 → WSL 虚拟机崩溃 | 改成解二次方程 `a·t + q·t² = usable`，`live_limit` 38477 → **5556**，自动 1024→896→768 降级 |
| `PYTORCH_CUDA_ALLOC_CONF` 被三个文件互相覆盖，最终生效的是遗留的 `max_split_size_mb:128` | 在启动器里统一设置 `expandable_segments:True`，三处对齐 |
| 材质阶段 NAF 在 1024² 上做邻域注意力，8GB 卡唯一越界的阶段 | `_adaptive_naf_target()` 按可用显存自动降到 640² |
| 阶段之间不回收闲置权重 | `_evict_idle_modules()` 在材质/解码阶段前退回用不到的权重 |
| 文生3D 需要 4B 文生图模型 + 13GB 显存 | 独立 venv + `model_cpu_offload`，**出图前先停 3D 后端**（否则 WSL 内存被吃干） |
| 进度条是摆设（`_DESC_MAP` 精确匹配，真实 desc 全带后缀） | 改成最长前缀匹配 |

## 实测性能（GTX 1070 8GB，8 步）

| 阶段 | 耗时 |
|---|---|
| 稀疏结构采样 | 61 s（修复前 385 s） |
| 形状采样（512） | 35 s |
| HR 形状采样（自动降级 768） | ≈120~200 s |
| 材质采样 | 107 s |
| 解码 + 渲染 + 导出 | ≈3 分钟 |
| **Pixal3D 合计** | **约 8 分钟** |
| 文生图 FLUX.2-klein-4B（512², 4 步） | **236~383 s** |

## 快速开始

```
双击 一键启动.bat        # 日常启动图生3D
双击 重启后端.bat        # 改过代码之后
双击 一键文生3D.bat      # 一句话 → GLB（首次会下约 24GB FLUX 权重）
```

运行在 **WSL2 (Ubuntu-24.04)** 里。Python 环境 `/home/ccb/trellis2-wsl-venv`（需自建）。

## ⚠️ 环境依赖：这不是「解压即用」的包

**本仓库是代码快照，不含运行环境，也不含权重。** clone 下来能读到全部源码和文档，
但要真正跑起来，还需要自己准备下面这些。请先读完再动手。

### 1. 文生图环境 —— 有脚本，一键建

```bash
bash _t2i_setup.sh     # 建 /home/ccb/flux-venv + 从魔搭下 FLUX.2-klein-4B（约 24GB）
```

### 2. 3D 环境 —— **本仓库没有建它的脚本**（已知缺口）

`/home/ccb/trellis2-wsl-venv` 需要你自己建，而且**不是 `pip install -r` 就能搞定**：
除了 torch / transformers / gradio，还要编译若干带 CUDA 的扩展：

| 扩展 | 用途 | 本仓库里的编译脚本 |
|---|---|---|
| `flex_gemm` | 稀疏卷积后端 | 无（需按上游说明编译） |
| `natten` | NAF 邻域注意力（Pixal3D 必需） | 无 |
| `nvdiffrast` | 可微光栅化 | `_build_nvdiff.sh` / `_build_nvdiff2.sh` |
| `renderutils` | 渲染 | `_build_renderutils.sh` / `_build_renderutils2.sh` |
| `cumesh` | 网格简化 | `_build_cumesh.sh` |

> 这些脚本里的路径原本写死了作者机器的位置，**用之前先按自己环境改一遍**。

### 3. 模型权重 —— 自动下载，但要自己确认许可证

见下一节。TRELLIS.2 / Pixal3D 的权重通过界面里的「模型下载」或 `_准备环境.py` 获取。

### 4. 路径可移植性

主流程脚本（`_启动WSL后端.ps1`、`_t2i_*.sh`、`t2i_to_3d.py`）已改成**按脚本自身位置
推导路径**，clone 到任意目录都能用。诊断脚本（`_bench_*.py` / `_check_*.py`）
同样已改为 `_ROOT` 相对路径。

环境位置可用环境变量覆盖：

```bash
export TRELLIS2_PY=/你的/venv/bin/python    # 3D 环境
export FLUX_PY=/你的/flux-venv/bin/python   # 文生图环境
export FLUX_VENV=/你的/flux-venv            # 建环境时用
export FLUX_MODEL_DIR=/你的/FLUX.2-klein-4B # 权重位置
```

## 模型与许可证（都不在本仓库里）

| 模型 | 许可证 | 获取方式 |
|---|---|---|
| TRELLIS.2-4B / Pixal3D-fp16 | 见 Hugging Face 模型页 | `_准备环境.py` / 界面里的「模型下载」 |
| RMBG-2.0 | **非商用** | 首次运行自动下载 |
| DINOv3 / NAF | 各自条款 | 自动下载 |
| FLUX.2-klein-4B | Apache-2.0 | `_t2i_setup.sh` 从[魔搭](https://modelscope.cn/models/black-forest-labs/FLUX.2-klein-4B)下载 |

**请勿**把权重提交进仓库，也请勿再分发非商用权重。

## 文档

- [`使用说明.md`](使用说明.md) —— 极简使用
- [`使用与测试手册.md`](使用与测试手册.md) —— 完整手册：启动链路、每个脚本、L0~L4 分级测试、故障排查
- [`文生3D说明.md`](文生3D说明.md) —— 文生3D 一键包
- [`AI接口.md`](AI接口.md) —— 给 AI / 自动化的调用说明
- [`测试报告-2026-10-02.md`](测试报告-2026-10-02.md) —— 实测数据与根因分析
- [`GitHub发布清单.md`](GitHub发布清单.md) —— 发布前自检

## 许可证

MIT（见 [`LICENSE`](LICENSE)）。模型权重适用各自许可证，与本仓库无关。
