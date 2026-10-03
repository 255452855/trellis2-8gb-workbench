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

**新机器（什么都没有，第一次装）**

```
双击 start.bat           # 全自动：WSL 发行版 → 系统依赖/CUDA 编译器 →
                         # Python 环境（编译 4 个 CUDA 扩展）→ 权重（约 41GB）→
                         # 起后端 → 打开浏览器。首次约 1~2 小时。
                         # 中途要求重启就重启一次，然后再双击它（已完成的会跳过）。
```

**作者这台机器上的日常入口（实测在用的那三个，行为未改动）**

```
双击 一键启动.bat        # 图生3D
双击 重启后端.bat        # 改过代码之后
双击 一键文生3D.bat      # 一句话 → GLB（首次会下约 24GB FLUX 权重）
```

两条路最终执行的是**同一套启动契约**（`_run_backend.sh`：cwd=`engine/code`、
`ATTN_BACKEND=sdpa`、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,...`、
`HF_HOME=engine/code/MODELS`、权重齐了才 `HF_HUB_OFFLINE=1`），
逐项对照用一个假 python 抓出来比过（`.testrun/` 是本地诊断目录，不随仓库发布）。

## ⚠️ 「解压即用」到底做到哪一步了

`start.bat` 这条链路把以前**只能人工**的步骤自动化了：

| 以前 | 现在 |
|---|---|
| `/home/ccb/...`、`cuda-12.8`、`TORCH_CUDA_ARCH_LIST=6.1` 写死在脚本里 | `_lib.sh` 现场探测（`nvidia-smi --query-gpu=compute_cap` 等），结果写进 `runtime.conf`（gitignore，不进仓库） |
| 3D 环境没有建它的脚本 | `_setup_3d_venv.sh`：装 nvcc/build-essential → 按本卡架构选 PyTorch wheel → 编译 `flex_gemm`/`cumesh`/`nvdiffrast`/`o_voxel`/`natten` → 导入自检。**默认建到 `${TRELLIS2_VENV}-new`，绝不移动或删除已在用的环境** |
| 权重要自己按 README 一段段下 | `_setup_weights.sh`：自动选 HF/魔搭、gated 模型走上游 release zip、校验文件清单、8GB 卡自动把 Pixal3D 转 fp16 |
| pip 源写死某个镜像 | `t2_pick_pip_index` 逐个探测、挑最快的；代理环境下自动加 `no_proxy` |
| 编译扩展要等半小时（`natten` 一个就 28 分 55 秒） | 三层取用：本地 `prebuilt-wheels/` → 同 ABI 标签的 GitHub Release 包 → 现编。现编用 `pip wheel`，**顺手把 `.whl` 留进 `prebuilt-wheels/`**，以后重装环境 34 秒搞定 |

### 预编译 wheel（可选加速）

上游 `FlexGEMM` / `CuMesh` / `o-voxel` 都不发布 wheel（`CuMesh` 仓库连 release 都没有），
`natten` 官方 release 只覆盖 torch 2.7 —— 对这套 torch 2.6.0+cu124 用不上，所以只能现编。
本包的做法是编一次、留产物：

```bash
# 产物默认落在 <项目>/prebuilt-wheels/（已 gitignore，不进仓库）
ls prebuilt-wheels/
#   cumesh-0.0.1-cp312-cp312-linux_x86_64.whl
#   flex_gemm-1.0.0-cp312-cp312-linux_x86_64.whl
#   natten-0.21.0-cp312-cp312-linux_x86_64.whl      ← 现编要 28 分 55 秒的那一个
#   nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl
#   o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl
tar -cf wheels-py312-torch2.6.0-cu124-sm61.tar prebuilt-wheels/*.whl
```

**本仓库已经上传了这个包**，标签 `wheels-py312-torch2.6.0-cu124-sm61`，脚本在第 4 步之前
就会自己去取，你什么都不用做（前提是显卡 sm_6x/7x 且 torch 版本正好 2.6.0+cu124；
标签对不上会自动回退现编，**不会装错 ABI 的二进制**）。

慢网络下的两个实测事实，说清楚免得你以为卡死了：

- 这台机器到 GitHub Release 只有 **~30KB/s**，74MB 要 40 分钟左右 —— 和现编全套（约 35 分钟）
  差不多打平。所以脚本不会因为"下不动"就失败：取不到就现编，取到了才省时间。
- 下载**支持断点续传**（`curl -C -`，半成品留在 `/tmp/trellis2-build/wheels-*.tar`）。
  一轮没下完就再双击一次 `start.bat`，它会接着上次的字节继续下，不会从头再来。

想换源（比如走自己的加速通道或内网镜像）：

```bash
# 只在本次 shell 生效；也可以写进 runtime.conf 同目录的启动脚本里
export TRELLIS2_WHEEL_URL="https://你的镜像/releases/download/wheels-py312-torch2.6.0-cu124-sm61"
```

## ✅ 实测：从发布物一路跑到「引擎就绪」

不是代码审查结论，是这台机器上真跑出来的（GTX 1070 8GB / WSL2 Ubuntu-24.04）：

| 步骤 | 实测结果 |
|---|---|
| `git archive` 导出发布物 | 16 MB / 313 条目；不含权重、`runtime.conf`、调试脚本；`.bat` 全 CRLF、`.ps1` 带 BOM、`.sh` 全 LF |
| 解压到新目录后 `--status` / `--check` | 正常，缺权重时退出码 2（启动器据此才去装） |
| `_setup_3d_venv.sh` 从零建环境 | torch/依赖/natten/cumesh/o_voxel 全部编成；**cumesh 2 分 1 秒、o_voxel 2 分 31 秒、natten 28 分 55 秒** |
| 九个模块自检 | `torch 2.6.0+cu124 / torchvision / gradio 6.0.1 / transformers 4.57.3 / flex_gemm / natten 0.21.0 / nvdiffrast 0.4.0 / cumesh / o_voxel` 全 OK |
| 命中预编译 wheel 重装 | **34 秒**，日志里一句"现编"都没有 |
| 用新环境跑发布物起后端 | 端口 6 秒就绪；首页 HTTP 200（约 161 KB）；模型加载约 6 分钟（权重在 Windows 盘上走 drvfs）后状态面板显示 **就绪**，`TRELLIS.2 ✓` `Pixal3D ✓` |
| 显存守卫 | 日志给出 `VRAM required=4371MiB`、`TRELLIS.2 低显存模式已启用，主模型按阶段读盘`（即按显存自动 CPU offload） |
| 收尾 | 停掉后无残留 `app.py`、显存回落；作者原有环境 `import` 检查仍然完好 |

**仍未实测的部分**：`_setup_weights.sh` 真下 41 GB（要几小时网络）、FLUX 文生3D 环境、
以及一台真正干净的机器（没装过 WSL/驱动）。这几段靠上面同一条代码路径保证，
第一次在别人机器上跑请盯着输出。


想只补某一步，或先看缺什么：

```bash
wsl -d Ubuntu-24.04 --cd <本项目路径> -- bash ./_oneclick.sh --status
bash ./_oneclick.sh --only weights          # 只下权重
```

### 路径与已知坑

- 权重目录被加载侧**硬按相对位置**找（`engine/code/MODELS`，部分用 `os.getcwd()` 拼），
  它不认识 `TRELLIS2_MODELS`。想把权重放别的盘：
  `TRELLIS2_MODELS=/mnt/e/models bash _setup_weights.sh`，脚本会把
  `engine/code/MODELS` 做成指向它的符号链接（若该目录已有文件则直接报错，不删任何东西）。
- 界面里的「模型下载」（`install.py`）把权重下到 `models/`（**小写**）。Windows 不区分
  大小写所以没暴露问题，WSL 里区分，加载侧找的是 `MODELS/` —— 在 Linux 下请改用
  `_setup_weights.sh`。
- ⚠️ **`.gitignore` 的大小写陷阱（这个仓库之前正踩着）**：`**/MODELS/` 本意只排权重，
  但 Windows 上 `core.ignorecase=true`，它同样命中源码包 `engine/code/trellis2/models/`，
  于是整个模型定义包从未被提交 —— 别人 clone 后 `import trellis2` 报的是
  `cannot import name 'models' … partially initialized`，看着像循环导入，其实是被忽略了。
  现在 `.gitignore` 里加了 `!engine/code/trellis2/models/` 例外。自检命令：
  `git ls-files engine/code/trellis2/models | wc -l`（应为 7，不是 0）。
  另：`o-voxel/third_party/eigen` 被排掉是对的（不发布第三方源码），但它是**编译必需**，
  所以 `_setup_3d_venv.sh` 会在编 `o_voxel` 前自动取回 eigen。
- 可用的覆盖变量（也可写进 `runtime.conf`，键名见 `_lib.sh`）：
  `TRELLIS2_VENV` / `FLUX_VENV` / `TRELLIS2_MODELS` / `TRELLIS2_PORT` / `TRELLIS2_DISTRO`。

## 模型与许可证（都不在本仓库里）

| 模型 | 许可证 | 获取方式 |
|---|---|---|
| TRELLIS.2-4B / Pixal3D-fp16 | 见 Hugging Face 模型页 | `_setup_weights.sh`（或界面里的「模型下载」，见上方大小写坑） |
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
