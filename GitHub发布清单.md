# 能不能上 GitHub？—— 结论与清单

**结论：能，但必须先做三件事**：① 保留 MIT 版权声明；② 把权重和内嵌运行时挡在仓库外；③ 在 README 里写清"需要自己下权重"。

---

## 1. 许可证：代码 OK，权重不 OK

你改的 [`IgorAherne/trellis-stable-projectorz`](https://github.com/IgorAherne/trellis-stable-projectorz)：

- 它自己是 **MIT**，而且是 [`microsoft/TRELLIS`](https://github.com/microsoft/TRELLIS)（同样 **MIT**）的 fork。
- MIT 允许修改、再发布、甚至商用，**唯一硬性要求**：保留版权声明和许可证全文。

所以你要做的：

- [x] 仓库根目录放一份 `LICENSE`（MIT 全文，含原作者版权行）
- [x] **不要**删掉 `engine/code/LICENSE`
- [x] README 里写明"基于 microsoft/TRELLIS 与 IgorAherne/trellis-stable-projectorz 修改"

**但是** —— 代码 MIT **不代表权重也能发**。你机器上这几个权重各有各的许可证：

| 权重 | 位置 | 许可证 | 能否提交 |
|---|---|---|---|
| RMBG-2.0 | `MODELS/RMBG-2.0` | **非商用**（BRIA） | ❌ 绝对不能 |
| DINOv3 | `MODELS/dinov3` | 自有许可（Meta） | ❌ |
| NAF | `MODELS/NAF` | 自有 | ❌ |
| TRELLIS.2-4B | `MODELS/TRELLIS.2-4B` | 见目录内 | ❌ |
| Pixal3D-fp16 | `MODELS/Pixal3D-fp16` | 需自行确认 | ❌ |
| FLUX.2-klein-4B | `MODELS/FLUX.2-klein-4B` | Apache-2.0 | ❌（体积） |

**一律不进仓库**，README 里给下载方式即可。这样既没有许可证风险，也没有 30GB 的仓库。

## 2. 必须挡在仓库外的东西（约 42 GB）

| 目录 | 体积 | 为什么 |
|---|---|---|
| `engine/code/MODELS` | **31.7 GB** | 权重 + 许可证问题 |
| `engine/code/venv` | **10.3 GB** | 虚拟环境，别人要自己建 |
| `engine/tools/git` | 0.4 GB | 别人的 Git 二进制 |
| `engine/tools/python` | 0.2 GB | 内嵌 Python 运行时 |
| 根目录 `bin/ include/ lib/ lib64/ pyvenv.cfg` | — | 一个装在仓库根的 venv |
| `_backup_trellis2/`、`__pycache__/`、`*_bench*.log`、`*.bak-*` | — | 本地调试残留 |
| `engine/code/o-voxel/third_party/eigen` | — | vendored 第三方，有自己的仓库 |

**我已经写好 `.gitignore`**，上面这些全部覆盖。

## 3. 密钥扫描：干净

我扫了 `app.py` / `pipeline_worker.py` / `api_spz/` / `trellis2/` / `_progress.py` / `.workbuddy-ai/`，
匹配 `sk-*`、`hf_*`、`ghp_*`、`AKIA*`、以及 `api_key/token/password/secret = "..."` 形态：

**没有发现明文密钥。**

> 另外你还**没有 `.git`**，所以历史是干净的 —— 从第一次提交就带 `.gitignore`，
> 就不会出现"提交完才发现塞了 30GB 权重"的经典事故。

## 4. 建议的 README 骨架

```markdown
# Pixal3D / TRELLIS.2 本地 3D 工作台

基于 microsoft/TRELLIS 与 IgorAherne/trellis-stable-projectorz 修改，
针对 **GTX 1070 8GB / Pascal** 做了大量显存与性能适配。

## 本仓库包含什么
- 只用一张 8GB 显卡跑通 TRELLIS.2 / Pixal3D 的显存守卫与降级策略
- Pascal(sm_61) 的注意力后端修复（14 倍提速，见 docs）
- 文生3D 一键包（FLUX.2-klein-4B → 图生3D）

## 本仓库**不含**什么（需要你自己下）
- 所有模型权重（许可证各不相同）
- Python 虚拟环境

## 快速开始
双击 `一键启动.bat` …

## 许可证
MIT。模型权重适用各自许可证。
```

## 5. 发布前自检

```bat
git add -A
git status --short

REM —— 三条必须看的检查 ——
REM 1) 权重/环境/工具目录不能被收进来（engine/code/MODELS、venv、engine/tools）
git ls-files | findstr /i "engine/code/MODELS/ engine/tools/ venv/"        :: 应该没有输出
REM 2) 源码包 engine/code/trellis2/models 必须**在**仓库里（它是代码不是权重）。
REM    以前 .gitignore 的 `**/MODELS/` 在 Windows 大小写不敏感，把它一起排掉了，
REM    结果 clone 后 import trellis2 报"循环导入"。
git ls-files engine/code/trellis2/models | find /c /v ""                    :: 应该是 7，不是 0
REM 3) 体积：只应该有个位数十几 MB 的代码和文档
git count-objects -vH
```

⚠️ 别再用 `findstr /i "MODELS venv tools"` 做黑名单 —— 它会连源码包 `trellis2/models/`
一起报成"违规"，反而教人把这个包删掉。按上面的路径前缀查。

行尾与编码（`.gitattributes` 已经钉死，但要确认导出后仍成立）：

```bat
git ls-files --eol -- *.bat *.ps1 *.sh
REM 期望：*.bat / *.ps1 = i/lf w/crlf（ps1 另需 UTF-8 BOM）；*.sh = i/lf w/lf
```

预编译加速包（可选，但能省别人半小时）：把 `_setup_3d_venv.sh` 编出来的
`prebuilt-wheels/*.whl` 打成 `wheels-py312-torch2.6.0-cu124-sm61.tar`
作为 Release 资产上传；**wheel 不进 git**（`.gitignore` 已排除 `prebuilt-wheels/`、`*.whl`）。


## 6. 可选：想连权重一起发

不行 —— 无论是许可证还是体积都不合适。正确做法：

- 权重放 **Hugging Face / 魔搭** 的模型仓库
- 本仓库只放**代码 + 一个下载脚本**（就像 `_t2i_setup.sh` 里那样自动从魔搭拉）

如果确实要发大文件，用 **Git LFS**，但 RMBG-2.0 这种非商用权重**仍然不能**发。
