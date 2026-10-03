# AI / 自动化接口说明

本工作台**已经有两个可编程接口**，不用新造。下面写清各自能干什么、怎么调、有什么坑。

---

## A. Gradio API —— `http://127.0.0.1:8080`（推荐，已经在跑）

后端一起来就带着这个 API，`一键启动.bat` 之后直接可用。用 `gradio_client` 调。

### A.1 全部 14 个端点（实测 `dev/_probe_api.py` 输出）

| api_name | 作用 |
|---|---|
| **`/image_to_3d`** | **主生成入口**，见下方完整签名 |
| **`/extract_glb`** | **导出 GLB**：`(decimation_target, texture_size) → (导出的模型, 下载的 glb 文件)` |
| `/_runtime_panel_html` | 顶部状态面板 HTML（判断「就绪/加载中/启动失败」） |
| `/_runtime_panel_html_1` | 同上，另一个绑定 |
| `/_on_generate_click` | `(resolution, model_choice)` → 本次生成的时间预期文案 |
| `/get_seed` | `(randomize_seed, seed) → 随机种子` |
| `/preprocess_image_1` | `(image) → 参考图`（去背景等预处理） |
| `/_download_models` | `(selected_models, source_choice) → HTML` 权重下载 |
| `/start_session` | 会话初始化 |
| `/lambda`, `/lambda_1` … `/lambda_4` | 内部小函数（多视角顺序拼接等），一般不用 |

### A.2 `/image_to_3d` 完整签名（22 个参数，顺序不能错）

```
image, reference_images, seed, model_choice, resolution,
ss_guidance_strength, ss_guidance_rescale, ss_sampling_steps, ss_rescale_t,
shape_slat_guidance_strength, shape_slat_guidance_rescale,
shape_slat_sampling_steps, shape_slat_rescale_t,
tex_slat_guidance_strength, tex_slat_guidance_rescale,
tex_slat_sampling_steps, tex_slat_rescale_t,
enable_python_profiling, enable_torch_profiling, enable_sync_hunter,
profiler_delay_sec, profiler_max_duration_sec, profiler_max_events,
auto_degrade
```

- `auto_degrade`（**第 23 个，新增**，默认 `True`）：UI 上那个「显存不足时自动降级」复选框。
  - `True`：装不下就自动降分辨率 / NAF 目标尺寸，**保证跑得完**。
  - `False`：按原设置硬跑，装不下时由 WSL 驱动把显存页换到系统内存 ——
    不会失败，但**会慢很多**，也可能把宿主机拖卡。
  - 对应环境变量 `PIXAL3D_AUTO_DEGRADE` 与 `PIXAL3D_NAF_ADAPTIVE`（由 worker 按此参数设置）。

- `model_choice`: `'TRELLIS.2'` | `'Pixal3D'`
- `resolution`: `'512'` | `'1024 快速'` | `'1024 高精'` | `'1536'`（注意有空格和中文）
- 返回 `(3d_预览, 状态HTML)`

### A.3 已实测跑通的最小例子

`dev/_test_pixal3d.py` 就是现成的模板（本次测试跑通了 5 次）：

```python
from gradio_client import Client, handle_file

c = Client("http://127.0.0.1:8080")
res = c.predict(
    handle_file("/mnt/d/IDM/TRELLIS2/engine/code/assets/example_image/xxx.webp"),  # image
    [],                 # reference_images 多视角补充
    42,                 # seed
    "Pixal3D",          # model_choice
    "512",              # resolution
    7.5, 0.7, 8, 5.0,   # ss_*
    7.5, 0.5, 8, 3.0,   # shape_slat_*
    1.0, 0.0, 8, 3.0,   # tex_slat_*
    False, False, False, 80, 3, 300,   # profiler 相关
    api_name="/image_to_3d",
)
preview, status = res
```

导出 GLB：

```python
exported, glb_path = c.predict(80, 2048, api_name="/extract_glb")
# decimation_target=面数上限, texture_size=贴图尺寸
```

### A.4 必须注意

1. **代理**：`gradio_client` 走 httpx，会读 `HTTP_PROXY/http_proxy`。本机 `.wslconfig`
   里 `autoProxy=true`，**只要 Windows 系统代理一开，WSL 里就会被注入
   `http_proxy=127.0.0.1:7897`，然后连不上 8080**（报 `httpx.ConnectTimeout`）。
   调用前先清掉这几个变量（`dev/_test_pixal3d.py` 的场景就是这么跑的）。
2. **先等就绪**：端口通了只代表 Gradio 界面起来了，worker 的 pipeline 还要 15 秒左右。
   提前提交会被拒：`'模型还在加载中（约 3~4 分钟），请稍候再试。'`
   轮询 `/_runtime_panel_html` 里出现「就绪」再提交。
3. **别用返回值判断成功与否**：失败时 `/image_to_3d` **照样返回**，只是 HTML 里是
   `<div ...>生成失败</div>` + 错误原因。要自己检查返回体里有没有「生成失败」。
   （`dev/_test_pixal3d.py` 没检查，所以它永远打印 `✅ 完成` —— 这是个坑。）

---

## B. StableProjectorz REST API —— `http://127.0.0.1:7960`（正经 REST，带 HTML 文档）

代码在 `engine/code/api_spz/`，是给 StableProjectorz 插件用的 FastAPI 服务，
**独立进程、独立启动**，不是 Gradio 的一部分。

### B.1 端点（`api_spz/routes/generation.py`）

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/` | 存活信息 |
| GET | `/ping` | 探活 |
| GET | `/status` | 当前生成状态 |
| POST | `/generate` | 主生成（`Dict` 入参，走 UI 那套参数） |
| POST | `/generate_no_preview` | 只生成不预览（**更适合自动化**） |
| POST | `/generate_multi_no_preview` | 多图生成 |
| POST | `/interrupt` | 中断当前生成 |
| GET | `/info/supported_operations` | 支持的操作类型 |
| GET | `/download/model` | 下载模型 |
| GET | `/download/spz-ui-layout/generation-3d-panel` | 前端面板布局 |

- 接口文档：`engine/code/api_spz/api-documentation.html`
- 启动脚本：`engine\run-stableprojectorz\run-stableprojectorz.bat`
  （或 `engine\tools\projectorz-internal.bat`）
- 默认监听 `127.0.0.1:7960`

### B.2 唯一的硬限制

它**自己也会把模型加载进显存**。8GB 卡上它和 Gradio 后端
（`一键启动.bat` 那套）**不能同时跑**，会互相抢显存直接 OOM。
用哪个就先关另一个。

---

## C. 该用哪个？

| 场景 | 选 |
|---|---|
| AI / 脚本要「丢一张图进去，拿一个 GLB 出来」 | **A（Gradio API）**，已经在跑，不用额外启动 |
| 要标准 REST + 给 Blender / SPZ 插件接 | **B（api_spz）** |
| 人手动用 | 浏览器开 `127.0.0.1:8080` |
