# -*- coding: utf-8 -*-
"""
生成进度上报 —— 把当前阶段写到 JSON 文件，浏览器 UI 轮询读取。

为什么要走文件：worker 是独立进程，UI 在主进程，跨进程共享状态用文件最省事。

状态文件路径由环境变量 PIXAL3D_PROGRESS 指定；没设就静默跳过。

状态字段：
  running  —— 是否有正在进行的任务。UI 靠这个判断显示/隐藏进度条，
              而不是靠时间戳新鲜度（阶段之间的空档可能长达几十秒）。
  key      —— 阶段标识，用于换算总进度
  stage    —— 给人看的阶段描述
  step/total/pct —— 阶段内的步进
"""
import json
import os
import time

_PATH = os.environ.get("PIXAL3D_PROGRESS") or ""

# 阶段顺序（用于算总进度）
STAGES = [
    ("preprocess", "预处理图片"),
    ("sparse", "稀疏结构"),
    ("shape_lr", "形状（低分辨率）"),
    ("shape_hr", "形状（高分辨率）"),
    ("shape", "形状"),
    ("texture", "材质"),
    ("decode", "解码网格"),
    ("render", "渲染预览"),
    ("simplify", "网格简化"),
    ("export", "导出 GLB"),
]

# 把 tqdm 的 desc 映射到阶段 key
_DESC_MAP = {
    "Sampling sparse structure": "sparse",
    "Sampling HR shape SLat": "shape_hr",
    "Sampling shape SLat": "shape",
    "Sampling texture SLat": "texture",
}


def _key_for_desc(desc: str):
    """按**前缀**匹配阶段，而不是精确相等。

    实际传进来的 tqdm 描述都带后缀：
        "Sampling sparse structure (proj)"
        "Sampling shape SLat (proj)"
        "Sampling HR shape SLat (proj, 768)"
        "Sampling texture SLat (proj)"
    以前这里是 `_DESC_MAP.get(desc)` 精确匹配 —— 上面四个**一个都匹配不上**，
    于是 report() 每次都在第一行 return，进度条永远拿不到步进，
    只在 decode/render 这类 stage() 调用上跳一下，看起来就像个摆设。
    取最长匹配，避免 "Sampling shape SLat" 抢走 "Sampling HR shape SLat"。
    """
    best = None
    for prefix in _DESC_MAP:
        if desc.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return _DESC_MAP[best] if best else None

# 进程内记住当前状态，避免每次重读文件
_STATE = {"running": False, "key": "sparse", "stage": "", "step": 0, "total": 0, "pct": -1}


def _write():
    if not _PATH:
        return
    payload = dict(_STATE)
    payload["ts"] = time.time()
    try:
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, _PATH)
    except Exception:
        pass


def start():
    """一次生成任务开始。"""
    _STATE.update({"running": True, "key": "sparse", "stage": "准备中",
                   "step": 0, "total": 0, "pct": -1})
    _write()


def finish():
    """任务结束（成功或失败都调），UI 据此隐藏进度条。"""
    _STATE.update({"running": False, "stage": "已完成", "pct": -1})
    _write()


def report(desc: str, step: int = 0, total: int = 0):
    """采样循环里调用：desc 是 tqdm 的描述，step/total 是步数。"""
    key = _key_for_desc(desc)
    if key is None:
        return
    _STATE.update({
        "running": True,
        "key": key,
        "stage": desc,
        "step": step,
        "total": total,
        "pct": int(step / total * 100) if total else 0,
    })
    _write()


def stage(key: str, detail: str = ""):
    """非采样阶段调用（解码、渲染等）。"""
    label = dict(STAGES).get(key, key)
    _STATE.update({
        "running": True,
        "key": key,
        "stage": detail or label,
        "step": 0,
        "total": 0,
        "pct": -1,
    })
    _write()


def clear():
    """彻底删掉状态文件。"""
    if not _PATH:
        return
    try:
        if os.path.exists(_PATH):
            os.remove(_PATH)
    except Exception:
        pass
