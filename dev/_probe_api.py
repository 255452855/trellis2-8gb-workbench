# -*- coding: utf-8 -*-
"""探测运行中的 app 的 Gradio API，找出生成接口与参数。"""
import json
import sys

from gradio_client import Client

URL = "http://127.0.0.1:8080"

try:
    c = Client(URL)
except Exception as e:
    print("连接失败:", type(e).__name__, e)
    sys.exit(1)

print("=== 已连接 ===")
try:
    api = c.view_api(return_format="dict")
except TypeError:
    # 老版本不支持 return_format
    c.view_api()
    sys.exit(0)

named = (api or {}).get("named_endpoints", {})
print("接口数量:", len(named))
for name, info in named.items():
    params = info.get("parameters", [])
    print(f"\n--- {name}  ({len(params)} 个参数) ---")
    for i, p in enumerate(params):
        print(f"  [{i}] {p.get('parameter_name')}  ({p.get('python_type',{}).get('type','?')})"
              f"  default={p.get('parameter_default')!r}")
    rets = info.get("returns", [])
    print(f"  返回 {len(rets)} 个")
