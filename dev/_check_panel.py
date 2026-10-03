# -*- coding: utf-8 -*-
import re, sys
from gradio_client import Client
c = Client("http://127.0.0.1:8080")
html = c.predict(api_name="/_runtime_panel_html")
html = html[0] if isinstance(html, (list, tuple)) else html
text = re.sub(r"<[^>]+>", " | ", html)
text = re.sub(r"\s*\|\s*(\|\s*)+", " | ", text)
print(text.strip()[:800])
