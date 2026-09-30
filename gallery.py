"""
gallery.py

Builds one static HTML page presenting every annotated image from a run
side by side with its result -- the "presenting the results" half of
visual output, so you're not left opening 72+ individual JPEGs one at a
time to see what happened. Pure static HTML/CSS, no server needed: open
gallery.html directly in a browser, image paths are relative to it.
"""

import html
from pathlib import Path

ACTION_COLOR = {
    "log_healthy": "#3aa93a",
    "flag_for_harvest": "#e0972a",
    "schedule_frequent_monitoring": "#d4b400",
    "flag_for_treatment": "#d23c3c",
}

CARD_TEMPLATE = """
<div class="card">
  <img src="{img_src}" alt="{pod_id}">
  <div class="meta">
    <div class="pod-id">{pod_id} <span class="crop">{crop}</span></div>
    <div class="stage">stage: {stage} &nbsp;|&nbsp; {disease}</div>
    <div class="action" style="background:{color}">{action}</div>
    <div class="trend">{trend}</div>
  </div>
</div>
"""

PAGE_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OpenCV-Farming-Agent — Run Results</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; background: #f4f4f4; margin: 0; padding: 24px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px 0; }}
  .subtitle {{ color: #666; margin-bottom: 20px; font-size: 13px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }}
  .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.15); }}
  .card img {{ width: 100%; display: block; }}
  .meta {{ padding: 8px 10px 10px 10px; }}
  .pod-id {{ font-weight: 600; font-size: 13px; }}
  .crop {{ font-weight: 400; color: #888; text-transform: capitalize; }}
  .stage {{ font-size: 12px; color: #444; margin: 3px 0; }}
  .action {{ display: inline-block; color: #fff; font-size: 11px; padding: 2px 8px; border-radius: 10px; margin-top: 2px; }}
  .trend {{ font-size: 11px; color: #888; margin-top: 4px; font-style: italic; }}
</style>
</head>
<body>
  <h1>OpenCV-Farming-Agent — Run Results</h1>
  <div class="subtitle">{subtitle}</div>
  <div class="grid">
    {cards}
  </div>
</body>
</html>
"""


def build_gallery(entries, out_path, images_dir_name, subtitle=""):
    """entries: list of dicts with pod_id, crop_type, growth_stage,
    disease_flag, disease_name, action, trend, image_filename (just the
    filename, relative to images_dir_name next to this html file)."""
    cards = []
    for e in entries:
        disease = html.escape(e["disease_name"]) if e["disease_flag"] else "no disease"
        cards.append(CARD_TEMPLATE.format(
            img_src=f"{images_dir_name}/{e['image_filename']}",
            pod_id=html.escape(e["pod_id"]),
            crop=html.escape(e["crop_type"]),
            stage=html.escape(str(e["growth_stage"])),
            disease=disease,
            action=html.escape(e["action"]),
            color=ACTION_COLOR.get(e["action"], "#888"),
            trend=html.escape(e.get("trend", "")),
        ))

    page = PAGE_TEMPLATE.format(subtitle=html.escape(subtitle), cards="\n".join(cards))
    Path(out_path).write_text(page)
    return out_path
