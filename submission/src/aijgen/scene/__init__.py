"""SAM 3 scene questions (docs/sam3_vqa_plan.md): closed-dictionary detection, filtering, cache and templates.

vocab      dictionaries (dicts/<dataset>.txt), absent lists, name helpers
detect     SAM 3 model, box filtering, per-frame detections and their cache, per-object depth
geometry   image-plane relations, mask erosion, marker placement
formats    the benches' own prompt formats
templates  what questions may rely on (build_scene) and the question templates (make_items)
render     markers drawn on the saved frames
"""
