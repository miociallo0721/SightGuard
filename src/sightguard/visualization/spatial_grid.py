"""Offline six-panel diagnostics and cell/object evidence inspector."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from sightguard.scene.spatial_grid.schema import SpatialGridFrame, SpatialGridInput

COLORS = {
    "GROUND_EVIDENCE": (70, 180, 60),
    "OCCUPANCY_EVIDENCE": (40, 130, 235),
    "MIXED": (185, 60, 210),
    "UNKNOWN": (105, 105, 105),
}


def save_spatial_preview(
    frame: np.ndarray,
    data: SpatialGridInput,
    result: SpatialGridFrame,
    directory: Path,
    *,
    min_pixel_confidence: float = 0.35,
) -> dict[str, str]:
    """Display scaling is separate from stored evidence; use actual input arrays."""
    directory.mkdir(parents=True, exist_ok=True)
    original = frame.copy()
    semantic = frame.copy()
    candidates = frame.copy()
    if result.source_status["semantic"] == "valid":
        ground = np.isin(data.class_map, (3, 6, 11)) & (data.confidence_map >= min_pixel_confidence)
        semantic[ground] = (semantic[ground] * 0.4 + np.array([70, 180, 60]) * 0.6).astype(np.uint8)
        candidate = data.candidate_components > 0
        candidates[candidate] = (60, 180, 220)
    depth_view = np.zeros_like(frame)
    if result.source_status["depth"] == "valid":
        valid = np.isfinite(data.depth_map) & (data.depth_map >= 0)
        if valid.any():
            low, high = np.quantile(data.depth_map[valid], (0.01, 0.99))
            if high > low:
                scaled = np.zeros(data.depth_map.shape, np.float32)
                scaled[valid] = np.clip((data.depth_map[valid] - low) / (high - low), 0, 1)
                depth_view = cv2.applyColorMap(
                    (scaled * 255).astype(np.uint8), cv2.COLORMAP_INFERNO
                )
                depth_view[~valid] = (105, 105, 105)
    detections = frame.copy()
    for relation in result.object_relations:
        if relation.clipped_bbox_xyxy is not None:
            x1, y1, x2, y2 = (int(x) for x in relation.clipped_bbox_xyxy)
            cv2.rectangle(detections, (x1, y1), (x2, y2), (40, 130, 235), 2)
            cv2.putText(
                detections,
                f"d{relation.detection_index} t{relation.track_id}",
                (x1, max(15, y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
    overlay = frame.copy()
    for cell in result.cells:
        x1, y1, x2, y2 = cell.bounds_xyxy
        roi = overlay[y1:y2, x1:x2]
        roi[:] = (roi * 0.55 + np.array(COLORS[cell.state]) * 0.45).astype(np.uint8)
        cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (220, 220, 220), 1)
    panels = []
    for image, label in zip(
        (original, semantic, depth_view, detections, overlay, candidates),
        (
            "RGB",
            "Ground semantics (not safe)",
            "Relative depth p01-p99 (not meters)",
            "YOLO bbox candidates (not masks)",
            "Grid: green ground / orange bbox / purple mixed / gray unknown",
            "Bottom-connected candidates; pose unknown",
        ),
        strict=True,
    ):
        resized = cv2.resize(image, (640, round(frame.shape[0] * 640 / frame.shape[1])))
        header = np.zeros((30, 640, 3), np.uint8)
        cv2.putText(header, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
        panels.append(np.vstack((header, resized)))
    composite = np.vstack([np.hstack(panels[i : i + 2]) for i in (0, 2, 4)])
    name = f"{data.key.frame_index:06d}"
    if not cv2.imwrite(str(directory / f"{name}.jpg"), composite):
        raise OSError("preview write failed")
    if not cv2.imwrite(str(directory / f"{name}_grid.jpg"), overlay):
        raise OSError("grid write failed")
    return {"comparison": f"previews/{name}.jpg", "grid": f"previews/{name}_grid.jpg"}


def write_evidence_script(path: Path, result: SpatialGridFrame, references: dict[str, str]) -> None:
    payload = {"frame": result.to_dict(), "references": references}
    path.write_text(
        "window.renderEvidence(" + json.dumps(payload, allow_nan=False) + ");", encoding="utf-8"
    )


def write_spatial_viewer(output: Path, samples: list[dict[str, str | int]]) -> None:
    """Lazy local scripts allow file:// inspection without fetch/CORS or a server."""
    template = r"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>SightGuard Spatial Evidence v1</title>
<style>body{font:16px system-ui;margin:24px;background:#111;color:#eee} select,button{font:inherit}
main{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%}pre{white-space:pre-wrap;
word-break:break-all;background:#222;padding:12px;max-height:700px;overflow:auto}
#grid{cursor:crosshair}label{display:block;margin:10px 0}</style>
<h1>Spatial Grid Fusion v1</h1>
<p>图像空间证据；UNKNOWN 不等于安全。框不是真实占用；没有导航指令。</p>
<label>真实采样帧 <select id="sample"></select></label>
<main><section><img id="comparison" alt="RGB / semantic / depth / bbox / grid / candidates">
<p>点击下方网格查看单元证据。绿：地面；橙：框候选；紫：MIXED；灰：UNKNOWN。</p>
<img id="grid" alt="Clickable source-image grid"></section><section>
<label>单元 <select id="cell"></select></label><pre id="evidence"></pre>
<label>对象 <select id="object"></select></label><pre id="relation"></pre><pre id="sources"></pre>
</section></main><script>
const samples=__SAMPLES__;
const $=id=>document.getElementById(id);
let current=null,prefix='';
function option(select,value,text){const o=document.createElement('option');
o.value=value;o.textContent=text;select.append(o);}
samples.forEach((x,i)=>option($('sample'),i,x.video_id+' frame '+x.frame_index));
function cell(){const c=current.cells[Number($('cell').value)];
$('evidence').textContent=JSON.stringify({key:current.key,cell:c},null,2);}
function object(){const o=current.object_relations[Number($('object').value)];
$('relation').textContent=JSON.stringify(o||{status:'no_detections'},null,2);}
window.renderEvidence=payload=>{const target=samples[Number($('sample').value)];
if(payload.frame.key.video_id!==target.video_id
||payload.frame.key.frame_index!==target.frame_index)return;
current=payload.frame;
$('comparison').src=prefix+payload.references.comparison;$('grid').src=prefix+payload.references.grid;
$('cell').replaceChildren();current.cells.forEach((c,i)=>
option($('cell'),i,'r'+c.row+' c'+c.col+' '+c.state));
$('object').replaceChildren();current.object_relations.forEach((o,i)=>
option($('object'),i,'d'+o.detection_index+' track '+o.track_id));
$('sources').textContent=JSON.stringify({key:current.key,sources:current.sources,
failures:current.failures,limitations:current.limitations},null,2);cell();object();};
function sample(){const x=samples[Number($('sample').value)];prefix=x.video_id+'/';
const script=document.createElement('script');
script.src=prefix+'frames/'+String(x.frame_index).padStart(6,'0')+'.js';
script.onload=()=>script.remove();document.body.append(script);}
$('sample').onchange=sample;$('cell').onchange=cell;$('object').onchange=object;
$('grid').onclick=event=>{if(!current)return;const rect=$('grid').getBoundingClientRect();
const x=(event.clientX-rect.left)/rect.width*current.key.width;
const y=(event.clientY-rect.top)/rect.height*current.key.height;
const c=current.cells.find(c=>x>=c.bounds_xyxy[0]&&x<c.bounds_xyxy[2]
&&y>=c.bounds_xyxy[1]&&y<c.bounds_xyxy[3]);
if(c){$('cell').value=c.cell_id;cell();}};sample();</script></html>"""
    (output / "index.html").write_text(
        template.replace("__SAMPLES__", json.dumps(samples, allow_nan=False)), encoding="utf-8"
    )
