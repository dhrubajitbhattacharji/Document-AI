from functools import lru_cache
from typing import List, Dict, Any
import os
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except ImportError:
    DocumentFile = None
    ocr_predictor = None

@lru_cache(maxsize=1)
def load_orientation_model(det_arch: str = "db_resnet50", reco_arch: str = "parseq"):
    if ocr_predictor is None:
        raise RuntimeError("python-doctr not installed")
    model = ocr_predictor(
        det_arch=det_arch,
        reco_arch=reco_arch,
        pretrained=True,
        det_bs=2,
        reco_bs=64,
        assume_straight_pages=False,
        straighten_pages=True,
        detect_orientation=True,
    )
    return model

def _to_document_file(path: str) -> Any:
    if DocumentFile is None:
        raise RuntimeError("DocTR not available")
    suffix = Path(path).suffix.lower()
    if suffix == '.pdf':
        return DocumentFile.from_pdf(path)
    else:
        return DocumentFile.from_images([path])


def infer_orientation(paths: List[str],
                      det_arch: str = "db_resnet50",
                      reco_arch: str = "parseq") -> List[Dict[str, Any]]:
    model = load_orientation_model(det_arch=det_arch, reco_arch=reco_arch)

    results: List[Dict[str, Any]] = []
    for idx, p in enumerate(paths):
        doc = _to_document_file(p)
        prediction = model(doc)
        export = prediction.export()
        page_data = export['pages'][0]
        orientation_info = page_data.get('orientation', {})
        angle = orientation_info.get('value', 0)
        np_img = np.array(doc[0])
        rotated = np_img
        if angle and abs(angle) > 1:
            bg = _estimate_background_color(np_img)
            rotated = _rotate_with_canvas(np_img, abs(angle), bg)

        results.append({
            'page_index': idx,
            'path': p,
            'angle': angle,
            'original_image': np_img,
            'rotated_image': rotated,
        })
    return results


def save_orientation_outputs(infos: List[Dict[str, Any]], output_dir: str) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    saved = []
    for info in infos:
        base = Path(info['path']).stem
        before_path = Path(output_dir) / f"{base}_before.png"
        after_path = Path(output_dir) / f"{base}_rot{int(info['angle'])}.png"
        import cv2
        cv2.imwrite(str(before_path), cv2.cvtColor(info['original_image'], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(after_path), cv2.cvtColor(info['rotated_image'], cv2.COLOR_RGB2BGR))
        saved.append({
            'page_index': info['page_index'],
            'angle': info['angle'],
            'before_path': str(before_path),
            'after_path': str(after_path),
            'changed': int(info['angle']) != 0
        })
    return saved

def _estimate_background_color(img_rgb: np.ndarray) -> tuple:
    h, w = img_rgb.shape[:2]
    bw = max(2, int(0.02 * min(h, w))) 
    top = img_rgb[0:bw, :, :]
    bottom = img_rgb[h-bw:h, :, :]
    left = img_rgb[:, 0:bw, :]
    right = img_rgb[:, w-bw:w, :]
    border = np.concatenate([
        top.reshape(-1, 3), bottom.reshape(-1, 3),
        left.reshape(-1, 3), right.reshape(-1, 3)
    ], axis=0)
    med = np.median(border, axis=0)
    r, g, b = [int(x) for x in med]
    return (r, g, b)


def _rotate_with_canvas(img_rgb: np.ndarray, angle_deg: float, bg_rgb: tuple) -> np.ndarray:
    pil_img = Image.fromarray(img_rgb)
    rotated = pil_img.rotate(angle_deg, expand=True, fillcolor=bg_rgb)
    return np.array(rotated)


def orientation_correct_paths(paths: List[str], output_dir: str,
                              det_arch: str = "db_resnet50",
                              reco_arch: str = "parseq",
                              angle_threshold: float = 1.0) -> List[Dict[str, Any]]:

    infos = infer_orientation(paths, det_arch=det_arch, reco_arch=reco_arch)
    saved = save_orientation_outputs(infos, output_dir)
    for s in saved:
        if abs(s['angle']) <= angle_threshold:
            s['effective_path'] = s['before_path']
        else:
            s['effective_path'] = s['after_path']
    return saved
