"""AI-image detector runtime package.

Public API (shared by app.py, predict.py, tests and the evaluation scripts):

    analyze(image, crop="auto", model=None) -> Result        calibrated probability, verdict, per-region scores, flags
    explain(image, model=None, evidence_for="auto") -> Explanation   Grad-CAM of the model's own output
    available(model=None), active_model_name(), list_models(), load_model_card(model)
    load_rgb(path), to_rgb(pil), jpeg_quality(pil), find_face(pil), default_crop_policy(model)

Modules: inference (model registry + analysis), tiling (native-resolution tiles), gradcam (attention maps and
faithfulness checks), forensics (generic diagnostics the model does not use), paths (project locations).
"""
from .inference import (Explanation, Result, active_model_name, analyze, available, default_crop_policy, device, explain,
                        find_face, jpeg_quality, list_models, load_model_card, load_rgb, model_path, quality_flags,
                        to_rgb, _meta)

__all__ = ["Explanation", "Result", "active_model_name", "analyze", "available", "default_crop_policy", "device", "explain",
           "find_face", "jpeg_quality", "list_models", "load_model_card", "load_rgb", "model_path", "quality_flags",
           "to_rgb"]
