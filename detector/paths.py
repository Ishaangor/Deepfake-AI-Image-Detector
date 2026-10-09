"""Project paths, derived from this file's location so every entry point works from any working directory."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")          # checkpoints (*.pt), calibration (*.meta.json), model cards (*.card.json)
ARTIFACTS = os.path.join(ROOT, "artifacts")    # evaluation / training records (not needed at runtime)
DATA = os.path.join(ROOT, "data")              # local datasets and caches (git-ignored except the original face sets)
SAMPLES = os.path.join(ROOT, "samples")        # demo images with verified labels + manifest.json
