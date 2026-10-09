# data/

Training and evaluation data. **You do not need anything in this folder to run the app or the tests.**

| path | in git? | content | licence |
|---|---|---|---|
| `commfor/` | no (git-ignored) | Community Forensics downloads, tile caches (`cache/`) and the CompEval-v2 benchmark images (`ce/`), created by `training/` | CC BY-NC-SA 4.0 |
| `openimages/` | no (git-ignored) | Open Images V7 validation index | CC BY 4.0 annotations, CC BY 2.0 images |
| `ai_faces/`, `deepfake_dataset/`, `rvf10k/`, `train.csv`, `valid.csv` | **yes** (committed by a teammate in `cf7c601`, about 700 MB) | Face datasets of the earlier face-only phase | **Not documented** - see below |

## Licence warning for the committed face datasets
The repository history does not record where `ai_faces/`, `deepfake_dataset/` and `rvf10k/` came from, or under what
licence. They look like copies of public Kaggle/GitHub face datasets, which often have non-commercial or unclear terms
and contain photographs of real people. They have been left in place so that no teammate work is deleted. **Before
making the repository public, the owners should confirm their licences or remove them from the repository** (including
from history if needed). They are used only by the face-only scripts (`training/face_*`), not by the shipped model.
