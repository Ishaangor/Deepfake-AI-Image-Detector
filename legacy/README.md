# legacy/

The project's original Keras/TensorFlow code (an Xception face classifier), kept for history and rollback only. The
current app imports none of it unless `DETECTOR_BACKEND=legacy` is set. In that case `app.py` runs
`legacy/app_legacy.py` and `predict.py` uses `models/best_xception.keras`. TensorFlow is not in `requirements.txt`, so
install it separately if you need this path. These scripts are neither maintained nor tested.
