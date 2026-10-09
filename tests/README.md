# tests/

    python -m pytest tests -q        # from the repository root; about 3 minutes on a laptop GPU

* `test_detector.py` covers:
  * decoding, determinism, model selection, the CPU override, demo-sample checksums and file diagnostics;
  * agreement between the CLI, the library and the web UI (Streamlit AppTest);
  * the reset button, the decision band, and Grad-CAM's dependence on the class being explained.
* `test_parquet_stream.py` covers the memory-light parquet reader used by the training pipeline.
* `assets/` holds images from the original repository:
  * `face_fake*.jpg` and `face_real.jpg` carry labels from the teammate face datasets.
  * `Fakers.jpg`, `fake_20.png` and `real_20.png` have **no verified label**. They are used only for consistency tests, never as evidence of accuracy.
