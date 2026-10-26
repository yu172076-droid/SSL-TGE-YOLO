# Environments

Integrity and annotation checks need Python 3.9+ and the standard library only. They do not import PyTorch or unpickle weights.

`environments/requirements-original-project.txt` is the original local project requirements file, preserved as historical evidence. It contains old package versions and does not specify a complete tested training lock. Do not treat it as an instruction to install those old wheels on arbitrary current Python versions.

Native detector experiments require a compatible PyTorch/torchvision CUDA environment, NumPy, OpenCV, Pillow, PyYAML, matplotlib, SciPy, tqdm, pandas/seaborn, and the detector's remaining imports. The COCO evaluation path requires pycocotools. YOLOv8n uses a separate Ultralytics environment. TrackEval is an additional dependency; the MIT notice alone is not its implementation.

The Jetson experiments used Orin Nano Super 8GB, the specified 15W mode and TensorRT 8.5.2.2. Raspberry Pi CPU inference used ONNX Runtime 1.23.2. Native training, Jetson FP16/TensorRT and Raspberry Pi FP32 CPU costs are separate environments and must not be pooled.

Preserved experiment scripts include their resource guards and runtime adapters. Rebind paths and verify their expected source/protocol hashes before a new run. No new training or hardware inference was conducted while creating this GitHub release.
