---
title: Install Mayaku
summary: Install Mayaku with pip, add the optional export packages, and choose the device it runs on.
label: Installation
keywords: mayaku, install, pip, pytorch, cuda, apple silicon
---

## Requirements

- Python 3.10, 3.11 or 3.12.
- Linux or macOS.
- PyTorch 2.4 or newer and torchvision 0.19 or newer. They install with Mayaku if you don't already have them.

## Install

```bash
pip install mayaku
```

This installs everything needed to train, predict, evaluate and download models: `torch`, `torchvision`, `numpy`, `opencv-python-headless`, `pydantic`, `pyyaml`, `typer` and `pycocotools`. It also installs the `mayaku` command.

### GPU builds of PyTorch

`pip install mayaku` installs whichever PyTorch build pip picks by default. On an NVIDIA machine, make sure that build has CUDA support. If it doesn't, install the CUDA build of PyTorch first by following the selector at <https://pytorch.org/get-started/locally/>, then install Mayaku:

```bash
# example: CUDA 12.8 wheels; pick the index URL that pytorch.org gives you
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install mayaku
```

On Apple Silicon the default macOS wheels already include MPS support.

## Optional packages for export

Each export target is an optional extra. Install only the ones you deploy to:

| Target | Install | Notes |
|---|---|---|
| ONNX | `pip install "mayaku[onnx]" onnxruntime` | See the onnxruntime note below. |
| Core ML | `pip install "mayaku[coreml]"` | The `.mlpackage` can be written on any OS, but it is only checked and run on macOS. |
| OpenVINO | `pip install "mayaku[openvino]"` | Intel CPUs, integrated GPUs and NPUs. |
| TensorRT | `pip install "mayaku[tensorrt]"` | Linux with an NVIDIA GPU. Installs `tensorrt` and `nvidia-modelopt[onnx]`. |

**onnxruntime is not in any extra.** CPU and GPU machines need different packages, so install the one for your machine yourself:

```bash
pip install onnxruntime        # CPU, Apple Silicon
pip install onnxruntime-gpu    # NVIDIA GPU (CUDA)
```

You need onnxruntime to export to ONNX as well as to run an ONNX model. After writing the file, `mayaku export onnx` runs it once with onnxruntime to check its outputs against the PyTorch model.

## Check the installation

```bash
python -c "import mayaku; print(mayaku.__version__)"
mayaku --help
```

You can also run the command line as a module. It is the same program:

```bash
python -m mayaku.cli --help
```

## Devices

Every command and function that runs a model takes a `device` argument (`--device` on the command line):

| Value | Meaning |
|---|---|
| `auto` (default) | CUDA if available, else MPS (Apple Silicon), else CPU. |
| `cuda` | The first NVIDIA GPU. `cuda:1` and so on select another GPU. |
| `mps` | Apple Silicon GPU. |
| `cpu` | CPU only. |

Training on more than one GPU is covered in [Training](/mayaku/data-and-training/training/#multiple-gpus). It needs CUDA. MPS supports a single device only.

On Apple Silicon, Mayaku sets the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` when it is imported. Any PyTorch operation without an MPS kernel then runs on the CPU instead of raising an error. If you set the variable yourself before importing Mayaku, including to `0`, your value is kept.

## Install from source

```bash
git clone https://github.com/datamarkin/mayaku.git
cd mayaku
pip install -e .
```

Add extras the same way, for example `pip install -e ".[onnx,openvino]"`.

If the `mayaku` command fails with `ImportError: cannot import name 'app' from 'mayaku.cli.__main__'` after you update a source checkout, the installed command script is out of date. Run `pip install -e .` again to regenerate it.
