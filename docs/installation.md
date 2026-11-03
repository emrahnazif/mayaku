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

## Export runtimes install themselves

`pip install mayaku` has no extras. The runtime an export target needs is installed the first time you export to it or load one of its files, with a line saying what is being installed:

| Target | Installed on first use | Notes |
|---|---|---|
| ONNX | `onnxruntime-gpu` matching your torch's CUDA build with an NVIDIA GPU, else `onnxruntime` | Export runs the file once to check it, so exporting needs it too. |
| Core ML | `coremltools` | The `.mlpackage` can be written on any OS, but it is only checked and run on macOS. |
| OpenVINO | `openvino` | Intel CPUs, integrated GPUs and NPUs. |
| TensorRT | `tensorrt-cu12` (or `-cu13`) 10.x, matching your torch's CUDA build | Linux with an NVIDIA GPU. A TensorRT 11 you installed yourself is used as is; it needs `nvidia-modelopt[onnx]` (torch 2.8 or newer). |

Your installed `torch` and `torchvision` are pinned during every such install, so they are never replaced. On an older NVIDIA driver, install the torch build that matches it first (from [pytorch.org](https://pytorch.org/get-started/locally/)); mayaku then picks TensorRT to match.

To install runtimes yourself (offline machines, locked environments), set `MAYAKU_AUTOINSTALL=0`. A missing runtime then fails with the exact `pip install` command.

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
