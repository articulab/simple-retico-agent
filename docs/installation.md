# Installation

## Pre-requisties

- CUDA Toolkit installed ([all versions here](https://developer.nvidia.com/cuda-toolkit-archive))

To check which version to install you can run the `nvidia-smi` command on the terminal :

```bash
$ nvidia-smi
Wed Nov 21 19:41:32 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.72       Driver Version: 410.72       CUDA Version: 10.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce GTX 106...  Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   53C    P0    26W /  N/A |    379MiB /  6078MiB |      2%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      1324      G   /usr/lib/xorg/Xorg                           225MiB |
|    0      2844      G   compiz                                       146MiB |
|    0     15550      G   /usr/lib/firefox/firefox                       1MiB |
|    0     19992      G   /usr/lib/firefox/firefox                       1MiB |
|    0     23605      G   /usr/lib/firefox/firefox                       1MiB |
```

On the top right, we can see that the maximum version supported by computer's drivers is `CUDA Version: 10.0`.

## Basic installation (CPU support)

### clone repo

```bash
git clone https://github.com/articulab/simple-retico-agent
```

### Create and activate your virtual environment with python 3.11.7 version

With conda :

```bash
conda env create -n [env_name] python=3.11.7
conda activate [env_name]
```

### Install package and its dependencies

```bash
pip install .
```

## Installation to execute system on GPU (with CUDA support)

### Install cuda support for Deep Learning retico modules (ASR, LLM, TTS) (to speed up greatly the system's execution)

modify the following line with your installed cuda toolkit version (Here `cu118` is corresponds to an installed cuda 11.8 version)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall --no-cache
```

to check that your installation worked, test that torch has cuda supported :

```bash
python
import torch
torch.cuda.is_available()
>>> True
```

If `torch.cuda.is_available()` returns `True`, it worked and your DL models should be able to run on GPU. If it returns `False` your torch or cuda installation has problems (maybe the cuda and torch versions do not correspond to one another)

### Installation for llama-cpp-python's cuda support (GPU execution)

`llama-ccp-python` is a little bit particular and you will need to reinstall a cuda-supported version to make your LLM run on GPU.

To reinstall llama-cpp-python with cuda support on Windows :

```bash
pip uninstall llama-cpp-python && set "CMAKE_ARGS=-DGGML_CUDA=on" && set "FORCE_CMAKE=1" && pip install llama-cpp-python --no-cache-dir
```

to check that your LLM is running on GPU, run your system with SimpleLLMModule's verbose argument set to True :

```python
llm = SimpleLLMModule(
    model_path=None,
    model_repo=model_repo,
    model_name=model_name,
    dialogue_history=dialogue_history,
    device=device,
    context_size=context_size,
    verbose=True,
)
```

And look for the `llm_load_tensors:` lines outputted in terminal during modules initialization :

The following lines are from an execution which has the LLM running on GPU, we can see that the complete model (33 out of 33 NN layers) has been offloaded to GPU :

```bash
...
llm_load_tensors: offloading 32 repeating layers to GPU
llm_load_tensors: offloading output layer to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:        CUDA0 model buffer size =  4095.05 MiB
llm_load_tensors:   CPU_Mapped model buffer size =    70.31 MiB
...
```

The following lines are from an execution which has the LLM running on CPU (no mention of GPU or CUDA, and the `CPU_Mapped model buffer` contains the full model):

```bash
...
llm_load_tensors:   CPU_Mapped model buffer size =  4165.37 MiB
...
```

if it doesn't work, and you are using conda, you can copy the exact environement I am using, and try to reinstall llama-cpp-python's cuda supported version:
`conda env update -n [env_name] -f env_requirements/retico_cuda_curr.yml --prune`
`pip uninstall llama-cpp-python && set "CMAKE_ARGS=-DGGML_CUDA=on" && set "FORCE_CMAKE=1" && pip install llama-cpp-python --no-cache-dir`
