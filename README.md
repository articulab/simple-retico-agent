### Son-of-Sara's retico-based dialog system

## Prerequisties

- Anaconda

## Installation

Clone the github repository

```bash
git clone https://github.com/articulab/retico_test.git
```

Go to the Getting started branch

```bash
git checkout getting_started
```

Create a new conda environment `retico` from the YAML file `retico.yml` to get all required packages.

```bash
conda env create -n retico -f env_requirements/retico.yml
```

Download the LLM weights of a quantized version of the Mistral-7B-Instruct-v0.2 model (from MistralAI) with [this link](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_S.gguf?download=true), it is from [this Huggingface page](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF).
At the root of your `retico_test` folder, create a `models` folder and put your newly downloaded `mistral-7b-instruct-v0.2.Q4_K_S.gguf` LLM weights file in it.

### Installation of articulab's clone of CoquiTTS

Clone the github repository

```bash
git clone <https://github.com/articulab/CoquiTTS.git>
```

Install the package on pip to make it available for imports

```bash
pip install -e .
```

## Run the system

Activate your new conda environment

```bash
conda activate retico
```

Run the dialog system

```bash
python main.py
```

The system will them launch every module and start deploying the deep learning models on the GPU (or CPU if it is the option you chose).
After the message "ASR running" has been printed in the terminal, the system is able to hear you through the push-to-talk microphone. To be heard, you must press the `M` key while speaking.
After a short time, you should hear the system answering you.
If you want to quit, and close the system, press `Q` key for a short time.
