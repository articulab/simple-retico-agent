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

Create a new conda environment `retico` from the YAML file `retico_2.yml` to get all required packages.

```bash
conda env create -n retico -f env_requirements/retico_2.yml
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
