# Wav2vec 2.0
Foreigner Korean speech voice recognition hackathon - CSLEE

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* To install foreignerspeech and develop locally:
``` bash
https://github.com/soohyunme/foreigner_speech
cd foreigner_speech
pip3 install --editable ./
python3 setup.py build develop
```
- We only test this implementation in Ubuntu 20.04.
- DockerFile is also supported in this repo.

## Instructions
 - We support script examples to execute code easily(check `script` folder)

```bash
# Guilde to make model with Foreignerspeech(orthographic transcription) 

# [1] unzip dataset
bash script/preprocess/unzip_foreigner_speech.sh

# [2] preprocess dataset & make manifest
bash script/preprocess/make_manifest.sh

# # [3] further pre-train the model
bash script/pretrain/run_further_pretrain.sh
 
# # [4] fine-tune the model
# bash script/finetune/run_ksponspeech_multimodel.sh

# # [5] inference the model
# bash script/inference/evaluate_multimodel.sh
```

## Pretrained model
 - E-Wav2vec 2.0 : Wav2vec 2.0 pretrained on Englsih dataset released by Fairseq(-py)

## Dataset
 - [Foreignerspeech](https://www.aihub.or.kr/aihubdata/data/view.do?&dataSetSn=505)