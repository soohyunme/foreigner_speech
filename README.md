# Wav2vec 2.0
Foreigner Korean speech voice recognition hackathon - CSLEE

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* To install foreignerspeech and develop locally:
``` bash
git clone https://github.com/soohyunme/foreigner_speech
cd foreigner_speech
pip3 install -e .
```
- We only test this implementation in Ubuntu 20.04.
- DockerFile is also supported in this repo.

## Instructions
 - We support script examples to execute code easily(check `script` folder)

```bash
# Guide to make model with Foreigner-speech(orthographic transcription) 

# [1] unzip dataset
bash script/preprocess/unzip_foreigner_speech.sh

# [2] preprocess dataset & make manifest
bash script/preprocess/make_manifest.sh

# [3] further pre-train the model
bash script/pretrain/run_further_pretrain.sh
 
# [4] fine-tune the model
bash script/finetune/run_foreigner_speech.sh

# [5] inference the model
bash script/inference/evaluate_foreigner_speech.sh
```

## Pretrained model
- E-Wav2vec 2.0 : Wav2vec 2.0 pretrained on Englsih dataset released by Fairseq(-py)
- Foreigner-wav2vec 2.0 : The model further pretrained on Foreigner-speech by using Englsih model
  - Fairseq Version : If you want to fine-tune your model with fairseq framework, you can download with this [LINK](https://drive.google.com/file/d/18qwxstEiG6YZdK536g_9A6Y4mss6ae0Q/view?usp=sharing)

## Dataset
- [Foreigner-speech](https://www.aihub.or.kr/aihubdata/data/view.do?&dataSetSn=505) : Korean speech data of foreigner speakers

## Acknowledgments
 - Our code was modified from [fairseq](https://github.com/pytorch/fairseq) and [K-wav2vec](https://github.com/JoungheeKim/K-wav2vec) codebase. We use the same license as fairseq.
 - The preprocessing code was developed with reference to [Kospeech](https://github.com/sooftware/KoSpeech).

## License
Our implementation code(-py) is MIT-licensed.
