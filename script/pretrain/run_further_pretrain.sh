# RUN Further pretrain
# task.data : path of preprocessed manifest folder
# checkpoint.save_dir : path to save pretrain checkpoint
# task.del_silence : whether use silence options which indicate removing prolonged silence in voice
# checkpoint.finetune_from_model : checkpoints which is used for init state. You can download english checkpoints in Fairseq github "https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md"

## before run code, please check config files to modify options required.

MANIFEST_DIR=/path/to/data
MODEL_PATH=/path/to/checkpoint

# Select config from ['base', 'conformer_base', 'conformer_large']
CONFIG_NAME=base 

python fairseq_cli/hydra_train.py \
  task.data=$MANIFEST_DIR \
  task.del_silence=True \
  checkpoint.finetune_from_model=$MODEL_PATH \
  checkpoint.save_dir=$(realpath .)/checkpoint/pretrain/further_pretrain \
  --config-dir config/pretraining/further \
  --config-name $CONFIG_NAME