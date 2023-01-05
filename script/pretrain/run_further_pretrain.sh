# RUN Further pretrain
# task.data : path of preprocessed manifest folder
# checkpoint.save_dir : path to save pretrain checkpoint
# checkpoint.finetune_from_model : checkpoints which is used for init state. You can download english checkpoints in Fairseq github "https://github.com/pytorch/fairseq/blob/master/examples/wav2vec/README.md"

# before run code, please check config files to modify options required.

MANIFEST_PATH=/path/to/data
MODEL_PATH=/path/to/checkpoint

# Select config from ['base', 'conformer_base', 'conformer_large']
CONFIG_NAME=base 

python fairseq_cli/hydra_train.py \
  task.data=$MANIFEST_PATH \
  checkpoint.finetune_from_model=$MODEL_PATH \
  checkpoint.save_dir=$(realpath .)/checkpoints/pretrain/further_pretrain \
  --config-dir config/pretraining/further \
  --config-name $CONFIG_NAME