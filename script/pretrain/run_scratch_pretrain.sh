# RUN Scratch pretrain
# task.data : path of preprocessed manifest folder
# checkpoint.save_dir : path to save pretrain checkpoint

## before run code, please check config files to modify options required.

MANIFEST_DIR=/path/to/data

# Select config ['base', 'conformer_base', 'conformer_large']
CONFIG_NAME=base

python fairseq_cli/hydra_train.py \
  task.data=$MANIFEST_DIR \
  checkpoint.save_dir=$(realpath .)/checkpoints/pretrain/scratch_pretrain \
  --config-dir config/pretraining/scratch \
  --config-name $CONFIG_NAME