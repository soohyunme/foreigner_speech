# RUN Scratch pretrain
# task.data : path of preprocessed manifest folder
# checkpoint.save_dir : path to save pretrain checkpoint
# task.del_silence : whether use silence options which indicate removing prolonged silence in voice

## before run code, please check config files to modify options required.

MANIFEST_DIR='/path/to/data'

# Select config ['base', 'conformer_base', 'conformer_large']
CONFIG_NAME=base

python fairseq_cli/hydra_train.py \
  task.data="$MANIFEST_DIR" \
  task.del_silence=True \
  checkpoint.save_dir=$(realpath .)/checkpoint/pretrain/scratch_pretrain \
  --config-dir config/pretraining/scratch \
  --config-name base