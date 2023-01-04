# RUN finetune with Multi-task architecture
# task.data : path of preprocessed manifest folder (multi-model must run in script which has grapheme and character vocabulary)
# checkpoint.save_dir : path to save finetune checkpoint
# model.w2v_path : pre-trained checkpoints to use for fine-tuining (use either further-pretrained model or scratch-pretrained model)

## before run code, please check config files to modify options required.

MANIFEST_DIR=/path/to/data
MODEL_PATH=/path/to/checkpoint

# Select config from ['100h', '960h']
CONFIG_NAME=100h 

python fairseq_cli/hydra_train.py \
  task.data=$MANIFEST_DIR \
  checkpoint.save_dir=$(realpath .)/checkpoints/finetune/foreigner/further \
  model.w2v_path=$MODEL_PATH \
  --config-dir configs/finetuning  \
  --config-name $CONFIG_NAME