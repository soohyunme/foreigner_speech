# Evaluate fine-tuned model

# before run code, please check config files to modify options required.

MANIFEST_PATH=/path/to/data
SAVE_DIR=/path/to/save
CHECKPOINT_PATH=$(realpath .)/checkpoints/finetune/foreigner/further/checkpoint_best.pt ## it is dummy, please modify it

# SUBSET indiates evaluation set. our manifest include valid, test
SUBSET=test

# select decoder type [viterbi, kenlm, parlance, fairseqlm]
DECODER=viterbi

# max token, default:4000000
MAX_TOKEN=4000000

python inference/infer.py $MANIFEST_PATH --task audio_finetuning \
    --path $CHECKPOINT_PATH \
    --gen-subset $SUBSET \
    --results-path $SAVE_DIR \
    --w2l-decoder $DECODER \
    --criterion ctc \
    --labels ltr \
    --max-tokens $MAX_TOKEN \
    --post-process letter