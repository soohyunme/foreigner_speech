from tqdm import tqdm
import numpy as np
import json
import os

SEED            = 42
ROOT_PATH       = './'
JSON_PATH       = os.path.join(ROOT_PATH, 'json')
SAVE_PATH       = os.path.join(ROOT_PATH, 'filelist')
json_files      = os.listdir(JSON_PATH)

## Data load
info_list = []
for file in tqdm(json_files):
    with open(os.path.join(JSON_PATH, file), 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    info_list.append(data)

## Remove special text and create character labels
label_dict = {}

print('Start mapping special text')
for x in tqdm(info_list):
    x['transcription']['ReadingLabelText'] = x['transcription']['ReadingLabelText'].replace("\n", "").replace("un/","").replace("sn/","").replace("+","").replace("/","").replace("  "," ").replace(",","").replace(".","")
    for ch in x['transcription']['ReadingLabelText']:
        if ch not in label_dict:
            label_dict[ch] = 1
        else:
            label_dict[ch] += 1

## Save label
with open(os.path.join(SAVE_PATH, 'label.json'), 'w', encoding='utf-8') as f:
    json.dump(sorted(label_dict), f, ensure_ascii=False, indent=2)

## Index shuffle
np.random.seed(SEED)

indices = np.arange(len(info_list))
np.random.shuffle(indices)

## Data split
train_idx   = indices[:-5000]
val_idx     = indices[-5000:]

train_list  = []
val_list    = []

print('Start split')
for i, row in tqdm(enumerate(info_list), total=len(info_list)):
    if i in train_idx:
        train_list.append(row)
    else:
        val_list.append(row)

print('train : %s'%(len(train_idx)))
print('valid : %s'%(len(val_idx)))

## Save json
print('Start save..')
with open(os.path.join(SAVE_PATH, 'foreigner_train.json'), 'w', encoding='utf-8') as f:
    json.dump(train_list, f, ensure_ascii=False, indent=2)

with open(os.path.join(SAVE_PATH, 'foreigner_valid.json'), 'w', encoding='utf-8') as f:
    json.dump(val_list, f, ensure_ascii=False, indent=2)
print('Done !')