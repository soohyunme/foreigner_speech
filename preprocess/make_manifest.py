import argparse
import glob
import re
import os
import sys
import random
import json
import numpy as np
import soundfile
from preprocess import preprocess, sentence_filter


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="wav", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument(
        "--token-limit", default=sys.maxsize, type=int, help="maximum number of characters"
    )
    parser.add_argument(
        "--test-percent",
        default=0.05,
        type=float,
        metavar="D",
        help="percentage of data to use as test set (between 0 and 1)",
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument('--preprocess-mode', type=str, default='phonetic',
        help='Ex) (70%)/(칠 십 퍼센트) 확률이라니 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)/(백 프로)가 왜 안돼?'
                'phonetic: 칠 십 퍼센트 확률이라니 모 몬 소리야 진짜 백 프로가 왜 안돼?'
                'spelling: 70% 확률이라니 뭐 뭔 소리야 진짜 100%가 왜 안돼?')
    return parser


def convert_wav_path(path):
    parts = os.path.dirname(path).split(os.sep)
    if parts[-3] == '1.Training':
        new_path = path.replace(parts[-1], 'TS_' + parts[-1])
    elif parts[-3] == '2.Validation':
        new_path = path.replace(parts[-1], 'VS_' + parts[-1])
    
    new_path = new_path.replace('라벨링데이터', '원천데이터')
    return new_path


def load_script(args, data_name, json_list):
    errors = list()
    file_info = list()
    transcriptions = list()
    texts = list()

    reg = re.compile(r'.*[a-zA-Z0-9]')
    limit_count = 0
    remove_count = 0

    for fname in json_list:
        wav_dir = convert_wav_path(os.path.dirname(fname))
        with open(fname, 'r') as f:
            info_data = json.load(f)
            file_path = os.path.join(wav_dir, info_data['fileName'])
            
            if (info_data['transcription']['ReadingLabelText']):
                transcription = info_data['transcription']['ReadingLabelText']
            else:
                transcription = info_data['transcription']['AnswerLabelText']
            new_sentence = sentence_filter(raw_sentence=transcription, mode=args.preprocess_mode)
            
            if len(new_sentence) > args.token_limit:
                limit_count+=1
                continue
            if reg.match(new_sentence):
                remove_count+=1
                continue

            try:
                frames = soundfile.info(file_path).frames
            except:
                errors.append(file_path)
                continue

            file_info.append("{}\t{}".format(os.path.relpath(file_path, args.root), frames))
            transcriptions.append(new_sentence)
            texts.append(" ".join(list(new_sentence.replace(" ", "|"))) + " |")
        
    print("[{}] Length ignore numbers : {}".format(data_name, limit_count))
    print("[{}] digit and alphabet ignore numbers : {}".format(data_name, remove_count))

    return errors, file_info, transcriptions, texts
    

def save_files(args, dataset, errors, file_info, transcriptions, texts):
    with open(os.path.join(args.dest, dataset + ".tsv"), "w") as tsv_out, open(
        os.path.join(args.dest, dataset + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.dest, dataset + ".wrd"), "w"
    ) as wrd_out:
        print(args.root, file=tsv_out)
        for tsv_item, wrd_item, ltr_item in zip(file_info, transcriptions, texts):
            print(tsv_item, file=tsv_out)
            print(wrd_item, file=wrd_out)
            print(ltr_item, file=ltr_out)

    if errors:
        with open(os.path.join(args.dest, "error.txt"), "a+") as error_f:
            for error_item in errors:
                print(error_item, file=error_f)
    return
    print("save files [{}]".format(dataset))
    return 


def save_dict(args, transcriptions):
    vocab_list = list()
    vocab_freq = list()

    for transcription in transcriptions:
        for grapheme in transcription:
            grapheme = " ".join(list(grapheme.replace(' ', '|').upper()))	
            if grapheme not in vocab_list:	
                vocab_list.append(grapheme)	
                vocab_freq.append(1)	
            else:	
                vocab_freq[vocab_list.index(grapheme)] += 1

    vocab_freq, vocab_list = zip(*sorted(zip(vocab_freq, vocab_list), reverse=True))
    with open(os.path.join(args.dest, 'dict.ltr.txt'), 'w') as write_f:
        for idx, (grpm, freq) in enumerate(zip(vocab_list, vocab_freq)):
            print("{} {}".format(grpm, freq), file=write_f)


def clean_up(dir_path):
    dataset = ['train', 'valid', 'test']
    ext_list = ['.tsv', '.wrd', '.ltr', '_error.txt']
    dict_path = os.path.join(dir_path, 'dict.ltr.txt')
    file_list = [dict_path]

    for file_head in dataset:
        for ext in ext_list:
            file_path = os.path.join(os.path.join(dir_path, file_head + ext))
            file_list.append(file_path)
    
    for fname in file_list:
        if os.path.exists(fname):
            os.remove(fname)
            print("[{}] already exists -> removed".format(fname))
    
    return
        

def split_dataset(args, json_list):
    rand = random.Random(args.seed)
    flag = np.array([rand.random() < args.test_percent for _ in range(len(json_list))])

    train_list = json_list[~flag]
    test_list = json_list[flag]

    return train_list, test_list


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    else:
        clean_up(args.dest)
    
    for folder in ['1.Training', '2.Validation']:
        assert os.path.isdir(os.path.join(args.root, folder)), "root 경로를 확인해주세요. [{}]".format(args.root)
        for dir in ['라벨링데이터', '원천데이터']:
            if dir not in os.listdir(os.path.join(args.root, folder)):
                assert os.path.isdir(folder), "'{}' 폴더를 찾을 수 없습니다. [{}]".format(dir, os.path.join(args.root, folder, dir))

    train_search_path=os.path.join(args.root, '1.Training', '라벨링데이터', "**/*.json")
    valid_search_path=os.path.join(args.root, '2.Validation', '라벨링데이터', "**/*.json")
    
    train_json_list = np.array(glob.glob(train_search_path, recursive=True))
    tmp_json_list = np.array(glob.glob(valid_search_path, recursive=True))
    valid_json_list, test_json_list = split_dataset(args, tmp_json_list)

    data_name = 'train' 
    train_errors, train_file_info, train_transcriptions, train_texts = load_script(args, data_name, train_json_list)
    save_files(args, data_name, train_errors, train_file_info, train_transcriptions, train_texts)
    save_dict(args, train_transcriptions)
    
    data_name = 'valid' 
    valid_errors, valid_file_info, valid_transcriptions, valid_texts = load_script(args, data_name, valid_json_list)
    save_files(args, data_name, valid_errors, valid_file_info, valid_transcriptions, valid_texts)

    data_name = 'test' 
    test_errors, test_file_info, test_transcriptions, test_texts = load_script(args, data_name, test_json_list)
    save_files(args, data_name, test_errors, test_file_info, test_transcriptions, test_texts)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)