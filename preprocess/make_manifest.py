import argparse
import glob
import os
import sys
import random
import json
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
    parser.add_argument('--preprocess-mode', type=str, default='phonetic',
        help='Ex) (70%)/(칠 십 퍼센트) 확률이라니 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)/(백 프로)가 왜 안돼?'
                'phonetic: 칠 십 퍼센트 확률이라니 모 몬 소리야 진짜 백 프로가 왜 안돼?'
                'spelling: 70% 확률이라니 뭐 뭔 소리야 진짜 100%가 왜 안돼?')
    return parser


def save_dict(args, vocab_freq, vocab_list):
    vocab_freq, vocab_list = zip(*sorted(zip(vocab_freq, vocab_list), reverse=True))
    with open(os.path.join(args.dest, 'dict.ltr.txt'), 'w') as write_f:
        for idx, (grpm, freq) in enumerate(zip(vocab_list, vocab_freq)):
            print("{} {}".format(grpm, freq), file=write_f)


def make_manifest(args, dataset, dir_dict):
    with open(os.path.join(args.dest, dataset + ".tsv"), "w") as tsv_out, open(
        os.path.join(args.dest, dataset + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.dest, dataset + ".wrd"), "w"
    ) as wrd_out:

        print(args.root, file=tsv_out)

        dir_name=os.path.join(dir_dict['dataset_dir'][dataset], '라벨링데이터')
        search_path=os.path.join(args.root, dir_name, "**/*.json")

        vocab_list = list()
        vocab_freq = list()
        limit_count = 0

        for fname in glob.iglob(search_path, recursive=True):
            parts = os.path.dirname(fname).split(os.sep)
            wav_dir = os.sep.join(dir if i !=len(parts) - 2 else dir_dict['dataset_dir_tag'][dataset] + dir for i, dir in enumerate(parts)).replace('라벨링데이터', '원천데이터')

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

                try:
                    frames = soundfile.info(file_path).frames
                except:
                    with open(os.path.join(args.dest, "error.txt"), "a+") as error_f:
                        print(file_path, file=error_f)
                    continue

                print(
                "{}\t{}".format(os.path.relpath(file_path, args.root), frames), file=tsv_out
                )
                print(new_sentence, file=wrd_out)
                print(
                    " ".join(list(new_sentence.replace(" ", "|"))) + " |", file=ltr_out
                )
                if dataset != 'train':
                    continue

                for grapheme in new_sentence:	
                    grapheme = " ".join(list(grapheme.replace(' ', '|').upper()))	
                    if grapheme not in vocab_list:	
                        vocab_list.append(grapheme)	
                        vocab_freq.append(1)	
                    else:	
                        vocab_freq[vocab_list.index(grapheme)] += 1
        
        print("Ignore numbers : ", limit_count)

        if dataset == 'train':
            save_dict(args, vocab_freq, vocab_list)


def main(args):
    dir_dict = {
        'dataset_dir':{
            'train':'1.Training',
            'valid':'2.Validation'
        },
        'dataset_dir_tag':{
            'train':'TS_',
            'valid':'VS_'
        },
        'inner_dir':[
            '라벨링데이터', 
            '원천데이터'
        ]
    }

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
    
    for folder in dir_dict['dataset_dir'].values():
        assert os.path.isdir(os.path.join(args.root, folder)), "root 경로를 확인해주세요. [{}]".format(args.root)
        for dir in dir_dict['inner_dir']:
            if dir not in os.listdir(os.path.join(args.root, folder)):
                assert os.path.isdir(folder), "'{}' 폴더를 찾을 수 없습니다. [{}]".format(dir, os.path.join(args.root, folder, dir))

    if os.path.exists(os.path.join(args.dest, "error.txt")):
        os.remove(os.path.join(args.dest, "error.txt"))

    make_manifest(args, 'train', dir_dict)
    make_manifest(args, 'valid', dir_dict)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)