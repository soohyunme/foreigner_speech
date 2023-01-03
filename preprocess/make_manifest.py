import argparse
import glob
import os
import random
import json
import soundfile


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
    return parser


def main(args):
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    for folder in ['1.Training','2.Validation']:
        assert os.path.isdir(os.path.join(args.root, folder)), "root 경로를 확인해주세요. [{}]".format(args.root)
        for dir in ['라벨링데이터', '원천데이터']:
            if dir not in os.listdir(os.path.join(args.root, folder)):
                assert os.path.isdir(folder), "'{}' 폴더를 찾을 수 없습니다. [{}]".format(dir, os.path.join(args.root, folder, dir))

    if os.path.exists(os.path.join(args.dest, "error.txt")):
        os.remove(os.path.join(args.dest, "error.txt"))

    dir_path = os.path.realpath(args.root)

    dataset = 'train'
    with open(os.path.join(args.dest, dataset + ".tsv"), "w") as tsv_out, open(
        os.path.join(args.dest, dataset + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.dest, dataset + ".wrd"), "w"
    ) as wrd_out:

        print(dir_path, file=tsv_out)

        dir_name=os.path.join('1.Training', '라벨링데이터')
        search_path=os.path.join(args.root, dir_name, "**/*.json")

        vocab_list = list()
        vocab_freq = list()

        for fname in glob.iglob(search_path, recursive=True):
            parts = os.path.dirname(fname).split(os.sep)
            wav_dir = os.sep.join(dir if i !=len(parts) - 2 else 'TS_'+dir for i, dir in enumerate(parts)).replace('라벨링데이터', '원천데이터')

            with open(fname, 'r') as f:
                info_data = json.load(f)
                file_path = os.path.join(wav_dir, info_data['fileName'])
                transcription = info_data['transcription']['ReadingLabelText'] if \
                    info_data['transcription']['ReadingLabelText'] != '' else info_data['transcription']['AnswerLabelText']

                try:
                    frames = soundfile.info(file_path).frames
                except:
                    with open(os.path.join(args.dest, "error.txt"), "a+") as error_f:
                        print(file_path, file=error_f)
                    continue

                print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=tsv_out
                )
                print(transcription, file=wrd_out)
                print(
                    " ".join(list(transcription.replace(" ", "|"))) + " |", file=ltr_out
                )

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

    dataset = 'valid'
    with open(os.path.join(args.dest, dataset + ".tsv"), "w") as tsv_out, open(
        os.path.join(args.dest, dataset + ".ltr"), "w"
    ) as ltr_out, open(
        os.path.join(args.dest, dataset + ".wrd"), "w"
    ) as wrd_out:

        print(dir_path, file=tsv_out)

        dir_name=os.path.join('2.Validation', '라벨링데이터')
        search_path=os.path.join(args.root, dir_name, "**/*.json")

        for fname in glob.iglob(search_path, recursive=True):
            parts = os.path.dirname(fname).split(os.sep)
            wav_dir = os.sep.join(dir if i !=len(parts) - 2 else 'VS_'+dir for i, dir in enumerate(parts)).replace('라벨링데이터', '원천데이터')

            with open(fname, 'r') as f:
                info_data = json.load(f)
                file_path = os.path.join(wav_dir, info_data['fileName'])
                transcription = info_data['transcription']['ReadingLabelText'] if \
                    info_data['transcription']['ReadingLabelText'] != '' else info_data['transcription']['AnswerLabelText']

                try:
                    frames = soundfile.info(file_path).frames
                except:
                    with open(os.path.join(args.dest, "error.txt"), "a+") as error_f:
                        print(file_path, file=error_f)
                    continue

                print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=tsv_out
                )
                print(transcription, file=wrd_out)
                print(
                    " ".join(list(transcription.replace(" ", "|"))) + " |", file=ltr_out
                )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)