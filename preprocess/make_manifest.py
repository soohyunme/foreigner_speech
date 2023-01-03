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
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="WAV",
        help="if set, path must contain this substring for a file to be included in the manifest",
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

    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f:
        print(dir_path, file=train_f)

        dir_name=os.path.join('1.Training', '라벨링데이터')
        search_path=os.path.join(args.root, dir_name, "**/*.json")

        for fname in glob.iglob(search_path, recursive=True):
            parts = os.path.dirname(fname).split(os.sep)
            wav_dir = os.sep.join(dir if i !=len(parts) - 2 else 'TS_'+dir for i, dir in enumerate(parts)).replace('라벨링데이터', '원천데이터')

            with open(fname, 'r') as f:
                info_data = json.load(f)
                file_path = os.path.join(wav_dir, info_data['fileName'])
                
                try:
                    frames = soundfile.info(file_path).frames
                except:
                    with open(os.path.join(args.dest, "error.txt"), "a+") as error_f:
                        print(file_path, file=error_f)
                    continue
                
                print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=train_f
                )
    
    with open(os.path.join(args.dest, "valid.tsv"), "w") as valid_f:
        print(dir_path, file=valid_f)

        dir_name=os.path.join('2.Validation', '라벨링데이터')
        search_path=os.path.join(args.root, dir_name, "**/*.json")

        for fname in glob.iglob(search_path, recursive=True):
            parts = os.path.dirname(fname).split(os.sep)
            wav_dir = os.sep.join(dir if i !=len(parts) - 2 else 'VS_'+dir for i, dir in enumerate(parts)).replace('라벨링데이터', '원천데이터')

            with open(fname, 'r') as f:
                info_data = json.load(f)
                file_path = os.path.join(wav_dir, info_data['fileName'])
                
                try:
                    frames = soundfile.info(file_path).frames
                except:
                    with open(os.path.join(args.dest, "error.txt"), "a+") as error_f:
                        print(file_path, file=error_f)
                    continue
                
                print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=valid_f
                )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
