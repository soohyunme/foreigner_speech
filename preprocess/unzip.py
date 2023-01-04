import argparse
import glob
from zipfile import ZipFile
import shutil
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "src", metavar="DIR", help="src"
    )
    parser.add_argument(
        "--dest", default=None, type=str, metavar="DIR", help="dest directory"
    )
    parser.add_argument(
        "--ext", default="zip", type=str, metavar="EXT", help="extension to look for"
    )
    return parser


def unzip(src_path, dest_path):
    with ZipFile(src_path, 'r') as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            try:
                member.filename = member.filename.encode('cp437').decode('euc-kr', 'ignore')
                zf.extract(member, dest_path)
            except:
                raise Exception("unzip error '" + str(src_path) + "'")


def main(args):
    root = os.path.dirname(args.src) if args.dest is None else args.dest

    search_path = os.path.join(args.src, "**/*." + args.ext)
    assert len(list(glob.iglob(search_path, recursive=True))) != 0, "root 경로에서 압축파일을 찾을 수 없습니다. root_path : [{}]".format(args.src)

    for src_path in glob.iglob(search_path, recursive=True):
        parts = src_path.split(os.sep)
        dir = os.sep.join(parts[len(args.src.split(os.sep))-1:]).split('.'+args.ext)[0]
        dest_dir = os.path.join(root, dir)

        unzip(src_path, dest_dir)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
