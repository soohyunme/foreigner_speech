import argparse
import glob
import os
import shutil
from zipfile import ZipFile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", metavar="DIR", help="src")
    parser.add_argument(
        "--dest", default=None, type=str, metavar="DIR", help="dest directory"
    )
    parser.add_argument(
        "--ext", default="zip", type=str, metavar="EXT", help="extension to look for"
    )
    return parser


def unzip(src_path, dest_path):
    with ZipFile(src_path, "r") as zf:
        zipInfo = zf.infolist()
        for member in zipInfo:
            try:
                member.filename = member.filename.encode("cp437").decode(
                    "euc-kr", "ignore"
                )
                zf.extract(member, dest_path)
            except:
                raise Exception("unzip error '" + str(src_path) + "'")


def convert_path(args, path):
    fdir, fname = os.path.split(path)
    fname = os.path.splitext(fname)[0]
    parts = fdir.split(os.sep)[-3:]

    if "add" in parts[-1]:
        parts[-1] = parts[-1].split("_")[0]
        fname = fname.split("_add")[0]

    fdir = os.sep.join(parts)

    return os.path.join(args.dest, fdir, fname)


def main(args):
    search_path = os.path.join(args.src, "**/*." + args.ext)
    assert (
        len(list(glob.iglob(search_path, recursive=True))) != 0
    ), f"root 경로에서 압축파일을 찾을 수 없습니다. root_path : [{args.src}]"

    for src_path in glob.iglob(search_path, recursive=True):
        dest_dir = convert_path(args, src_path)
        unzip(src_path, dest_dir)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
