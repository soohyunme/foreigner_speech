# Preprocess for foreigner speech data
# Dataurl : https://www.aihub.or.kr/aihubdata/data/view.do?&dataSetSn=505

# All related data must in same directory described below
# RootDir
#   |-------1.Training
#         |------라벨링데이터
#             |----------1. 베트남어
#             |----------2. 영어
#         |------원천데이터
#             |----------TS_1. 베트남어
#             |----------TS_2. 영어
#   |-------2.Validation
#         |------라벨링데이터
#             |----------1. 베트남어
#             |----------2. 영어
#         |------원천데이터
#             |----------VS_1. 베트남어
#             |----------VS_2. 영어

# Put your absolute 'RootDir' path below
ROOT_DIR='/code/gitRepo/data/aihub/Foreigner_speech' ## it is dummy, please modify it

# Put your absolute destination path
DESTINATION='/manifest/path' ## it is dummy, please modify it

## Run unzip code
python preprocess/make_manifest.py \
    "$ROOT_DIR" \
    --dest "$DESTINATION"