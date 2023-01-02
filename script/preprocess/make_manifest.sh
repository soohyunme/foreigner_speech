# Preprocess for foreigner speech data
# Dataurl : https://www.aihub.or.kr/aihubdata/data/view.do?&dataSetSn=505

# All related data must in same directory described below
# RootDir
#     |-------1.Training
#         |------라벨링데이터
#                 |----------1. 베트남어
#                         |----------1. 한국일반
#                         |----------2. 한국생활I
#         |------원천데이터
#                 |----------TS_1. 베트남어
#                         |----------1. 한국일반
#                         |----------2. 한국생활I
#     |-------2.Validation
#         |------라벨링데이터
#                 |----------1. 베트남어
#                         |----------1. 한국일반
#                         |----------2. 한국생활I
#         |------원천데이터
#                 |----------VS_1. 베트남어
#                         |----------1. 한국일반
#                         |----------2. 한국생활I

# Put your absolute 'RootDir' path below
ROOT_DIR='/code/gitRepo/data/aihub/Foreigner_speech' ## it is dummy, please modify it

# Put your absolute destination path
DESTINATION='/root/test' ## it is dummy, please modify it

## Run unzip code
python preprocess/make_manifest.py \
    "$ROOT_DIR" \
    --dest "$DESTINATION"