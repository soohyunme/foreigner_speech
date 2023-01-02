# Unzip for foreigner speech data
# Dataurl : https://www.aihub.or.kr/aihubdata/data/view.do?&dataSetSn=505

# All related data must in same directory described below
# RootDir
#   |-------1.Training
#         |------라벨링데이터
#             |----------1. 베트남어.zip
#             |----------2. 영어.zip
#         |------원천데이터
#             |----------TS_1. 베트남어.zip
#             |----------TS_2. 영어.zip
#   |-------2.Validation
#         |------라벨링데이터
#             |----------1. 베트남어.zip
#             |----------2. 영어.zip
#         |------원천데이터
#             |----------VS_1. 베트남어.zip
#             |----------VS_2. 영어.zip

# Put your absolute 'RootDir' path below
ROOT_DIR='/code/gitRepo/data/aihub/Foreigner_speech' ## it is dummy, please modify it

# Put your absolute destination path
DESTINATION='/path/to/unzip/data' ## it is dummy, please modify it

## Run unzip code
python preprocess/unzip.py \
    "$ROOT_DIR" \
    --dest "$DESTINATION"