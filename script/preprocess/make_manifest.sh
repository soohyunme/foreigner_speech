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

# if length of script is over limit, it will be excluded
LIMIT=200

# Select transcription type
# Ex) (70%)/(칠 십 퍼센트) 확률이라니 (뭐 뭔)/(모 몬) 소리야 진짜 (100%)/(백 프로)가 왜 안돼?
# phonetic: 칠 십 퍼센트 확률이라니 모 몬 소리야 진짜 백 프로가 왜 안돼
# spelling: 70% 확률이라니 뭐 뭔 소리야 진짜 100%가 왜 안돼
PROCESS_MODE=phonetic
# PROCESS_MODE=spelling

# percentage of data to use as test set (between 0 and 1)
TEST_RATIO=0.05

# Run unzip code
python preprocess/make_manifest.py \
    "$ROOT_DIR" \
    --dest "$DESTINATION" \
    --preprocess-mode $PROCESS_MODE \
    --token-limit $LIMIT \
    --test-percent $TEST_RATIO