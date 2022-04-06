####################################################
# RMSE(roor mean squared error) 채점 코드 예시입니다. 3-1. RMSE 예시
####################################################
import os
import sys
import pandas as pd
import numpy as np
###############################################################################
# 편집할 구간: 채점에 사용할 함수 정의
# rmse: 점수를 계산할 함수입니다. 정답(y_true), 예측(y_pred)을 인자로 입력받아 score를 반환합니다.
def rmse(y_true, y_pred):
return np.sqrt(((y_pred - y_true) ** 2).mean())
###############################################################################
# 해당 코드가 전달받는 인자들의 목록입니다.
# sys.argv[1]: 업로드하신 정답파일의 경로입니다. 자동으로 매겨지는 부분으로 그대로 놔두시면 됩니다.
# sys.argv[2]: 제출된 결과 파일의 경로를 받는 인자입니다. 점수를 계산하는 함수에서 y_pred 자리에 들어갈 값입니다.
##############################################################################
# 편집할 구간: 정답 및 제출 파일 형식에 맞는 불러오기 코드
# 값 불러오기
pred = pd.read_csv(sys.argv[2]).to_numpy()[:,1:] # 참가자가 제출한 결과 파일을 .csv 테이블 형태로 불러옵니다.
gt = pd.read_csv(sys.argv[1]).to_numpy()[:,1:] # 정답파일을 .csv 테이블 형태로 불러옵니다.
# 제출 결과 및 정답 파일 경로는 자동으로 설정되므로 직접
입력하지 말고 비워두시면 됩니다.
# 스코어 산출
score = rmse(gt, pred) # 위에서 정의한 함수에 불러온 정답 테이블 및 결과 테이블을 입력하여 스코어를 계산합니다.
print(f"score:{score}")
##############################################################################



import os
import sys
import pandas as pd
###############################################################################
# 편집할 구간: 채점에 사용할 함수 정의
# 정확도 계산에 사용할 함수로 sklearn의 accuracy_score를 불러옵니다.
from sklearn.metrics import accuracy_score
###############################################################################
# 해당 코드가 전달받는 인자들의 목록입니다.
# sys.argv[1]: 업로드하신 정답파일의 경로입니다. 자동으로 매겨지는 부분으로 그대로 놔두시면 됩니다.
# sys.argv[2]: 제출된 결과 파일의 경로를 받는 인자입니다. 점수를 계산하는 함수에서 y_pred 자리에 들어갈 값입니다.
##############################################################################
# 편집할 구간: 정답 및 제출 파일 형식에 맞는 불러오기 코드
# 값 불러오기
pred = pd.read_csv(sys.argv[2]).to_numpy()[:,1:].astype(int).reshape(-1, 1) # 참가자가 제출한 결과 파일을 .csv
테이블 형태로 불러옵니다.
gt = pd.read_csv(sys.argv[1]).to_numpy()[:,1:].astype(int).reshape(-1, 1) # 정답파일을 .csv 테이블 형태로
불러옵니다.
# 제출 결과 및 정답 파일 경로는
자동으로 설정되므로 직접 입력하지 말고 비워두시면 됩니다.
# 스코어 산출
score = accuracy_score(gt, pred) # 위에서 가져온 함수에 불러온 정답 테이블 및 결과 테이블을 입력하여 스코어를
계산합니다.
print(f"score:{score}")
##############################################################################
