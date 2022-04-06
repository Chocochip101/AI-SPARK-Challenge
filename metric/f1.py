####################################################
# F1 Score 
####################################################
import os
import sys
import pandas as pd
import numpy as np
import math
###############################################################################
# 편집할 구간: 채점에 사용할 함수 정의
# rmse: 점수를 계산할 함수입니다. 정답(y_true), 예측(y_pred)을 인자로 입력받아 score를 반환합니다.
def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """Compare machine summary with user summary (keyshot-based).
    Args:
    --------------------------------
    machine_summary and user_summary should be binary vectors of ndarray type.
    eval_metric = {'avg', 'max'}
    'avg' averages results of comparing multiple human summaries.
    'max' takes the maximum (best) out of multiple comparisons.
    """
    machine_summary = machine_summary.astype(np.float32)
    user_summary = user_summary.astype(np.float32)
    n_users,n_frames = user_summary.shape

    # binarization
    machine_summary[machine_summary > 0] = 1
    user_summary[user_summary > 0] = 1

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        zero_padding = np.zeros((n_frames - len(machine_summary)))
        machine_summary = np.concatenate([machine_summary, zero_padding])
    f_scores = []
    prec_arr = []
    rec_arr = []
    for user_idx in range(n_users):
        gt_summary = user_summary[user_idx,:]
        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        f_scores.append(f_score)
        prec_arr.append(precision)
        rec_arr.append(recall)

    if eval_metric == 'avg':
        final_f_score = np.mean(f_scores)
        final_prec = np.mean(prec_arr)
        final_rec = np.mean(rec_arr)
    elif eval_metric == 'max':
        final_f_score = np.max(f_scores)
        max_idx = np.argmax(f_scores)
        final_prec = prec_arr[max_idx]
        final_rec = rec_arr[max_idx]
    return final_f_score, final_prec, final_rec
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
