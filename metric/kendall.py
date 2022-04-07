from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata

import hdf5storage
import json
import tables
import numpy as np
import pandas as pd

import os
import scipy.io as sio

tvsum_mat_path = "/root/clip/ydata-tvsum50.mat"
tvsum_pred_path = "/root/clip/video_datasets/pred.csv"


def load_tvsum_mat(filename):
    data = hdf5storage.loadmat(filename, variable_names = ['tvsum50'])

    # ravel(): flatten
    # tvsum50: ndarray (1, 50) -> (50, )
    data = data['tvsum50'].ravel()

    data_list = []

    for item in data:
        video, category, title, length, nframes, user_anno, gt_score = item

        item_dict = {
            'video': video[0, 0],
            'category': category[0, 0],
            'title': title[0, 0],
            'length': length[0, 0],
            'nframes': nframes[0, 0],
            'user_anno': user_anno,
            'gt_score': gt_score
        }

        data_list.append((item_dict))

    return data_list

def get_rc_func(metric):
    if metric == 'kendalltau':
        f = lambda x, y: kendalltau(rankdata(-x), rankdata(-y))
    elif metric == 'spearmanr':
        f = lambda x, y: spearmanr(x, y)
    else:
        raise RuntimeError

    return f


class RankCorrelationEvaluator(object):
    # __call__() : https://jinmay.github.io/2019/12/03/python/python-callable/
    def __call__(self):
        res = []

        for d in data:
            user_anno = d['user_anno'].T
            N = user_anno.shape[1]
            D, mean_tau, min_tau, max_tau = [], [], [], []

            pred_x = self.get_score(d['video'])
            D = [self.rc_func(x, pred_x)[0] for x in user_anno]

            res.append({'video': d['video'],
                        'mean': np.mean(D),
                        'min': np.min(D),
                        'max': np.max(D),
                        'cc': np.asarray(D),
                        })

        return res


class HumanEvaluator(RankCorrelationEvaluator):
    def __init__(self, metric):
        self.rc_func = get_rc_func(metric)

    def __call__(self):
        res = []

        for d in data:
            user_anno = d['user_anno'].T
            N = user_anno.shape[1]

            max_rc, min_rc, avr_rc, rc = [], [], [], []

            for i, x in enumerate(user_anno):
                # leave-one-out
                R = [self.rc_func(x, user_anno[j])[0] for j in range(len(user_anno)) if j != i]

                max_rc.append(max(R))
                min_rc.append(min(R))
                avr_rc.append(np.mean(R))
                rc += R

            res.append({'video': d['video'],
                        'mean': np.mean(avr_rc),
                        'min': np.mean(min_rc),
                        'max': np.mean(max_rc),
                        'cc': np.asarray(rc)
                        })

        return res


class TvSumEvaluator(RankCorrelationEvaluator):
    def __init__(self, metric):
        self.rc_func = get_rc_func(metric)

    def __call__(self):
        res = []
        tvsum_pred_df = pd.read_csv(tvsum_pred_path, sep='\t', names=["video_name", "pred_result"])

        for d in data:
            user_anno = d['user_anno'].T
            N = user_anno.shape[1]

            max_rc, min_rc, avr_rc, rc = [], [], [], []

            # for i, x in enumerate(user_anno): # 0 to 19
                # leave-one-out
                # So, len(R) = 20 - 1 = 19
                # self.rc_func(x, user_anno[j])[0]: correlation, no p-value
                # R = [self.rc_func(x, user_anno[j])[0] for j in range(len(user_anno)) if j != i]
            
            R = [self.rc_func(tvsum_pred_df['pred_result'], user_anno[idx])[0] for idx in range(len(user_anno))]

            max_rc.append(max(R))
            min_rc.append(min(R))
            avr_rc.append(np.mean(R))
            rc += R

            res.append({'video': d['video'],
                        'mean': np.mean(avr_rc),
                        'min': np.mean(min_rc),
                        'max': np.mean(max_rc),
                        'cc': np.asarray(rc)
                        })

        return res


class RandomEvaluator(RankCorrelationEvaluator):
    def __init__(self, metric):
        self.rc_func = get_rc_func(metric)

        rand_scores = {}

        for d in data:
            user_anno = d['user_anno'].T
            N = user_anno.shape[1]
            rand_scores[d['video']] = np.random.random((N,)) # random score

        self.rand_scores = rand_scores

    def get_score(self, v_id):
        return self.rand_scores[v_id]


# gt_score: "mean(user_anno)" per frame
data = load_tvsum_mat(tvsum_mat_path)
# tvsum

"""
(1) spearman
"""
metric = 'spearmanr'
# human_res = HumanEvaluator(metric)()
tvsum_res = TvSumEvaluator(metric)()
mean_arr = np.asarray([x['mean'] for x in tvsum_res])
print("human" + ": mean %.3f"%(np.mean(mean_arr)))

"""
(2) kendalltau
"""
metric = 'kendalltau'
human_res = HumanEvaluator(metric)()
mean_arr = np.asarray([x['mean'] for x in human_res])
print('human'+': mean %.3f'%(np.mean(mean_arr)))
