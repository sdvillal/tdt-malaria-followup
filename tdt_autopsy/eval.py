# coding=utf-8
"""Further evaluation of models on different datasets."""
from __future__ import print_function
from future.utils import PY3

import os.path as op

from functools import partial

import numpy as np
import pandas as pd

if PY3:
    # Wrong import in Scoring
    # This will be fixed in rdkit 2017.3
    import sys
    sys.modules['exceptions'] = 'FAKE NEWS'
from rdkit.ML.Scoring import Scoring

from tdt_autopsy.config import (COMPETITION_GT_CSV,
                                CCL_EXPERIMENTS_DIR,
                                CCL_LOGREGS_EXPERIMENTS_DIR,
                                CCL_TREES_EXPERIMENTS_DIR, SEREINA_MODERN_DIR)


def read_benchmark_labels():
    """Returns a pandas dataframe with the labels from the competition dataset."""
    # Find full version + our take in tdt private
    # N.B. score is MAR3D7_pEC50
    unl_gt = pd.read_csv(COMPETITION_GT_CSV,
                         names=('id', 'score', 'smiles', 'active'),
                         index_col=False)
    unl_gt = unl_gt[['id', 'smiles', 'score', 'active']]
    return unl_gt.sort_values('id').set_index('id', drop=False)


def _read_ccl_scores(path):
    if not op.isfile(path):
        raise Exception('Cannot find the submission file %r' % path)
    scores = pd.read_csv(path, names=('id', 'smiles', 'score'), index_col=False)
    return scores.sort_values('id').set_index('id', drop=False)


def read_ccl_final_scores(screening=False,
                          calibrated=False,
                          average_folds=True,
                          linr_stacker=True):
    fn = 'final-{calibrated}-{result_agg}-{result_stacker}-{dset}.csv'.format(
        calibrated='calibrated' if calibrated else 'nonCalibrated',
        result_agg='averageFolds' if average_folds else 'lastFold',
        result_stacker='avg' if not linr_stacker else 'stacker=linr',
        dset='scr' if screening else 'unl'
    )
    return _read_ccl_scores(op.join(CCL_EXPERIMENTS_DIR, fn))

CCL_FINAL_EXTERNAL_SCORES = {
    'ccl_final_avg_linr': partial(read_ccl_final_scores, average_folds=True, linr_stacker=True),
    'ccl_final_avg_avg': partial(read_ccl_final_scores, average_folds=True, linr_stacker=False),
    'ccl_final_lastFold_linr': partial(read_ccl_final_scores, average_folds=False, linr_stacker=True),
    'ccl_final_lastFold_avg': partial(read_ccl_final_scores, average_folds=False, linr_stacker=False),
}


def read_ccl_logregs_scores(screening=False,
                            average_folds=True,
                            linr_stacker=True):
    # should bring back calibration before aggregation
    fn = 'logreg-{result_agg}_{dset}-{result_stacker}.txt'.format(
        result_agg='folds-average' if average_folds else 'only-last-fold',
        dset='scr' if screening else 'unl',
        result_stacker='stacker=linr' if linr_stacker else 'averaged'
    )
    return _read_ccl_scores(op.join(CCL_LOGREGS_EXPERIMENTS_DIR, fn))


CCL_LOGREG_EXTERNAL_SCORES = {
    'ccl_logregs_avg_linr': partial(read_ccl_logregs_scores, average_folds=True, linr_stacker=True),
    'ccl_logregs_avg_avg': partial(read_ccl_logregs_scores, average_folds=True, linr_stacker=False),
    'ccl_logregs_lastFold_linr': partial(read_ccl_logregs_scores, average_folds=False, linr_stacker=True),
    'ccl_logregs_lastFold_avg': partial(read_ccl_logregs_scores, average_folds=False, linr_stacker=False),
}


def read_ccl_trees_scores(screening=False, linr_stacker=True):
    fn = 'trees_{dset}-{result_stacker}.txt'.format(
        dset='scr' if screening else 'unl',
        result_stacker='stacker=linr' if linr_stacker else 'averaged'
    )
    return _read_ccl_scores(op.join(CCL_TREES_EXPERIMENTS_DIR, fn))


CCL_TREES_EXTERNAL_SCORES = {
    'ccl_trees_linr': partial(read_ccl_trees_scores, linr_stacker=True),
    'ccl_trees_avg': partial(read_ccl_trees_scores, linr_stacker=False),
}


def read_sg_scores(screening=False, modern=True):
    fn = ('rank_ordered_list_external_testset.txt' if not screening else
          'rank_ordered_list_1K_commercial_compounds.txt')
    path = op.join(SEREINA_MODERN_DIR,
                   'results',
                   'modern' if modern else 'old',
                   fn)
    df = pd.read_csv(path, sep='\t', index_col=False)
    df = df.rename(columns={'#SAMPLE': 'id',
                            'SMILES': 'smiles',
                            'Max_Probability': 'score',
                            'Max_Rank': 'max_rank'})
    df = df[['id', 'smiles', 'score', 'max_rank']]
    return df.sort_values('id').set_index('id', drop=False)


def read_sg_moderner_scores(f4096=True):
    fn = ('ranked_list_test_cmps-fpsize=1024-2048.dat.gz' if not f4096 else
          'ranked_list_test_cmps-fpsize=4096.dat.gz')
    path = op.join(SEREINA_MODERN_DIR, 'results', 'moderner', fn)
    df = pd.read_csv(path, sep='\t', index_col=False)
    df = df.rename(columns={'#Identifier': 'id',
                            'SMILES': 'smiles',
                            'Max_Proba': 'score',
                            'Max_Rank': 'max_rank',
                            'Similarity': 'similarity'})
    df = df[['id', 'smiles', 'score', 'max_rank', 'similarity']]
    return df.sort_values('id').set_index('id', drop=False)


SG_EXTERNAL_SCORES = {
    'sg_old': partial(read_sg_scores, modern=False),
    'sg_modern': partial(read_sg_scores, modern=True),
    'sg_moderner': partial(read_sg_moderner_scores, f4096=False),
    'sg_moderner_4096': partial(read_sg_moderner_scores, f4096=True),
}


def score_result(scores=read_sg_moderner_scores,
                 actual=read_benchmark_labels,
                 dataset='competition-external',
                 model='sg-moderner-4096'):
    # Strong contract of proper ordering...
    # Maybe we should merge by index instead
    if callable(actual):
        actual = actual()
    actual = actual.active.values
    if callable(scores):
        scores = scores()
    scores = scores.score.values

    # For rdkit Scoring API
    actual_sorted = actual[np.argsort(scores)[::-1]].reshape(-1, 1)

    return dict(
        dataset=dataset,
        model=model,
        auc=Scoring.CalcAUC(actual_sorted, 0),
        enrichment_1=Scoring.CalcEnrichment(actual_sorted, 0, (0.01,))[0],
        enrichment_5=Scoring.CalcEnrichment(actual_sorted, 0, (0.05,))[0],
        enrichment_10=Scoring.CalcEnrichment(actual_sorted, 0, (0.1,))[0],
        rie20=Scoring.CalcRIE(actual_sorted, 0, 20),
        bedroc20=Scoring.CalcBEDROC(actual_sorted, 0, 20),
    )


ALL_EXTERNAL_SCORES = dict(**CCL_FINAL_EXTERNAL_SCORES)
ALL_EXTERNAL_SCORES.update(CCL_LOGREG_EXTERNAL_SCORES)
ALL_EXTERNAL_SCORES.update(CCL_TREES_EXTERNAL_SCORES)
ALL_EXTERNAL_SCORES.update(SG_EXTERNAL_SCORES)


if __name__ == '__main__':
    external_gt = read_benchmark_labels()
    results = [score_result(scores=scorer, actual=external_gt,
                            dataset='competition_benchmark',
                            model=model)
               for model, scorer in ALL_EXTERNAL_SCORES.items()]
    results_df = pd.DataFrame(results)
    order = ['model', 'dataset', 'auc',
             'enrichment_1', 'enrichment_5', 'enrichment_10',
             'bedroc20', 'rie20']
    results_df = results_df[order].sort_values(['dataset', 'model']).reset_index(drop=True)

    pd.set_option('precision', 2)
    results_df.sort_values('auc', ascending=False).to_html('results-external.html', index=False)

    # TODO: rebuild also CCL trees / logregs, put a moderner version there
    # TODO: do not fail if a result is missing (catch, continue)
