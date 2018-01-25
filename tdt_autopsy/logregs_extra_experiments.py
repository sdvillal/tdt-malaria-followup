from __future__ import print_function, division

import os
import os.path as op
from itertools import product
import time
import traceback as tb

import numpy as np
import pandas as pd
import feather
import joblib
from scipy.sparse import isspmatrix_csr, isspmatrix_csc
from whatami import call_dict, What, whatable
from whatami.wrappers.what_sklearn import whatamise_sklearn
whatamise_sklearn()

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, Normalizer, MaxAbsScaler
from lightning.classification import CDClassifier

import tqdm
tqdm.tqdm.monitor_interval = 0

from ccl_malaria.features import MurmurFolder, zero_columns
from tdt_autopsy.eval import score_result, read_benchmark_labels
from tdt_autopsy.substructure_collision_analysis import X_train_feats, munge_rdk_hashes_df, rdkhash_feature_matrix

from tdt_autopsy.config import DATA_DIR
from minioscail.common.misc import ensure_dir


# --- In disk  location

RESULTS_DIR = op.join(DATA_DIR, '--unfolded-explorations', 'logreg-explorations')
RESULTS_CACHE_DIR = op.join(RESULTS_DIR, 'results')


# --- Models

class LOGREG_MODELS(object):

    # TODO: Should remove X, y, train asap, (move initialization based on dset to fit)
    #       quick and dirty stuff...

    # --- Models we based our TDT submission on (see logreg_analysis.logreg_experiments_to_deploy)
    #
    # Remember...
    # # Choose a few good results (maybe apply diversity filters or ensemble selection or...)
    # # These decisions where informed by some plotting (see tutorial)
    # # (e.g., keep number maneageble, keep most regularized amongst these with higher performance...)
    # deployment_cond_1 = df.query('cv_seed < 5 and '
    #                              'num_present_folds == num_cv_folds and '
    #                              'penalty == "l1" and '
    #                              'C == 1 and '
    #                              'class_weight == "auto" and '
    #                              'tol == 1E-4 and '
    #                              'folder_size < 1 and '
    #                              'folder_seed == -1 and '
    #                              'auc_mean > 0.92')
    #
    # deployment_cond_2 = df.query('num_present_folds == num_cv_folds and '
    #                              'penalty == "l2" and '
    #                              'C == 5 and '
    #                              'class_weight == "auto" and '
    #                              'tol == 1E-4 and '
    #                              'folder_size < 1 and '
    #                              'folder_seed == -1 and '
    #                              'auc_mean > 0.93')

    @staticmethod
    def tdt_l1(X=None, y=None, train=True):
        c = LogisticRegression(solver='liblinear',
                               penalty='l1',
                               C=1,
                               class_weight='balanced',
                               tol=0.0001,
                               fit_intercept=True,
                               intercept_scaling=1,
                               dual=False,
                               verbose=0,
                               random_state=0)
        if train:
            return c.fit(X, y)
        return c

    @staticmethod
    def tdt_l2(X=None, y=None, train=True):
        c = LogisticRegression(solver='liblinear',
                               penalty='l2',
                               C=5,
                               class_weight='balanced',
                               tol=0.0001,
                               fit_intercept=True,
                               intercept_scaling=1,
                               dual=False,
                               verbose=0,
                               random_state=0)
        if train:
            return c.fit(X, y)
        return c

    # --- Some stuff I do not want to forget to try at some time...

    @staticmethod
    def l1l2_cd(X, y, train=True):
        X = X.astype(np.float)
        c = CDClassifier(loss='modified_huber',
                         multiclass=False,
                         penalty='l1/l2',
                         alpha=1e-4,
                         C=1.0 / X.shape[0],
                         max_iter=20,
                         tol=0.0001,
                         verbose=1,
                         random_state=0)
        if train:
            return c.fit(X, y)
        return c

    @staticmethod
    def l2_cd(X, y):
        X = X.astype(np.float)
        return CDClassifier(loss='log',
                            multiclass=False,
                            penalty='l2',
                            alpha=1e-4,
                            C=5,
                            max_iter=20,
                            tol=0.0001,
                            verbose=1,
                            random_state=0).fit(X, y)

    @staticmethod
    def l2_saga(X, y):
        return LogisticRegression(solver='saga',
                                  penalty='l2',
                                  C=5,
                                  intercept_scaling=1,
                                  fit_intercept=True,
                                  tol=0.0001,
                                  class_weight='balanced',
                                  max_iter=1000,
                                  dual=False,
                                  verbose=1,
                                  random_state=0).fit(X, y)

    # sgd = SGDClassifier(loss='log', penalty='l2', fit_intercept=True, random_state=2)


# --- Custom preprocessors

def _fast_binarizer(X, ultra_fast=False, copy=False):
    if copy:
        X = X.copy()
    if not ultra_fast:
        X.eliminate_zeros()
    X.data = np.ones_like(X.data)
    return X


class FastBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, copy=False):
        self.copy = copy

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        if not isspmatrix_csc(X) and not isspmatrix_csr(X):
            raise ValueError('X must be csc or csr')
        return self

    # noinspection PyUnusedLocal
    def transform(self, X, y='deprecated', copy=None):
        if not isspmatrix_csc(X) and not isspmatrix_csr(X):
            raise ValueError('X must be csc or csr')
        if copy is None:
            copy = self.copy
        if copy:
            X = X.copy()
        X.data = np.ones_like(X.data)
        return X


@whatable
class Folder(MurmurFolder):
    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        if not isspmatrix_csc(X) and not isspmatrix_csr(X):
            raise ValueError('X must be csc or csr')
        return self

    # noinspection PyUnusedLocal
    def transform(self, X, y='deprecated', copy=None):
        if not isspmatrix_csc(X) and not isspmatrix_csr(X):
            raise ValueError('X must be csc or csr')
        return self.fold(X)


def pre_model(model,
              copy=True,
              binarize_threshold=None,
              binarize_fast=True,
              folder=None,
              scale=False,
              normalize=None):
    steps = []
    # Make fingerprint binary?
    if binarize_threshold is not None:
        if binarize_fast and binarize_threshold == 0:
            steps.append(('binarize', FastBinarizer(copy=copy)))
        else:
            steps.append(('binarize', Binarizer(threshold=binarize_threshold, copy=copy)))
    # Fold?
    if folder is not None:
        steps.append(('fold', folder))
    # Normalize column-wise
    if scale:
        steps.append(('scale', MaxAbsScaler(copy=copy)))
    # Normalize row-wise
    if normalize is not None:
        steps.append(('normalize', Normalizer(norm=normalize, copy=copy)))
    # The model
    steps.append(('model', model))
    return Pipeline(steps)


# --- Experiments

def print_matrix_summary(X, y=None, name='train matrix'):
    X.eliminate_zeros()  # Probably a noop
    print('%s: %d entries; %d zeros; %d == 1; %d > 1' %
          (name,
           X.shape[0] * X.shape[1],
           X.shape[0] * X.shape[1] - X.nnz,
           (X == 1).sum(),
           (X > 1).sum()))
    if y is not None:
        print('%d positives, %d negatives' % (y.sum(), len(y) - y.sum()))


def logreg_stats(logreg, small=1E-9):
    sparsity = 0
    for coef in logreg.coef_:
        sparsity += (np.abs(coef) < small).sum() / len(coef)
    sparsity /= len(logreg.coef_)

    return {
        'model_sparsity':  sparsity
    }


def run_one_exp(Xlab, y_lab,
                Xunl, y_unl, i2m_unl,
                hdf,
                model_name, model,
                data_name='competition-external',
                recompute=False, cache_dir=RESULTS_CACHE_DIR,
                model_stats=None,
                save_model=True,
                save_predictions=False,
                min_radius=None, max_radius=None,
                binarize_threshold=0,
                folder=None, allow_unseen_in_folding=False,
                scale=False,
                row_normalizer=None):

    # With our design, this is a given.
    # All matrices have at least the first columns as given by the labelled data.
    assert Xunl.shape[1] >= Xlab.shape[1]

    # Model configuration
    model = model(Xlab, y_lab, train=False)
    model = pre_model(model=model,
                      binarize_threshold=binarize_threshold,
                      folder=folder,
                      scale=scale,
                      normalize=row_normalizer)
    result = call_dict(ignores=('Xlab', 'y_lab',
                                'Xunl', 'y_unl', 'i2m_unl',
                                'hdf',
                                'recompute', 'cache_dir',
                                'model_stats', 'save_model', 'save_predictions'))
    what = What(name='logreg_result', conf=result)
    # FIXME: establish what are real keys...
    result['id'] = what.id(maxlength=1)

    cache_path = op.join(ensure_dir(cache_dir, result['id'][:2], result['id']), 'result.pkl')
    if not recompute:
        try:
            result = pd.read_pickle(cache_path)
            if 'fail' not in result:
                print('Loaded result from %s' % cache_path)
                return result
        except IOError:
            pass

    # No real need, but it is fast with our data and better safe...
    Xlab = Xlab.copy()
    Xunl = Xunl.copy()

    print('Experiment:', what.id())
    start_total = time.time()

    try:

        # --- Pre-column manipulations
        print('Pre-columns manipulation...')

        columns_start = time.time()

        # Apply radius constraints (as usual, just make columns constant 0)
        if max_radius is not None:
            columns_out = hdf.query('radius > %d' % max_radius).column.values
            Xlab = zero_columns(Xlab, columns_out)
            Xunl = zero_columns(Xunl, columns_out)
        if min_radius is not None:
            columns_out = hdf.query('radius < %d' % min_radius).column.values
            Xlab = zero_columns(Xlab, columns_out)
            Xunl = zero_columns(Xunl, columns_out)

        # Remove unseen features from unl if we are not gonna use them
        if not allow_unseen_in_folding or folder is None:
            Xunl = Xunl.tocsc()[:, :Xlab.shape[1]].tocsr()

        result['pre_column_manipulation_s'] = time.time() - columns_start

        print('Fitting...')

        # --- Fit
        start_fit = time.time()
        if not save_model:
            model = clone(model)
        model = model.fit(Xlab, y_lab)
        if save_model:
            result['model'] = model
        if model_stats is not None:
            result.update(model_stats(model.named_steps.model))
        result['fit_s'] = time.time() - start_fit
        print('Train took %.2fs' % result['fit_s'])

        # --- Predict
        print('Predicting...')
        start_eval = time.time()

        scores = pd.DataFrame(model.predict_proba(Xunl)[:, 1],
                              columns=['score'],
                              index=i2m_unl)
        if save_predictions:
            result['predictions'] = scores
        result['predict_s'] = time.time() - start_eval
        print('Predict took %.2fs' % result['predict_s'])

        # --- Evaluate
        eval_start = time.time()
        evaluation = score_result(scores, y_unl, model=model_name, dataset=data_name)
        del evaluation['model']
        del evaluation['dataset']
        result.update(evaluation)
        result['eval_s'] = time.time() - eval_start
        print('Eval took %.2fs' % result['eval_s'])
    except Exception as ex:
        tb.print_exc()
        result['fail'] = ex
    finally:
        # Add total time
        result['total_s'] = time.time() - start_total
        print('Total took %.2fs' % result['total_s'])
        # Save result
        pd.to_pickle(result, cache_path)
        pd.read_pickle(cache_path)  # Ensure we can read it back
        print('Saved result to %r' % cache_path)
        # Return
        return result


def compute_results(recompute=False, num_jobs=4, fold=False, l1too=False):
    print('Loading data...')
    # The competition benchmark (N.B. also features not in train)
    i2m_unl, i2f_unl, Xunl, _ = rdkhash_feature_matrix(fpt='ecfp', dset='unl')
    y_unl = read_benchmark_labels()
    # The training data
    i2m_lab, i2f_lab, f2i_lab, Xlab, y_lab = X_train_feats(fpt='ecfp', dsets='lab',
                                                           use_representatives=False, transductive=False)
    # Information about the features
    hdf = munge_rdk_hashes_df(fpt='ecfp', nthreads=4, recompute=False, columns=['column', 'radius'])
    hdf = hdf.query('column < %d' % max(Xunl.shape[1], Xlab.shape[1]))

    # Show some info about our datasets
    print_matrix_summary(Xlab, y_lab, 'train')
    print_matrix_summary(Xunl, y_unl['active'], 'test')

    # Experimental variants...

    models = [
        ('tdtl2', LOGREG_MODELS.tdt_l2, logreg_stats),
    ]
    if l1too:
        models += [
            # Slow and not so competitive
            ('tdtl1', LOGREG_MODELS.tdt_l1, logreg_stats),
        ]

    max_radii = [None, 2, 3, 4]  # 6, 8, 10
    min_radii = [None]  # But this could also be of interest

    # None => use counts.
    # TODO: we need to implement folding for counts (anyway most counts are == 1)
    # If curious in a hurry, we could just compute everythonn that uses counts and no folding
    binarizers = [0, None] if not fold else [0]

    #
    # TODO: implement folding for counts
    # TODO: implement multiple foldings to create a mask
    #   - Several seeds, let collisions happen freely
    #   - Fix number of bits set, try to hash until bit pattern has a desired cardinality
    #   - Or for colliding if binary, sum if counts
    # TODO: do not save the map, but still recompute the collision rate
    # (find collumns that collide and their distribution in the matris, i.e.
    # sum of values of each cell they collide, how many molecules they are in...)
    #
    if fold:
        folders = []
        for seed in range(4):
            folders += [
                Folder(seed=seed, fold_size=511, save_map=False),
                Folder(seed=seed, fold_size=1023, save_map=False),
                Folder(seed=seed, fold_size=2047, save_map=False),
                Folder(seed=seed, fold_size=4091, save_map=False),
                Folder(seed=seed, fold_size=8191, save_map=False),
                Folder(seed=seed, fold_size=16383, save_map=False),
                Folder(seed=seed, fold_size=32767, save_map=False),
            ]
        unseen_in_foldings = True, False
    else:
        folders = [None]
        unseen_in_foldings = [None]

    row_normalizers = 'l1', 'l2', None

    experiments = list(product(
        models,
        min_radii,
        max_radii,
        binarizers,
        folders,
        unseen_in_foldings,
        row_normalizers
    ))

    joblib.Parallel(n_jobs=num_jobs, batch_size=1, pre_dispatch=num_jobs)(
        joblib.delayed(run_one_exp)(
            Xlab, y_lab,
            Xunl, y_unl, i2m_unl,
            hdf,
            model_name=model_name, model=model,
            model_stats=model_stats,
            save_model=False, save_predictions=False,
            data_name='competition-external',
            min_radius=min_radius, max_radius=max_radius,
            binarize_threshold=binarizer,
            folder=folder, allow_unseen_in_folding=unseen_in_folding,
            scale=False,
            row_normalizer=row_normalizer,
            recompute=recompute,
        )
        for ((model_name, model, model_stats),
             min_radius, max_radius,
             binarizer,
             folder, unseen_in_folding,
             row_normalizer) in tqdm.tqdm(experiments, unit='experiment')
    )


def compute_at_home():
    import socket
    host = socket.gethostname()
    if host == 'snowy':
        compute_results(recompute=False, num_jobs=4, fold=False, l1too=False)
        compute_results(recompute=False, num_jobs=4, fold=True, l1too=False)
    elif host == 'mumbler':
        compute_results(recompute=False, num_jobs=8, fold=True, l1too=False)
        compute_results(recompute=False, num_jobs=8, fold=False, l1too=False)
        compute_results(recompute=False, num_jobs=8, fold=True, l1too=True)
        compute_results(recompute=False, num_jobs=8, fold=False, l1too=True)


# --- Experiment analysis

def load_all_results(path=RESULTS_CACHE_DIR, remove_objects=False):
    # Load all these nasty quick and dirty pickles
    results = []
    for dirpath, dirnames, filenames in os.walk(path):
        if 'result.pkl' in filenames:
            results.append(pd.read_pickle(op.join(dirpath, 'result.pkl')))

    # Dataframeit!
    df = pd.DataFrame(results)

    # Extract parameters from nested descriptions (folder, pipeline)
    # TODO: whatid2columns needs some love @ whatami
    # df = whatid2columns(df, 'folder', columns=['fold_size'], prefix='folder_')
    df['fold_size'] = df['folder'].apply(lambda x: None if x is None else x.fold_size)
    df['fold_seed'] = df['folder'].apply(lambda x: None if x is None else x.seed)

    # Reorganize to have tidy columns
    column_order = [
        # Experimental variables
        'data_name',
        'model_name',
        'binarize_threshold',
        'folder',
        'fold_size',
        'allow_unseen_in_folding',
        'max_radius',
        'min_radius',
        'row_normalizer',
        'scale',
        # Model stats
        'sparsity',
        # Performance in the competition challenge dataset
        'auc',
        'enrichment_1',
        'enrichment_5',
        'enrichment_10',
        'bedroc20',
        'rie20',
        # Times
        'pre_column_manipulation_s',
        'fit_s',
        'predict_s',
        'eval_s',
        'total_s',
        'folder',
        # Experiment ID
        'id',
        # Model details
        'model',
        # Failures
        'fail',
    ]
    columns = [column for column in column_order if column in df.columns]
    columns += [column for column in df.columns if column not in columns]
    df = df[columns]

    # Do not keep objects
    if remove_objects:
        del df['model']
        del df['folder']

    return df


def tidy_results(recompute=False):

    cache_path = op.join(RESULTS_DIR, 'results.tidy.feather')

    df = None
    if not recompute:
        try:
            df = feather.read_dataframe(cache_path)
        except IOError:
            pass
    if df is None:
        df = load_all_results(remove_objects=True)
        feather.write_dataframe(df, cache_path)

    # Make missings sendible for grouping

    def fillna(df, column, value):
        df[column] = df[column].fillna(value=value)

    fillna(df, 'row_normalizer', 'none')
    fillna(df, 'fold_size', np.inf)
    fillna(df, 'allow_unseen_in_folding', False)
    fillna(df, 'binarize_threshold', np.inf)
    fillna(df, 'max_radius', np.inf)
    fillna(df, 'min_radius', -np.inf)
    df['fold_size'] = df['fold_size'].apply(lambda x: int(x) if np.isfinite(x) else 2**32-1)
    df['is_counts'] = df['binarize_threshold'].apply(lambda x: x != 0)

    return df


if __name__ == '__main__':
    compute_at_home()
    tidy_results(recompute=True).info()
    print('Open logreg_competition_benchmark_explorations.ipynb')

#
# Other things to try:
#   - Folding counts
#   - Removal of duplicates by selecting representative (it should just be a couple of lines)
#   - Differences of weights and performance with representatives on/off
#   - ... (sdd logreg_competition_benchmark_explorations.ipynb)
# Other important things to report:
#   - Check for actual collisions (better use the big SMARTS <-> rdkhash tables prepared)
#   - A proper dataset (My malaria export from chembl is already prepared)
#
