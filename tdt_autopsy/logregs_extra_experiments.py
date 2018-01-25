from __future__ import print_function, division

import os
import os.path as op
import time
import traceback as tb
from itertools import product

import feather
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from toolz import isiterable

from whatami import call_dict, What, whatable, what2id
from whatami.wrappers.what_sklearn import whatamise_sklearn
whatamise_sklearn()

from sklearn.base import TransformerMixin, BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, Normalizer, MaxAbsScaler
from lightning.classification import CDClassifier

import tqdm
tqdm.tqdm.monitor_interval = 0

from ccl_malaria.features import MurmurFolder, zero_columns as make_columns_zero
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
        if not issparse(X):
            raise ValueError('X must be sparse (dense implementation coming soon, or not)')
        return self

    # noinspection PyUnusedLocal
    def transform(self, X, y='deprecated', copy=None):
        if not issparse(X):
            raise ValueError('X must be sparse (dense implementation coming soon, or not)')
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
        if not issparse(X):
            raise ValueError('X must be sparse (dense implementation coming soon, or not)')
        return self

    # noinspection PyUnusedLocal
    def transform(self, X, y='deprecated', copy=None):
        if not issparse(X):
            raise ValueError('X must be sparse (dense implementation coming soon, or not)')
        return self.fold(X)

    #
    # TODO: do not save the map, but still recompute the collision rate
    # (find collumns that collide and their distribution in the matris, i.e.
    # sum of values of each cell they collide, how many molecules they are in...)
    #


@whatable
class MultiFolder(object):

    def __init__(self, fold_size=1023, seeds=2, force_num_hashes=False, target_density=None,
                 safe=True, as_binary=True, save_map=False):

        # At some point we should implement these
        if target_density is not None:
            raise NotImplementedError('To implement: target_density')
        if force_num_hashes:
            raise NotImplementedError('To implement: force_num_hashes')
        if save_map:
            raise NotImplementedError('To implement: save_map')

        if not isiterable(seeds):
            seeds = tuple(range(seeds))
        self.seeds = seeds

        self.fold_size = fold_size
        self.safe = safe
        self.as_binary = as_binary
        self._save_map = save_map

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        if not issparse(X):
            raise ValueError('X must be sparse (dense implementation coming soon, or not)')
        return self

    # noinspection PyUnusedLocal
    def transform(self, X, y='deprecated', copy=None):
        if not issparse(X):
            raise ValueError('X must be sparse (dense implementation coming soon, or not)')
        Xfolded = None
        for seed in self.seeds:
            # TODO: it is easy to make a meta-folder that will keep track of key -> fold assignments
            # (just merge the dictionaries of these MurmurFolders.save_map into dicts key -> [folds])
            folder = MurmurFolder(fold_size=self.fold_size, seed=seed,
                                  positive=True, safe=self.safe,
                                  as_binary=self.as_binary, save_map=False)
            if Xfolded is None:
                Xfolded = folder.fold(X)
            else:
                Xfolded = Xfolded + folder.fold(X)
        return Xfolded

    def fold(self, X):
        return self.transform(X)


@whatable
class ZeroColumns(object):

    def __init__(self, columns, invert=False, origin='from_train', copy=True):
        # N.B. as usual, the numpy array hash will be valid for a couple of releases...
        # So for the time being, I also add a origin attribute and hide columns in the id
        self._columns = np.unique(columns)
        self.origin = origin
        self.invert = invert
        self._copy = copy

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        if not issparse(X):
            raise ValueError('X must be sparse (dense implementation coming soon, or not)')
        return self

    # noinspection PyUnusedLocal
    def transform(self, X, y='deprecated', copy=False):
        if not issparse(X):
            raise ValueError('X must be sparse (dense implementation coming soon, or not)')
        if copy is None:
            copy = self._copy
        if copy:
            X = X.copy()
        return make_columns_zero(X, self._columns, zero_other=self.invert)


def pre_model(model,
              copy=True,
              zero_columns=None,
              binarize_threshold=None,
              binarize_fast=True,
              folder=None,
              scale=False,
              normalize=None):
    steps = []
    # Get rid of some columns by making them constant zero?
    if zero_columns is not None:
        steps.append(('zero_columns', zero_columns))
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
    # All together
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
                zero_columns=None,
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
                      zero_columns=zero_columns,
                      binarize_threshold=binarize_threshold,
                      folder=folder,
                      scale=scale,
                      normalize=row_normalizer)
    result = call_dict(ignores=('Xlab', 'y_lab',
                                'Xunl', 'y_unl', 'i2m_unl',
                                'hdf',
                                'recompute', 'cache_dir',
                                'model_stats', 'save_model', 'save_predictions'))
    # whatamise
    # FIXME: narrow down to proper key (recursively)
    for column in ('model', 'folder', 'zero_columns'):
        result[column] = what2id(result[column])
    what = What(name='result', conf=result)
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
            Xlab = make_columns_zero(Xlab, columns_out)
            Xunl = make_columns_zero(Xunl, columns_out)
        if min_radius is not None:
            columns_out = hdf.query('radius < %d' % min_radius).column.values
            Xlab = make_columns_zero(Xlab, columns_out)
            Xunl = make_columns_zero(Xunl, columns_out)

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
            # Of course, this should actually be model independent, only fallback to pickle
            pd.to_pickle(model, op.join(cache_dir, 'model.pkl'))
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


def compute_results(recompute=False, num_jobs=4, binarize=True, l1=False, zero_dupes=False):
    print('Loading data...')
    # The competition benchmark (N.B. also features not in train)
    i2m_unl, i2f_unl, Xunl, _ = rdkhash_feature_matrix(fpt='ecfp', dset='unl')
    y_unl = read_benchmark_labels()
    # The training data
    i2m_lab, i2f_lab, f2i_lab, Xlab, y_lab = X_train_feats(fpt='ecfp', dsets='lab',
                                                           use_representatives=False, transductive=False)
    # Information about the features
    hdf = munge_rdk_hashes_df(fpt='ecfp', nthreads=4, recompute=False, columns=['column', 'radius',
                                                                                'representative_train',
                                                                                'representative_transductive'])
    hdf = hdf.query('column < %d' % max(Xunl.shape[1], Xlab.shape[1]))

    # Show some info about our datasets
    print_matrix_summary(Xlab, y_lab, 'train')
    print_matrix_summary(Xunl, y_unl['active'], 'test')

    # Experimental variants...

    if l1:
        models = [
            # Slow and not so competitive
            ('tdtl1', LOGREG_MODELS.tdt_l1, logreg_stats),
        ]
    else:
        models = [
            ('tdtl2', LOGREG_MODELS.tdt_l2, logreg_stats),
        ]

    if zero_dupes:
        zerofiers = [
            ZeroColumns(columns=hdf['representative_train'].values, invert=True, origin='from_train'),
            ZeroColumns(columns=hdf['representative_transductive'].values, invert=True, origin='transductive')
        ]
    else:
        zerofiers = [None]

    max_radii = [None, 2, 3, 4]  # 6, 8, 10
    min_radii = [None]  # But this could also be of interest

    if binarize:
        binarizers = [0]
    else:
        binarizers = [None]

    folders = [None]
    if binarize:
        folder_as_binary = [True, False]  # True, True, True -> True or True, True, True -> 3
    else:
        folder_as_binary = [False]        # 3, 2, 5 -> 10
    num_seeds = 2
    hashes_per_cols = (2, 3)
    for seed, as_binary in product(range(num_seeds), folder_as_binary):
        folders += [
            # Folder(seed=seed, fold_size=511, as_binary=as_binary, save_map=False),
            Folder(seed=seed, fold_size=1023, as_binary=as_binary, save_map=False),
            Folder(seed=seed, fold_size=2047, as_binary=as_binary, save_map=False),
            Folder(seed=seed, fold_size=4091, as_binary=as_binary, save_map=False),
            Folder(seed=seed, fold_size=8191, as_binary=as_binary, save_map=False),
            Folder(seed=seed, fold_size=16383, as_binary=as_binary, save_map=False),
            Folder(seed=seed, fold_size=32767, as_binary=as_binary, save_map=False),
        ]
        for hashes_per_col in hashes_per_cols:
            seeds = tuple(range(hashes_per_col * seed, hashes_per_col * seed + hashes_per_col))
            # noinspection PyTypeChecker
            folders += [
                # MultiFolder(seeds=seeds, fold_size=511, as_binary=as_binary, save_map=False),
                # MultiFolder(seeds=seeds, fold_size=1023, as_binary=as_binary, save_map=False),
                MultiFolder(seeds=seeds, fold_size=2047, as_binary=as_binary, save_map=False),
                MultiFolder(seeds=seeds, fold_size=4091, as_binary=as_binary, save_map=False),
                MultiFolder(seeds=seeds, fold_size=8191, as_binary=as_binary, save_map=False),
                MultiFolder(seeds=seeds, fold_size=16383, as_binary=as_binary, save_map=False),
                MultiFolder(seeds=seeds, fold_size=32767, as_binary=as_binary, save_map=False),
            ]
    if zero_dupes:
        unseen_in_foldings = False,
    else:
        unseen_in_foldings = True, False

    row_normalizers = None,  # 'l1', 'l2', None

    experiments = list(product(
        models,
        zerofiers,
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
            zero_columns=zerofier,
            min_radius=min_radius, max_radius=max_radius,
            binarize_threshold=binarizer,
            folder=folder, allow_unseen_in_folding=unseen_in_folding,
            scale=False,
            row_normalizer=row_normalizer,
            recompute=recompute,
        )
        for ((model_name, model, model_stats),
             zerofier,
             min_radius, max_radius,
             binarizer,
             folder, unseen_in_folding,
             row_normalizer) in tqdm.tqdm(experiments, unit='experiment')
    )


def compute_at_home():
    import socket
    host = socket.gethostname()
    if host == 'snowy':
        order = [(False, True), (False, True), (False, True)]
        num_jobs = 4
    elif host == 'mumbler':
        order = [(False, True), (True, False), (True, False)]
        num_jobs = 10
    else:
        raise Exception('unknown host %s' % host)
    for l1, zero_dupes, binarize in product(*order):
        compute_results(recompute=False,
                        num_jobs=num_jobs,
                        l1=l1, zero_dupes=zero_dupes, binarize=binarize)


# --- Experiment analysis

def human_sort_columns(df):

    _TIDY_COLUMN_ORDER = [
        # Experimental variables
        'data_name',
        'model_name',
        'binarize_threshold',
        'is_counts',
        'folder',
        'allow_unseen_in_folding',
        'fold_size',
        'fold_seed',
        'min_radius',
        'max_radius',
        'row_normalizer',
        'scale',
        # Model stats
        'model_sparsity',
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
    columns = [column for column in _TIDY_COLUMN_ORDER if column in df.columns]
    columns += [column for column in df.columns if column not in columns]
    return df[columns]


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
    df = human_sort_columns(df)

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
    fillna(df, 'fold_seed', -1)
    fillna(df, 'allow_unseen_in_folding', False)
    fillna(df, 'binarize_threshold', np.inf)
    fillna(df, 'max_radius', np.inf)
    fillna(df, 'min_radius', -np.inf)
    df['fold_seed'] = df['fold_seed'].astype(np.int)
    df['fold_size'] = df['fold_size'].apply(lambda x: int(x) if np.isfinite(x) else 2**32-1)
    df['is_counts'] = df['binarize_threshold'].apply(lambda x: x != 0)

    return human_sort_columns(df)


# --- Legacy results readers

def tidy_results_01(cache_path=op.join(RESULTS_DIR, '01results', 'results.tidy.feather')):
    """These are old results, we can still read the dataframe."""

    df = feather.read_dataframe(cache_path)

    # Make missings sendible for grouping

    def fillna(df, column, value):
        df[column] = df[column].fillna(value=value)

    fillna(df, 'row_normalizer', 'none')
    fillna(df, 'fold_size', np.inf)
    fillna(df, 'fold_seed', -1)
    fillna(df, 'allow_unseen_in_folding', False)
    fillna(df, 'binarize_threshold', np.inf)
    fillna(df, 'max_radius', np.inf)
    fillna(df, 'min_radius', -np.inf)
    df['fold_seed'] = df['fold_seed'].astype(np.int)
    df['fold_size'] = df['fold_size'].apply(lambda x: int(x) if np.isfinite(x) else 2**32-1)
    df['is_counts'] = df['binarize_threshold'].apply(lambda x: x != 0)

    return human_sort_columns(df)


if __name__ == '__main__':
    compute_at_home()
    tidy_results(recompute=True).info()
    print('Open logreg_competition_benchmark_explorations.ipynb')

#
# Other things to try:
#   - ... (see logreg_competition_benchmark_explorations.ipynb)
# Other important things to report:
#   - Check for actual collisions (better use the big SMARTS <-> rdkhash tables prepared)
#     Can be done efficiently by keeping stats with murmur hasher.
#   - A proper dataset (My malaria export from chembl is already prepared)
#
