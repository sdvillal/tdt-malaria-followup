from __future__ import print_function

import h5py
from sklearn.linear_model import LogisticRegression

from ccl_malaria.logregs_analysis import logreg_experiments_to_deploy, deployment_models
from ccl_malaria.molscatalog import MalariaCatalog
from ccl_malaria.trees_analysis import trees_results_to_pandas
from minioscail.common.misc import ensure_dir
from tdt_autopsy.config import DATA_DIR
from tdt_autopsy.logregs_extra_experiments import LOGREG_MODELS
from tdt_autopsy.substructure_collision_analysis import X_train_feats

smiles = 'CC1CCC/C(C)=C1/C=C/C(C)=C/C=C/C(C)=C/C=C/C=C(C)/C=C/C=C(C)/C=C/C2=C(C)/CCCC2(C)C'


#
# On reproducibility:
#   - we can install sklearn 0.14.1 from conda defaults
#   - rdkit 2013.09 is nowhere to be seen in rdkit channel
# Anyway they are oldie, so let's just stick to using newer versions.
#


def trees_deployers():

    #
    # We used this version of sklearn:
    #   https://github.com/scikit-learn/scikit-learn/tree/0.14.1/sklearn
    # ExtraTrees defaults:
    #   https://github.com/scikit-learn/scikit-learn/blob/34c4908369968dd0f77897ec9dd8c227e7545478/sklearn/tree/tree.py#L657-L666
    # RandomForest defaults:
    #   https://github.com/scikit-learn/scikit-learn/blob/34c4908369968dd0f77897ec9dd8c227e7545478/sklearn/ensemble/forest.py#L742-L755
    #
    # In no case there existed yet an auto-class balancing parameter.
    #

    # noinspection PyUnusedLocal
    deployers = trees_results_to_pandas()

    # noinspection PyUnusedLocal
    columns = [
        # featurizer setup
        'data_setup',         # Here: 'MalariaRDKFsExampleSet__dset=lab__remove_ambiguous=True__remove_with_nan=True'
        # partitioner setup
        'eval_setup',         # Here: 'oob'
        # model setup
        'model_setup',        # A unique ID string for the model setup
        'model_type',         # Here one of etc or rfc
        'model_num_trees',    # The number of trees built in the forest
        'model_seed',         # The random seed for the forest
        #
        'result',             # An OOBResult, allowing to load anything related to the result (model, partitions...)
        'oob_accuracy',       # Out of bag accuracy
        'oob_auc',            # Out of bag AUC
        'oob_enrichment5',    # Out of bag enrichment at 5%
        # timings
        'train_time',         # Time taken to train
        'test_time',          # Time taken to test
        # more metadata
        'title',      # This is a copy-pasta mistake (says malaria-trees-oob)
        'comments',   # Usually None
        'date',       # When was this trained
        'host',       # The machine where this was trained
        'idate',      # Usually None
        'fsource',    # Source code of the function that had been used to train the model
    ]

    # We did not save the ensembles. We recomputed them back in February but with a
    # different version of sklearn and then we deleted them thinking we had a backup
    # (and we did not). I do not feel like retraining, as we should then use more data,
    # we are using not so interesting features and overall I think our logistic regression
    # experiments are way more interesting.


def logreg_deployers():

    #
    # We used this version of sklearn:
    #   https://github.com/scikit-learn/scikit-learn/tree/0.14.1/sklearn
    # LogisticRegression defaults:
    #   https://github.com/scikit-learn/scikit-learn/blob/34c4908369968dd0f77897ec9dd8c227e7545478/sklearn/linear_model/logistic.py#L97-L99
    #
    # We did use 'auto' for class balancing, so we used resampling methods here.
    # Note that indeed we tested this in parameter selection, and "auto" was working best.
    #
    # Note the newer options (specially now saga optimizer is recommended) added since the competition:
    #  http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    #

    deployers = logreg_experiments_to_deploy()  # Here we have a fancy pandas query we decided almost "just because"

    # noinspection PyUnusedLocal
    columns = [
        # featurizer setup
        'data_setup',   # A string describing the features used (see later comment)
        'folder_seed',  # The seed used by the folder (-1 for unfolded)
        'folder_size',  # The size used by the folder (0 for unfolded)
        # model setup
        'model_setup',        # String identifying the model setup
        'penalty',            # Here: l1, l2
        'C',                  # Here: 1.0 or 5.0
        'intercept_scaling',  # Here: 1.0
        'fit_intercept',      # Here: True
        'class_weight',       # Here: auto
        'tol',                # Here: 0.0001
        'dual',               # Here: False
        'random_state',       # Here: some random seed
        # cached evaluation results
        'result',             # A Result object, allowing to load anything related to the result (model, partitions...)
        'num_present_folds',  # The number of considered folds
                              # Remember, many folds might not been computed if the previous folds were not promising
                              # For this selection of results: all the folds were computed
        'auc_mean',           # Mean AUC between folds
        'enrichement5_mean',  # Mean enrichment at 5% between folds
        # partitioner setup
        'eval_setup',         # A string representing the evaluation setup
        'cv_seed',            # Random seed for the cross-val splitter
        'num_cv_folds',       # Number of cross validation folds
        # more metadata
        'index',      # A string unique for each result
        'title',      # This is a copy-pasta mistake (says malaria-trees-oob)
        'comments',   # Usually None
        'date',       # When was this trained
        'host',       # The machine where this was trained
        'idate',      # Usually None
        'fsource',    # Source code of the function that had been used to train the model
    ]

    #
    # We are deploying only 8 model setups (see also the pandas query).
    #
    # The featurizer was always the same for the deployed models:
    #   MalariaFingerprintsExampleSet#
    #   dset=lab#
    #   folder=None#
    #   keep_ambiguous=False#
    #   only01=True#
    #   only_labelled=True#
    #   zero_dupes=all
    # Which means:
    #   - unfolded
    #   - we removed ambiguous from the training set
    #   - we used binary fingerprints (instead of counts)
    #   - we used only features present in labelled
    #   - we collapsed features that "transductively" were present in exactly the same molecules
    #
    # Note also: we are not anymore applying the "only use last fold because of stupid caching"
    # bug. If we deploy with the bug, we should then just keep the one model used per evaluation
    # (which was the last fold) and scale its contribution by the number of folds used
    # (that is, if num_folds was 7, use 7 * last_fold_model(x) when aggregating).
    #
    models = deployment_models(deployers_df=deployers, with_bug=False)

    results = deployers.result
    print('Deploying %d logistic regression experiments (%d classifiers)' % (len(results), len(models)))

    for result in results:
        print(result.logreg_coefs(0).shape)

    exit(22)

    #
    # Regarding removal of exact duplicates.
    # The clevermost would be to assign the smallest sub-prefix of the growing feature set
    # as the feature to use... it can be done after the fact.
    #

    #
    # print(mfm.mols_with_feature(1913))
    # print(molscatalog.mol(mfm.mols_with_feature(1913)[0]))
    # print(cansmi(molscatalog.mol(mfm.mols_with_feature(1913)[0])))

    # Featurizer
    # For simplicity, let's just create a text file
    # import os.path as op
    # import gzip
    # with open(op.expanduser('~/lab.merged.s2i'), 'rt') as reader:
    #     with gzip.open(op.expanduser('~/lab.merged.trans.s2i.gz'), 'wt') as writer:
    #         for line in reader:
    #             smiles, i = line.split()
    #             i = int(i)
    #             writer.write('%s %d %d\n' % (smiles, i, i2r[i]))

    # Models
    c = LogisticRegression()
    for model in models:
        print(model)


import os.path as op
DEPLOY_PATH = op.join(DATA_DIR, '--unfolded-explorations', 'deploy-models')


def simplemost_deploy(path=DEPLOY_PATH):

    # Load the train data
    _, i2f_lab, _, Xlab, y_lab = X_train_feats(fpt='ecfp', dsets='lab')

    # Train
    model = LOGREG_MODELS.tdt_l2(Xlab, y_lab, train=True)

    # Save

    path = op.join(ensure_dir(path), 'model.hdf5')

    model_description = {
        'train_data': 'full-TDT2014-unambiguous',
        'featurizer': 'type=ecfc,id=rdkithash,radius=all,folder=unfolded,zero_dupes=False',
        'preprocessor': 'none',
        'model_setup': str(model),
    }

    with h5py.File(path, 'w') as h5:
        model_g = h5.create_group('model')
        model_g['coef'] = model.coef_
        model_g['intercept'] = model.intercept_
        model_g['col2hash'] = i2f_lab
        for k, v in model_description.items():
            model_g.attrs[k] = v


import numpy as np
from scipy.sparse import csr_matrix
from rdkit.Chem import AllChem

mols = [
    'COc1ccc(NC(=O)CSc2nc(ns2)c3ccccc3Cl)c(OC)c1',
    'COC(=O)c1ccc(\C=C(\C#N)/C(=O)Nc2cccc(c2)[N+](=O)[O-])cc1',
]


def smiles2fpt(smi):
    if isinstance(smi, str):
        mol = AllChem.MolFromSmiles(smi)
    else:
        mol = smi
    fpt = AllChem.GetMorganFingerprint(mol,
                                       200,
                                       bitInfo=None,
                                       useFeatures=False,
                                       useChirality=False,
                                       useBondTypes=False)
    return fpt.GetNonzeroElements()


class Predictor(object):

    def __init__(self, path=DEPLOY_PATH):
        # Load the model into memory
        with h5py.File(op.join(path, 'model.hdf5')) as h5:
            self._logreg = LogisticRegression()
            self._logreg.coef_ = h5['model']['coef'][()]
            self._logreg.intercept_ = h5['model']['intercept'][()]
            self._hash2col = {h: c for c, h in enumerate(h5['model']['col2hash'][()])}
            self.meta = dict(h5['model'].attrs)

    def __call__(self, fpt):
        # batch_size
        # an optimization would be to store the max radius of any seen feature
        cols = np.zeros(len(fpt), dtype=np.int)
        data = np.zeros(len(fpt), dtype=np.float)
        num_set_vals = 0
        for rdkhash, count in fpt.items():
            col = self._hash2col.get(rdkhash)
            if col is not None:
                cols[num_set_vals] = col
                data[num_set_vals] = count
                num_set_vals += 1
        cols = cols[:num_set_vals]
        data = data[:num_set_vals]
        # FIXME: check when hash2col does not return properly sorted stuff
        X = csr_matrix((data, cols, [0, num_set_vals]),
                       shape=(1, len(self._hash2col)),
                       dtype=data.dtype)
        return self._logreg.predict_proba(X)[0]


if __name__ == '__main__':
    # simplemost_deploy()
    predictor = Predictor()

    mc = MalariaCatalog()

    import sys
    import tqdm
    from sklearn.metrics import roc_auc_score

    # for smi in mols:
    scores = []
    gt = []
    for molid in tqdm.tqdm(mc.lab(), unit='mol', file=sys.stdout):
        # smi = mc.molid2smiles(molid)
        smi = mc.molid2mol(molid)
        y = mc.molid2label(molid, as01=True)
        if smi is not None and y == y:
            scores.append(predictor(smiles2fpt(smi))[1])
            gt.append(y)
        if len(gt) > 0 and len(gt) % 10000 == 0 and np.sum(gt) > 0:
            print('\n %.2f' % roc_auc_score(gt, scores))

    print('\n %.2f' % roc_auc_score(gt, scores))


#
# Old example for a deployed model (from mol to prediction)
#
# class Fingerprinter(object):
#
#     def __init__(self, f2i, representatives=None, fingerprinter=None, return_as='dense'):
#         self.f2i = f2i
#         if fingerprinter is None:
#             fingerprinter = partial(morgan_fingerprint, max_radius=200, fcfp=False, explainer=None)
#         self.representatives = representatives
#         if representatives is not None:
#             self.f2i = {h: representatives[i] for h, i in self.f2i.keys()}
#         self.fingerprinter = fingerprinter
#         self.return_as = return_as
#
#     def __call__(self, mol_or_molifyable):
#         fpt = self.fingerprinter(mol_or_molifyable)
#         if len(fpt) == 0:
#             fpt = {}
#         elif isinstance(next(iter(fpt)), string_types):
#             fpt = {self.f2i[f]: len(c) for f, c in fpt.items()}
#         else:
#             fpt = {self.f2i[f]: c for f, c in fpt.items()}
#         if self.return_as == 'dense':
#             x = np.zeros(len(self.f2i), dtype=int)
#             for h, c in fpt.items():
#                 x[h] = c
#             return x
#         elif self.return_as == 'sparse' or self.return_as == 'csr':
#             raise NotImplementedError()
#         else:
#             return fpt
#
#
# def train_test(X, y, actual, f2i, model=LogisticRegression(), model_id='new-rdkhash-l2'):
#
#     print('Fitting...')
#     model.fit(X, y)
#
#     print('Testing...')
#     mc = MalariaCatalog()
#     scores = []
#     for molid in mc.unl():
#         fpt = morgan_fingerprint(mc.molid2mol(molid), max_radius=200, fcfp=False, explainer=None)
#         x = np.zeros(X.shape[1], dtype=np.int)
#         for h, c in fpt.items():
#             try:
#                 x[f2i[h]] = c
#             except KeyError:
#                 pass
#         scores.append({
#             'molid': molid,
#             'score': model.predict_proba(x.reshape(1, -1))[0][1]
#         })
#
#     scores = pd.DataFrame(scores).set_index('molid', drop=False)
#
#     result = score_result(scores, actual, model=model_id)
#
#     return model, scores, result
#
