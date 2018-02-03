from __future__ import division
import os.path as op
import numpy as np
import pandas as pd
import feather

from minioscail.common.misc import ensure_dir
from tdt_autopsy.config import DATA_DIR
from tdt_autopsy.substructure_collision_analysis import (munge_rdk_hashes_df,
                                                         tall_fat_rdkhash_feature_matrix)


RESULTS_DIR = op.join(DATA_DIR, '--unfolded-explorations', 'unseen-features-collisions-results')


def collisions_analysis():

    # noinspection PyUnusedLocal
    ALL_COLUMNS = [

        # The hash assigned by rdkit to the substructure
        # (rdkhash, column) forms the lookup up table to remap hashes to columns
        'rdkhash',

        # The column we assigned in our bijective map rdkhash -> column
        # Usually this is 0...num_hashes; it is sorted so that all features in lab come first,
        # followed by features not in lab but in amb, followed by features not in (lab + amb)
        # but in unl, followed by features not in (lab + amb + unl) but in scr
        'column',

        # Maps to the column of the representative, as computed in the labelled dataset
        # Column duplicates are found, by looking at the design matrix with only labelled molecules as rows,
        # and a single column is selected to represent the whole group of duplicates.
        'representative_train',

        # Maps to the column of the representative, as computed in the labelled dataset
        # Column duplicates are found, by looking at the design matrix with all molecules as rows,
        # and a single column is selected to represent the whole group of duplicates.
        # Called transductive in a poor homage to Vapnik.
        'representative_transductive',

        # The (min) radius reported to generate the feature
        # (in very few rare, rdkit-bug-like cases, there is more than one radius)
        'radius',
        # The number of SMARTS this hash maps to
        # Note that the map is many to many
        #   - The same hash can give rise to different SMARTS
        #     (mostly because of 32bit-hashing collisions and atom invariants collapsing different SMARTS structures)
        #   - The same SMARTS can give rise to different hashes
        #     (because of atom invariants seeing differences not representable by SMARTS)
        # Typically we have many more hashes than SMARTS (e.g. ~40M SMARTS vs ~50M hashes in TDT commercial dataset)
        'num_smarts',
        # The number of positive molecules having this feature
        'num_positive_mols',
        # The number of negative molecules having this feature
        'num_negative_mols',
        # The total number of occurrences of the hash in positive molecules
        'positive_counts',
        # The total number of occurrences of the hash in negative molecules
        'negative_counts',

        # Note:
        #   num_lab_mols = num_positive_mols + num_negative_mols
        #   lab_counts = positive_counts + negative_counts


        # Same counts for:
        #   - amb: ambiguous dataset (i.e. labelled but uncertain outcome)
        #   - unl: the 1056 mols competition dataset
        #   - scr: the 5.5+M mols commercial dataset
        'num_amb_mols',
        'amb_counts',
        'num_unl_mols',
        'unl_counts',
        'num_scr_mols',
        'scr_counts'
    ]

    # hdf = munge_rdk_hashes_df(fpt='ecfp', nthreads=4, recompute=False, columns=['column', 'radius',
    #                                                                             'representative_train',
    #                                                                             'representative_transductive'])
    hdf = munge_rdk_hashes_df(fpt='ecfp', nthreads=4, recompute=False, columns=None)
    hdf['num_lab_mols'] = hdf['num_positive_mols'] + hdf['num_negative_mols']
    hdf['lab_counts'] = hdf['positive_counts'] + hdf['negative_counts']
    hdf['num_total_mols'] = hdf['num_lab_mols'] + hdf['num_amb_mols'] + hdf['num_unl_mols'] + hdf['num_scr_mols']
    hdf['total_counts'] = hdf['lab_counts'] + hdf['amb_counts'] + hdf['unl_counts'] + hdf['scr_counts']

    # features not in lab
    unseen_by_the_models = hdf.query('lab_counts == 0')
    # average number of molecules and counts they appear in
    print('Number of mols unseen features: %.2f +/- %.2f' % (
        unseen_by_the_models['num_total_mols'].mean(), unseen_by_the_models['num_total_mols'].std()))
    print('Counts unseen features: %.2f +/- %.2f' % (
        unseen_by_the_models['total_counts'].mean(), unseen_by_the_models['total_counts'].std()))

    print(unseen_by_the_models.describe())

    # Number of mols unseen features: 1.00 +/- 0.04
    # Counts unseen features: 3.44 +/- 25.89
    #             rdkhash        column  representative_train  \
    # count  4.774449e+07  4.774449e+07            47744490.0
    # mean   2.082584e+09  2.669362e+07                  -1.0
    # std    1.238918e+09  1.378265e+07                   0.0
    # min    6.500000e+01  2.821374e+06                  -1.0
    # 25%    1.003117e+09  1.475750e+07                  -1.0
    # 50%    2.049858e+09  2.669362e+07                  -1.0
    # 75%    3.146440e+09  3.862974e+07                  -1.0
    # max    4.294967e+09  5.056586e+07                  -1.0
    #
    #        representative_transductive        radius    num_smarts  \
    # count                   47744490.0  4.774449e+07  4.774449e+07
    # mean                          -1.0  6.180301e+00  1.021905e+00
    # std                            0.0  1.597087e+00  1.638071e-01
    # min                           -1.0  0.000000e+00  1.000000e+00
    # 25%                           -1.0  5.000000e+00  1.000000e+00
    # 50%                           -1.0  6.000000e+00  1.000000e+00
    # 75%                           -1.0  7.000000e+00  1.000000e+00
    # max                           -1.0  2.900000e+01  1.500000e+01
    #
    #        num_positive_mols  num_negative_mols  positive_counts  negative_counts  \
    # count         47744490.0         47744490.0       47744490.0       47744490.0
    # mean                 0.0                0.0              0.0              0.0
    # std                  0.0                0.0              0.0              0.0
    # min                  0.0                0.0              0.0              0.0
    # 25%                  0.0                0.0              0.0              0.0
    # 50%                  0.0                0.0              0.0              0.0
    # 75%                  0.0                0.0              0.0              0.0
    # max                  0.0                0.0              0.0              0.0
    #
    #        num_amb_mols    amb_counts  num_unl_mols    unl_counts  num_scr_mols  \
    # count  4.774449e+07  4.774449e+07  4.774449e+07  4.774449e+07  4.774449e+07
    # mean   1.605023e-03  1.806470e-03  2.749637e-04  3.323525e-04  9.997577e-01
    # std    4.003057e-02  5.101925e-02  1.657975e-02  2.336585e-02  1.556377e-02
    # min    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00
    # 25%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00
    # 50%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00
    # 75%    0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00
    # max    1.000000e+00  2.400000e+01  1.000000e+00  1.400000e+01  1.000000e+00
    #
    #          scr_counts  num_lab_mols  lab_counts  num_total_mols  total_counts
    # count  4.774449e+07    47744490.0  47744490.0    4.774449e+07  4.774449e+07
    # mean   3.442396e+00           0.0         0.0    1.001638e+00  3.444535e+00
    # std    2.588580e+01           0.0         0.0    4.052276e-02  2.588651e+01
    # min    0.000000e+00           0.0         0.0    1.000000e+00  1.000000e+00
    # 25%    1.000000e+00           0.0         0.0    1.000000e+00  1.000000e+00
    # 50%    1.000000e+00           0.0         0.0    1.000000e+00  1.000000e+00
    # 75%    2.000000e+00           0.0         0.0    1.000000e+00  2.000000e+00
    # max    1.754800e+04           0.0         0.0    3.000000e+00  1.754800e+04

    # Show how feature appearance might follow or not a Zipf distribution
    #   https://en.wikipedia.org/wiki/Zipf%27s_law
    # Fit, report goodness of fit
    # Or just simply plot total counts or total mols...

    # Collisions mol <-> feature analysis
    print('There are %d not-in-train substructures with no radius limit' % len(unseen_by_the_models))
    out_of_train_stats(dest_dir=op.join(RESULTS_DIR, 'infinite-radius'))

    # For smaller ECFP radii
    ecfp_6_features = hdf.query('radius <= 3 and num_lab_mols == 0').column.values
    print('There are %d not-in-train substructures with radius up to 3' % len(ecfp_6_features))
    out_of_train_stats(only_features=ecfp_6_features, dest_dir=op.join(RESULTS_DIR, 'radius3'))

    ecfp_4_features = hdf.query('radius <= 2 and num_lab_mols == 0').column.values
    print('There are %d not-in-train substructures with radius up to 2' % len(ecfp_4_features))
    out_of_train_stats(only_features=ecfp_4_features, dest_dir=op.join(RESULTS_DIR, 'radius2'))

    return hdf


def out_of_train_stats(only_features=None, dest_dir=None):
    # Load the large matrix
    Xall, num_mols, num_unique_cols = tall_fat_rdkhash_feature_matrix()

    # Xall = make_columns_zero(Xall, np.arange(num_unique_cols[0]))
    if only_features is not None:
        # Beware: only_features must not contain "seen in train" features
        Xall = Xall.tocsc()[:, only_features].tocsr()
    else:
        Xall = Xall.tocsc()[:, num_unique_cols[0]:].tocsr()

    # Remove labelled molecules
    Xall = Xall[num_mols[0]:, :]

    print('Looking at %d not in train molecules and %d not in train features' % Xall.shape)

    # Compute stats:
    #   - How many molecules a feature appears in (with and without counts)?
    #   - How many substructures not in train each molecule has (with and without counts)?

    # number of features not in train on each molecule
    feature_counts_not_in_train_per_mol = np.asarray(Xall.sum(axis=1)).ravel()
    # number of molecules not in train each feature appears in
    molecule_counts_not_in_train_per_feature = np.asarray(Xall.sum(axis=0)).ravel()

    # make binary, recompute stats
    Xall.data = np.ones_like(Xall.data)

    # number of features not in train on each molecule
    num_features_not_in_train_per_mol = np.asarray(Xall.sum(axis=1)).ravel()
    # number of molecules not in train each feature appears in
    num_molecules_not_in_train_per_feature = np.asarray(Xall.sum(axis=0)).ravel()

    per_feature_df = pd.DataFrame(dict(
        molecule_counts_not_in_train_per_feature=molecule_counts_not_in_train_per_feature,
        num_molecules_not_in_train_per_feature=num_molecules_not_in_train_per_feature,
    ))

    print(per_feature_df.describe())
    per_molecule_df = pd.DataFrame(dict(
        feature_counts_not_in_train_per_mol=feature_counts_not_in_train_per_mol,
        num_features_not_in_train_per_mol=num_features_not_in_train_per_mol,
    ))

    print(per_molecule_df.describe())

    pd.set_option('display.max_rows', None)

    print(per_feature_df.groupby('num_molecules_not_in_train_per_feature').size())
    # How many molecules each unseen feature appears in...    #
    # Very many rare features, still loads of not so rare...

    print(per_molecule_df.groupby('num_features_not_in_train_per_mol').size())
    #
    # How many unseen features each molecule has?
    #
    # 270214 molecules have not unseen substructure.
    #  21261 molecules have one unseen substructure...
    #
    # num_features_not_in_train_per_mol
    # 0      270214
    # 1       21261
    # ...

    if dest_dir is not None:
        feather.write_dataframe(per_feature_df, op.join(ensure_dir(dest_dir), 'per_feature_df.feather'))
        feather.write_dataframe(per_molecule_df, op.join(ensure_dir(dest_dir), 'per_molecule_df.feather'))

    # TODO: histogram or better, CDF, possibly removing 0 and 1 or doing something so we can see the long tails


if __name__ == '__main__':
    collisions_analysis()
