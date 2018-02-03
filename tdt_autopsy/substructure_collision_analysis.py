from __future__ import print_function, division
from future.utils import string_types

import glob
import os.path as op
from functools import partial
from natsort import natsorted
from collections import defaultdict, Counter
from itertools import product, chain
from toolz import take

import array
import feather
import numpy as np
from scipy.sparse import vstack, coo_matrix, csr_matrix
import pandas as pd
import numba
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import tqdm
tqdm.tqdm.monitor_interval = 0

from rdkit.Chem import AllChem

from ccl_malaria.features import zero_columns
from ccl_malaria.logregs_fit import malaria_logreg_fpt_providers
from ccl_malaria.molscatalog import lab_molid2mol_memmapped_catalog, MalariaCatalog
from ccl_malaria.rdkit_utils import (group_substructures, unsanitized_mol_from_smiles, mol_from_smarts,
                                     has_substruct_match, has_query_query_match, morgan_fingerprint)
from minioscail.common.misc import ensure_dir
from tdt_autopsy.config import DATA_DIR


def representatives_df(add_num_mols=True, add_mols=False, query=None):

    # "Controlled" collisions

    rf_lab, _, _, _ = malaria_logreg_fpt_providers(None)
    mfm = rf_lab.mfm()
    i2s = mfm.substructures()
    _, i2r = mfm.duplicate_features_representatives()

    repr_df = pd.DataFrame({
        'r': i2r,                   # The substructure representative index
        'i': np.arange(len(i2r)),   # The substructure index
        's': i2s                    # The substructure SMARTS
    })

    if query:
        repr_df = repr_df.query(query).copy()

    if add_mols:
        molcatalog = lab_molid2mol_memmapped_catalog()
        repr_df['mols'] = repr_df['r'].apply(lambda r: molcatalog.mols(mfm.mols_with_feature(r)))
        if add_num_mols:
            repr_df['num_mols'] = repr_df['mols'].apply(len)
    elif add_num_mols:
        # We could just do this unique(r) times
        repr_df['num_mols'] = repr_df['r'].apply(lambda r: len(mfm.mols_with_feature(r)))

    return repr_df


def build_collapsing_features_info_df(add_all_counts=True,
                                      mols_instantiator=unsanitized_mol_from_smiles,
                                      pattern_instantiator=mol_from_smarts,
                                      matcher=has_substruct_match,
                                      dest_file=None):

    try:
        return pd.read_pickle(dest_file)
    except Exception:
        pass

    rf_lab, rf_amb, rf_unl, rf_scr = malaria_logreg_fpt_providers(None)
    mfm_lab, mfm_amb, mfm_unl, mfm_scr = rf_lab.mfm(), rf_amb.mfm(), rf_unl.mfm(), rf_scr.mfm()
    repr_df = representatives_df(add_num_mols=False, query='r < 1000000000')

    voluntary_colisions = []

    num_atoms_counts = np.zeros(100000, dtype=int)

    for r, rdf in tqdm.tqdm(repr_df.groupby('r')):
        num_lab_mols = len(mfm_lab.mols_with_feature(r))
        if add_all_counts:
            num_amb_mols = len(mfm_amb.mols_with_feature(r))
            num_unl_mols = len(mfm_unl.mols_with_feature(r))
            num_scr_mols = len(mfm_scr.mols_with_feature(r))
        else:
            num_amb_mols = num_unl_mols = num_scr_mols = -1
        try:
            graph, groups, representative, roots, leaves, num_atoms, has_cycles, has_equal_non_equal = (
                group_substructures(rdf['s'],
                                    matcher=matcher,
                                    mol_instantiator=mols_instantiator,
                                    pattern_instantiator=pattern_instantiator))
            num_groups = len(groups)
            min_num_atoms = min(num_atoms)
            max_num_atoms = max(num_atoms)
            na, nac = np.unique(num_atoms, return_counts=True)
            num_atoms_counts[na] += nac
            num_different_substructures = len(np.unique(representative))
        except RuntimeError:
            num_groups = min_num_atoms = max_num_atoms = 0
            num_different_substructures = 1
            has_cycles = has_equal_non_equal = None
        voluntary_colisions.append({
            'r': r,
            'num_substructures': len(rdf),
            'num_different_substructures': num_different_substructures,
            'num_groups': num_groups,
            'num_lab_mols': num_lab_mols,
            'num_amb_mols': num_amb_mols,
            'num_unl_mols': num_unl_mols,
            'num_scr_mols': num_scr_mols,
            'min_num_atoms': min_num_atoms,
            'max_num_atoms': max_num_atoms,
            'has_cycles': has_cycles,
            'has_equal_non_equal': has_equal_non_equal,
        })

    columns = [
        'r',
        'num_substructures',
        'num_different_substructures',
        'num_groups',
        'num_lab_mols',
        'num_amb_mols',
        'num_unl_mols',
        'num_scr_mols',
        'min_num_atoms',
        'max_num_atoms',
        'has_cycles',
        'has_equal_non_equal',
    ]
    collisions_df = pd.DataFrame(voluntary_colisions)[columns]

    if dest_file:
        pd.to_pickle((collisions_df, num_atoms_counts), dest_file)

    return collisions_df, num_atoms_counts


@numba.jit(nopython=True, parallel=True)
def last_non_zero(counts):
    for i in range(len(counts) - 1, -1, -1):
        if counts[i] != 0:
            return i
    return -1


@numba.jit(nopython=True, parallel=True)
def mean_std_from_counts(counts):
    total_length = 0
    total_count = 0
    for i, count in enumerate(counts):
        total_length += i * count
        total_count += count
    mean = total_length / total_count
    total_length_std = 0
    for i, count in enumerate(counts):
        total_length_std += (i - mean) ** 2 * count

    return mean, np.sqrt(total_length_std / (total_count - 1))


def expand(counts):
    expanded = []
    for i, count in enumerate(counts):
        expanded += [i] * count
    return expanded


#
# --- ECFP vs FCFP when one then forgets about atom invariants (+ context) and go back to SMARTS
# It is kinda stupid given we are just using morgan algorithm to explore the molecular graph
# Differences will be due to removal of duplicates in one that are not removed in the other
# Also, obviously, hash values are not relatable between one and the other, so we are left
# with comparing only via quite non-trustworthy smarts based methods.
#

def cansmartssmi(mol):
    return AllChem.MolToSmiles(AllChem.MolFromSmarts(mol))


def ecfpvsfcfp():
    mc = MalariaCatalog()
    for i, molid in enumerate(mc.unl()):
        mol = mc.molid2mol(molid)
        ecfp = morgan_fingerprint(mol, fcfp=False)
        fcfp = morgan_fingerprint(mol, fcfp=True)
        mols_ecfp = set(list(map(cansmartssmi, ecfp.keys())))
        mols_fcfp = set(list(map(cansmartssmi, fcfp.keys())))
        if len(mols_ecfp - mols_fcfp) or len(mols_fcfp - mols_ecfp):
            print(i, molid, mc.molid2smiles(molid))
            print('In ECFP', mols_ecfp - mols_fcfp)
            print('In FCFP', mols_fcfp - mols_ecfp)
            print(len(ecfp), len(mols_ecfp), len(fcfp), len(mols_fcfp))
            print('-' * 10)
        # print(ecfp == fcfp)
        # print(sorted(ecfp) == sorted(fcfp))
        # print(ecfp, fcfp)


# ecfpvsfcfp()
# exit(22)

# --- Collect rdkit hashed fingerprints

RDK_HASH_FPT_DIR = ensure_dir(op.join(DATA_DIR, '--unfolded-explorations', 'rdkit-hashes-fingerprints'))
RDK_HASH_FPT_SHARDED_DIR = ensure_dir(RDK_HASH_FPT_DIR, 'shards')


def collect_rdkit_fpt(start=0, step=1, stop=None, dset='lab', fcfp=False, use_tqdm=True):
    mc = MalariaCatalog()
    mols = mc.lab() if dset == 'lab' else mc.unl() if dset == 'unl' else mc.scr()
    mols = mols[start:stop:step]
    dest_file = op.join(RDK_HASH_FPT_SHARDED_DIR, '%s_%s_%r_%r_%r.txt' % (
        ('fcfp' if fcfp else 'ecfp'), dset, start, stop, step))
    if op.isfile(dest_file):
        print('%s already done, skipping' % dest_file)
        return
    with open(dest_file, 'wt') as writer:
        for i, molid in enumerate(tqdm.tqdm(mols, unit=' mols', disable=not use_tqdm)):
            int_molid = start + i * step
            writer.write('%s %d' % (molid, int_molid))
            if not use_tqdm and i > 0 and i % 1000:
                print(op.basename(dest_file), '%d of %d' % (i, len(mols)))
            try:
                fpt = morgan_fingerprint(mc.molid2mol(molid), max_radius=200, fcfp=fcfp, explainer=None)
                for rdkhash, count in sorted(fpt.items()):
                    writer.write(' %d %d' % (rdkhash, count))
            except Exception:
                writer.write(' FAIL')
                print('WARNING: problem with %s, skipping' % molid)
            finally:
                writer.write('\n')
        writer.write('DONE')


def collect_all_rdkit_fpt(num_jobs=12, dsets=('lab', 'unl', 'scr'), fcfps=(True, False)):
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(collect_rdkit_fpt)
        (start, stop=None, step=num_jobs, dset=dset, fcfp=fcfp)
        for fcfp, dset, start in product(fcfps, dsets, range(num_jobs))
    )


# collect_all_rdkit_fpt(num_jobs=4)
# exit(22)


def rdkhash_feature_matrix(fpt='ecfp', dset='lab', num_shards=4, f2i=None, recompute=False):

    dest_file = op.join(RDK_HASH_FPT_DIR, '%s_%s.matrix.pickle' % (fpt, dset))

    # --- Load cached
    if not recompute:
        try:
            return pd.read_pickle(dest_file)
        except IOError:
            pass

    # --- Recompute
    mc = MalariaCatalog()
    molids = []
    feats_set = set()
    if f2i is None:
        f2i = {}
    if not isinstance(f2i, dict):
        f2i = {f: i for i, f in enumerate(f2i)}

    # Handle damn amb special case
    amb = False
    if dset == 'amb':
        dset = 'lab'
        amb = True

    for start in range(num_shards):

        path = op.join(RDK_HASH_FPT_SHARDED_DIR, '%s_%s_%d_None_%d.txt' % (fpt, dset, start, num_shards))

        done = False

        with open(path, 'rt') as reader:
            for line in tqdm.tqdm(reader, desc='%s1st' % dset, unit='mol'):
                if line == 'DONE':
                    done = True
                    continue
                feats = line.strip().split()
                molid = feats[0]
                if len(feats) == 3 and 'FAIL' == feats[2]:
                    print('Failed: %s' % molid)
                    continue
                is_labelled = not np.isnan(mc.molid2label(molid, as01=True))
                if not is_labelled and dset == 'lab' and not amb:
                    # print('Ambiguous, but asked labelled: %s' % molid)
                    continue
                if is_labelled and amb:
                    # print('Labelled, but asked ambiguous: %s' % molid)
                    continue
                int_molid = int(feats[1])
                molids.append((int_molid, molid))
                hashes_counts = list(map(int, feats[2:]))
                feats_set.update(f for f in hashes_counts[0::2] if f not in f2i)
        if not done:
            raise Exception('Unfinished computation for %r' % path)

    # molid -> row number
    m2i = {m: i for i, (_, m) in enumerate(sorted(molids))}
    assert len(m2i) == len(molids)
    # rdkhash -> column number
    num_fs = len(f2i)
    f2i.update({f: i + num_fs for i, f in enumerate(sorted(feats_set))})
    print('Num mols: %d, Num feats: %d' % (len(m2i), len(f2i)))

    if dset != 'lab':
        y = None
    else:
        y = np.empty(len(m2i), dtype=np.bool)

    rows = array.array('I')
    cols = array.array('I')
    counts = array.array('I')

    for start in range(num_shards):
        path = op.join(RDK_HASH_FPT_SHARDED_DIR, '%s_%s_%d_None_%d.txt' % (fpt, dset, start, num_shards))
        with open(path, 'rt') as reader:
            for line in tqdm.tqdm(reader, desc='%s2nd' % dset, unit='mol', total=len(m2i) / num_shards):
                if line == 'DONE':
                    continue
                feats = line.strip().split()
                if len(feats) == 3 and 'FAIL' == feats[2]:
                    continue
                molid = feats[0]
                if molid not in m2i:
                    continue
                hashes_counts = list(map(int, feats[2:]))
                mi = m2i[molid]
                for h, c in zip(hashes_counts[::2], hashes_counts[1::2]):
                    rows.append(mi)
                    cols.append(f2i[h])
                    counts.append(c)
                if y is not None:
                    y[mi] = bool(mc.molid2label(molid, as01=True))

    X = coo_matrix((counts, (rows, cols)), shape=(len(m2i), len(f2i)), dtype=np.uint).tocsr()
    i2m = [molid for (molid, row) in sorted(m2i.items(), key=lambda x: x[1])]
    i2f = np.array([rdkhash for (rdkhash, col) in sorted(f2i.items(), key=lambda x: x[1])])

    pd.to_pickle((i2m, i2f, X.tocsr(), y), dest_file)

    return i2m, i2f, X.tocsr(), y


# Create design matrices for the different subsets
# Note that it must be done in the correct order ()
# for fpt in ('ecfp', 'fcfp'):
#     # noinspection PyTypeChecker
#     i2m, i2f, X, y = rdkhash_feature_matrix(fpt=fpt, dset='lab', num_shards=4, f2i=None, recompute=False)
#     # noinspection PyTypeChecker
#     i2m, i2f, X, y = rdkhash_feature_matrix(fpt=fpt, dset='amb', num_shards=4, f2i=i2f, recompute=False)
#     # noinspection PyTypeChecker
#     i2m, i2f, X, y = rdkhash_feature_matrix(fpt=fpt, dset='unl', num_shards=4, f2i=i2f, recompute=False)
#     # noinspection PyTypeChecker
#     i2m, i2f, X, y = rdkhash_feature_matrix(fpt=fpt, dset='scr', num_shards=4, f2i=i2f, recompute=False)
# exit(22)


def tall_fat_rdkhash_feature_matrix(fpt='ecfp'):

    # Load all
    _, _, Xlab, _ = rdkhash_feature_matrix(fpt=fpt, dset='lab')
    _, _, Xamb, _ = rdkhash_feature_matrix(fpt=fpt, dset='amb')
    _, _, Xunl, _ = rdkhash_feature_matrix(fpt=fpt, dset='unl')
    _, _, Xscr, _ = rdkhash_feature_matrix(fpt=fpt, dset='scr')

    # Get simple stats
    num_mols = (Xlab.shape[0], Xamb.shape[0], Xunl.shape[0], Xscr.shape[0])
    num_unique_cols = (
        Xlab.shape[1],
        Xlab.shape[1] - Xamb.shape[1],
        Xunl.shape[1] - Xlab.shape[1] - Xamb.shape[1],
        Xscr.shape[1] - Xunl.shape[1] - Xlab.shape[1] - Xamb.shape[1],
    )

    # Reshape

    def reshape(X, shape):
        return csr_matrix((X.data, X.indices, X.indptr), shape=shape)

    Xlab = reshape(Xlab, (Xlab.shape[0], Xscr.shape[1]))
    Xamb = reshape(Xamb, (Xamb.shape[0], Xscr.shape[1]))
    Xunl = reshape(Xunl, (Xunl.shape[0], Xscr.shape[1]))

    # Concat
    Xall = vstack((Xlab, Xamb, Xunl, Xscr))

    return Xall, num_mols, num_unique_cols


def find_rdkhash_dataset_duplicates(fpt='ecfp', transductive=True, recompute=False):

    cache_path = op.join(RDK_HASH_FPT_DIR, '%s_%s_duplicates_representatives.pickle' % (
        fpt, 'transductive' if transductive else 'labelled'
    ))

    if not recompute:
        try:
            return pd.read_pickle(cache_path)
        except IOError:
            pass

    # Load design matrices
    _, _, X, _ = rdkhash_feature_matrix(fpt=fpt, dset='lab')

    if transductive:
        for dset in ('amb', 'unl', 'scr'):
            _, _, Xo, _ = rdkhash_feature_matrix(fpt=fpt, dset=dset)
            X = vstack((X, Xo[:, 0:X.shape[1]]))

    # Find duplicates
    print('Finding duplicated groups')
    Xcsc = X.tocsc()
    dupes = defaultdict(list)
    for col in tqdm.tqdm(range(Xcsc.shape[1])):
        dupes[joblib.hash(Xcsc[:, col].indices)].append(col)
    dupes = sorted(map(tuple, dupes.values()))

    # Massage: get to_zero, representatives
    print('Selecting representatives per group')
    representatives = np.empty(X.shape[1], dtype=np.int)
    to_zero = []  # For use with "zero_columns"
    for group in tqdm.tqdm(dupes):
        group = sorted(group)
        representatives[group] = group[0]
        if len(group) > 1:
            to_zero += group[1:]
    to_zero = np.array(to_zero)

    print('Saving cache %s' % cache_path)
    pd.to_pickle((dupes, representatives, to_zero), cache_path)

    return dupes, representatives, to_zero

    # TODO: When we have substructures smiles, we can be cleverer at selecting representatives and splitting groups
    # (e.g. keep disjoint substructures, keep the smallest substructure of the family...)


# find_rdkhash_dataset_duplicates(fpt='ecfp', transductive=False, recompute=False)
# find_rdkhash_dataset_duplicates(fpt='fcfp', transductive=False, recompute=False)
# find_rdkhash_dataset_duplicates(fpt='ecfp', transductive=True, recompute=False)
# find_rdkhash_dataset_duplicates(fpt='fcfp', transductive=True, recompute=False)
# exit(33)

def X_train_feats(fpt='ecfp', dsets=('lab',), use_representatives=False, transductive=False):

    # Accept a single string dataset
    if isinstance(dsets, string_types):
        dsets = (dsets,)

    # Get the labelled information (i2f)
    i2m_lab, i2f_lab, Xlab, y_lab = rdkhash_feature_matrix(fpt=fpt, dset='lab')

    # Concat all the requested feature matrices, keeping only train columns
    Xs = []
    i2m = []
    ys = []
    for dset in dsets:
        if dset == 'lab':
            Xs.append(Xlab)
            i2m += i2m_lab
            ys.append(y_lab.astype(np.float))
        else:
            i2m_o, _, X_o, _ = rdkhash_feature_matrix(fpt=fpt, dset=dset)
            Xs.append(X_o[:, 0:len(i2f_lab)])
            i2m += i2m_o
            ys.append(np.full(len(i2m_o), np.nan, dtype=np.float))
    X = vstack(Xs)
    y = np.hstack(ys)

    assert X.shape == (len(i2m), len(i2f_lab))
    assert len(y) == len(i2m)

    # Zeroes all matrix columns that are represented by another column?
    if use_representatives:
        # Get the zeroing info
        dupes, representatives, to_zero = find_rdkhash_dataset_duplicates(fpt=fpt, transductive=transductive)
        # Zero matrix
        X = zero_columns(X, to_zero, zero_other=False)
        # Modify mapping
        f2i = {f: representatives[i] for i, f in enumerate(i2f_lab)}
    else:
        f2i = {f: i for i, f in enumerate(i2f_lab)}

    return i2m, i2f_lab, f2i, X, y


# --- Basic fpt stats.

FPT_STATS_DIR = op.join(DATA_DIR, '--unfolded-explorations', 'fpt-stats')
FPT_STATS_SHARDED_DIR = op.join(DATA_DIR, '--unfolded-explorations', 'fpt-stats', 'sharded')


class radius_hash(object):
    __slots__ = 'radiuses', 'hashes', 'dset_unique_mols_counts', 'dset_total_counts'

    def __init__(self):
        self.dset_unique_mols_counts = Counter()
        self.dset_total_counts = Counter()
        self.radiuses = set()  # fun name for radii
        self.hashes = set()

    def update_counts(self, dset, count_in_mol):
        self.dset_unique_mols_counts[dset] += 1
        self.dset_total_counts[dset] += count_in_mol

    def add(self, radius, rdkhash):
        self.radiuses.add(radius)
        self.hashes.add(rdkhash)

    def __repr__(self):
        return (repr(self.radiuses) +
                repr(self.hashes) +
                repr(self.dset_unique_mols_counts) +
                repr(self.dset_total_counts))

    def merge(self, another):
        self.radiuses.update(another.radiuses)
        self.hashes.update(another.hashes)
        for counter in ('dset_unique_mols_counts', 'dset_total_counts'):
            for dset, count in getattr(another, counter).items():
                getattr(self, counter)[dset] += count

    def to_dict(self):
        d = {
            'radiuses': tuple(sorted(self.radiuses)),
            'rdkhashes': tuple(sorted(self.hashes)),
        }
        for counter in ('dset_unique_mols_counts', 'dset_total_counts'):
            for dset, count in getattr(self, counter).items():
                d[counter.replace('dset', dset)] = count
        return d


def collect_fpt_stats(start=0, step=1, stop=None, save_each=1000, dset='lab', fcfp=False, use_tqdm=True):
    mc = MalariaCatalog()
    s2rh = defaultdict(radius_hash)
    mols = mc.lab() if dset == 'lab' else mc.unl() if dset == 'unl' else mc.scr()
    mols = mols[start:stop:step]
    shard = 0
    ensure_dir(FPT_STATS_SHARDED_DIR)
    for i, molid in enumerate(tqdm.tqdm(mols, unit=' mols', disable=not use_tqdm)):
        dest_file = op.join(FPT_STATS_SHARDED_DIR, '%s_%s_%r_%r_%r_%r.pickle' % (
            ('fcfp' if fcfp else 'ecfp'), dset, start, stop, step, shard))
        if not use_tqdm and i > 0 and i % 1000:
            print(op.basename(dest_file), '%d of %d' % (i, len(mols)))
        if op.isfile(dest_file):
            continue
        try:
            fpt = morgan_fingerprint(mc.molid2mol(molid), max_radius=200, fcfp=fcfp)
            for substruct_smiles, crhs in fpt.items():
                info = s2rh[substruct_smiles]
                info.update_counts(dset, len(crhs))
                for (center, radius, rdkhash) in crhs:
                    s2rh[substruct_smiles].add(radius, rdkhash)
        except Exception:
            print('WARNING: problem with %s, skipping' % molid)
        finally:
            if i > 0 and i % save_each == 0 or i == (len(mols) - 1):
                pd.to_pickle(s2rh, dest_file)
                s2rh = defaultdict(radius_hash)
                shard += 1


def collect_all_fpt_stats(num_jobs=12, dsets=('lab', 'unl', 'scr'), fcfps=(True, False)):
    joblib.Parallel(n_jobs=num_jobs)(
        joblib.delayed(collect_fpt_stats)
        (start, stop=None, step=num_jobs, dset=dset, fcfp=fcfp, save_each=5000)
        for fcfp, dset, start in product(fcfps, dsets, range(num_jobs))
    )


def pickle_files(path=FPT_STATS_SHARDED_DIR, dset='lab', fpt='ecfp'):
    return natsorted(glob.glob(op.join(path, '%s_%s_*.pickle' % (fpt, dset))))


def merge_fpt_stats_pickles(start=0, stop=100, path=FPT_STATS_SHARDED_DIR, dset='lab', fpt='ecfp'):

    def merge_fpt_stats_dict(fpt_stats_dest, fpt_stats_src):
        for smiles, stats in fpt_stats_src.items():
            if smiles not in fpt_stats_dest:
                fpt_stats_dest[smiles] = stats
            else:
                fpt_stats_dest[smiles].merge(stats)
        return fpt_stats_dest

    def merge_pickles(pickles):
        fpt_stats = None
        for pickle in tqdm.tqdm(pickles, unit='pickles'):
            if fpt_stats is None:
                fpt_stats = pd.read_pickle(pickle)
            else:
                fpt_stats = merge_fpt_stats_dict(fpt_stats, pd.read_pickle(pickle))
        return fpt_stats or {}

    if fpt is None or fpt == 'all':
        fpt = '*'

    return merge_pickles(pickle_files(path, dset, fpt)[start:stop])


def fpt_stats_df(dset='lab', fpt='ecfp',
                 start=None, stop=None,
                 path=FPT_STATS_DIR, recreate=False):

    def fptstats2df(fpt_stats):
        rows = []
        for sub, stats in tqdm.tqdm(fpt_stats.items()):
            d = stats.to_dict()
            d['sub'] = sub
            rows.append(d)
        df = pd.DataFrame(rows)

        col_order = [
            'sub',                       # Substructure smiles
            'radiuses',                  # Radii - in general, there should be just one radius per sub
            'rdkhashes',                 # (unfolded) hashes assigned by rdkit sub
            'lab_total_counts',          # Number of times the substructure appears in lab
            'lab_unique_mols_counts',    # Number of molecules the substructure appears in lab
            'unl_total_counts',          # Number of times the substructure appears in unl
            'unl_unique_mols_counts',    # Number of molecules the substructure appears in unl
            'scr_total_counts',          # Number of times the substructure appears in scr
            'scr_unique_mols_counts',    # Number of molecules the substructure appears in scr
        ]
        cols = [col for col in col_order if col in df.columns]
        cols += [col for col in df.columns if col not in col_order]

        return df[cols]

    dest_pickle = op.join(ensure_dir(path), '%s_%s_%r_%r.df.pickle' % (fpt, dset, start, stop))
    if not recreate:
        try:
            return pd.read_pickle(dest_pickle)
        except IOError:
            pass
    stats = merge_fpt_stats_pickles(dset=dset, fpt=fpt, start=start, stop=stop)
    df = fptstats2df(stats)
    pd.to_pickle(df, dest_pickle)
    return df


def num_or_zero(x):
    if x != x or not x:
        return 0
    return x


def merge_stats_dfs(left, right):

    # Let pandas do the merge for us
    df = pd.merge(left, right, how='outer', on='sub')

    # Collapse "tuple" columns
    tuple_columns = ('rdkhashes', 'radiuses')
    for col_name in tuple_columns:
        vals = []
        for left_val, right_val in zip(df[col_name + '_x'], df[col_name + '_y']):
            if not isinstance(left_val, tuple):
                left_val = ()
            if not isinstance(right_val, tuple):
                right_val = ()
            vals.append(tuple(set(left_val + right_val)))
        df[col_name] = vals
        del df[col_name + '_x']
        del df[col_name + '_y']

    # Collapse count columns
    dsets = 'lab', 'unl', 'scr'
    countss = 'total_counts', 'unique_mols_counts'

    for dset, count in product(dsets, countss):
        col_name = dset + '_' + count
        if (col_name + '_x') in df.columns:
            vals = []
            for left_val, right_val in df[[col_name + '_x', col_name + '_y']].itertuples(index=None, name=None):
                vals.append(num_or_zero(left_val) + num_or_zero(right_val))
            df[col_name] = vals
            del df[col_name + '_x']
            del df[col_name + '_y']

    return df


def merge_all_dfs(fpt='ecfp', shard_size=200, recache=False, add_ints=True):
    dest_file = op.join(FPT_STATS_DIR, '%s_all.df.pickle' % fpt)
    if not recache:
        try:
            print('Reading cache...')
            return pd.read_pickle(dest_file)
        except IOError:
            pass
    print('Recreating cache...')
    df = merge_stats_dfs(fpt_stats_df(dset='lab', fpt=fpt), fpt_stats_df(dset='unl', fpt=fpt))
    num_shards = int(np.ceil(len(pickle_files(dset='scr', fpt=fpt)) / shard_size))
    for start in range(num_shards):
        print('scr shard', start)
        df = merge_stats_dfs(df, fpt_stats_df(dset='scr', fpt=fpt,
                                              start=start * shard_size,
                                              stop=start * shard_size + shard_size))
    if add_ints:
        for col in ('lab_total_counts', 'lab_unique_mols_counts',
                    'unl_total_counts', 'unl_unique_mols_counts',
                    'scr_total_counts', 'scr_unique_mols_counts'):
            if col in df.columns:
                df[col] = df[col].apply(lambda x: int(num_or_zero(x)))
        df['num_radii'] = df['radiuses'].apply(len)
        df['radius'] = df['radiuses'].apply(min)
        df['num_rdkhashes'] = df.rdkhashes.apply(len)

    pd.to_pickle(df, dest_file)
    return df


# df = merge_all_dfs(fpt='ecfp', recache=True, add_ints=True)
# df = merge_all_dfs(fpt='fcfp', recache=True, add_ints=True)
# exit(22)


def to_rkdhashes_df(df, fpt='ecfp', recompute=False):

    dest_file = op.join(FPT_STATS_DIR, 'rdkhashes-%s.df.pickle' % fpt)

    if not recompute:
        try:
            return pd.read_pickle(dest_file)
        except IOError:
            pass

    # hash -> radius, num_sub, num_lab_mols, num_unl_mols, num_scr_mols
    def arr():
        return [0] * 9
    h2s = defaultdict(arr)

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), unit='sub'):
        for h in row['rdkhashes']:
            hrow = h2s[h]
            hrow[0] = h
            hrow[1] = min(row.radiuses)
            hrow[2] += 1
            if 'lab_unique_mols_counts' in row:
                hrow[3] += row['lab_unique_mols_counts']
            if 'lab_total_counts' in row:
                hrow[4] += row['lab_total_counts']
            if 'unl_unique_mols_counts' in row:
                hrow[5] += row['unl_unique_mols_counts']
            if 'unl_total_counts' in row:
                hrow[6] += row['unl_total_counts']
            if 'scr_unique_mols_counts' in row:
                hrow[7] += row['scr_unique_mols_counts']
            if 'scr_total_counts' in row:
                hrow[8] += row['scr_total_counts']
    df = pd.DataFrame(list(h2s.values()), columns=['rdkhash', 'radius', 'num_sub',
                                                   'num_lab_mols', 'num_lab_total',
                                                   'num_unl_mols', 'num_unl_total',
                                                   'num_scr_mols', 'num_scr_total'])
    df.to_pickle(dest_file)
    return df


#
# Create the dataframe of rdkit hashes, radius and diverse counts
#
# for fpt in ('ecfp', 'fcfp'):
#     df = merge_all_dfs(fpt=fpt)
#     del df['sub']
#     rdkhashes_df = to_rkdhashes_df(df, fpt=fpt, recompute=True)
#     print(rdkhashes_df.groupby('num_sub').size())
# print('Done')
# exit(22)

def munge_rdk_hashes_df(fpt='ecfp', columns=None, nthreads=1, recompute=False):

    cache_feather = op.join(FPT_STATS_DIR, '%s_rdkhashes_final.feather' % fpt)

    if not recompute:
        try:
            return feather.read_dataframe(cache_feather, columns=columns, nthreads=nthreads)
        except IOError:
            pass

    # Load the "rdkhashes from substructure smarts" big dataset
    # N.B. this can be a tad imprecise for radius (but should be correct 99.75% of the time)
    # It should be precise in the number of text substructures it maps to (but we should split lab, unl, scr)
    # Counts are not trustworthy, so we need to compute them apart
    # Revisit all the logic to get to "to_rdk_hashes_df" later
    print('Reading rdkhashes from substructure smarts matrix')
    # noinspection PyTypeChecker
    hdf = to_rkdhashes_df(None, fpt=fpt)[['rdkhash', 'radius', 'num_sub']].copy()
    hdf['radius'] = hdf['radius'].astype(np.uint32)
    hdf['num_sub'] = hdf['num_sub'].astype(np.uint32)
    # just in case, leave rdkhashes as int64 - at some point they might want to go that high

    # Add the mapping to the larger-design-matrix column
    # Read the scr matrix, to get the complete f2i info
    print('Reading the LARGE scr feature map (man, stream, sparseppit, do not use pickles...)')
    _, i2f, _, _ = rdkhash_feature_matrix(fpt=fpt, dset='scr', recompute=False)
    f2i = {f: i for i, f in enumerate(i2f)}
    print('Adding mapping to design matrix column')
    hdf['i'] = hdf['rdkhash'].apply(lambda h: f2i.get(h, -1)).astype(np.int32)
    # noinspection PyUnusedLocal
    _ = i2f = f2i = None

    # Add positive, negative counts
    # Read the train data.
    #   m2i: correspondence molid -> row
    #   f2i: correspondence hash -> column
    #   X: the design matrix (counts, int csr)
    #   y: the ground truth vector (binary dense)
    # WARNING: do not use rdkhash_feature_matrix here, as it brings amb molecules
    _, _, _, X, y = X_train_feats(fpt=fpt, dsets='lab', use_representatives=False, transductive=False)
    print('Adding positive/negative statistics')
    positives = np.nonzero(y)[0]
    negatives = np.nonzero(1 - y)[0]
    positive_counts = np.array(X[positives, :].sum(axis=0)).ravel()
    hdf['positive_counts'] = (hdf['i'].
                              apply(lambda i: positive_counts[i] if i < len(positive_counts) else 0).
                              astype(np.uint32))
    hdf['num_positive_mols'] = (hdf['i'].
                                apply(lambda i: 1 if i < len(positive_counts) and positive_counts[i] > 0 else 0).
                                astype(np.uint32))
    negative_counts = np.array(X[negatives, :].sum(axis=0)).ravel()
    hdf['negative_counts'] = (hdf['i'].
                              apply(lambda i: negative_counts[i] if i < len(negative_counts) else 0).
                              astype(np.uint32))
    hdf['num_negative_mols'] = (hdf['i'].
                                apply(lambda i: 1 if i < len(negative_counts) and negative_counts[i] > 0 else 0).
                                astype(np.uint32))
    # hdf['num_lab_mols'] = hdf['num_positive_mols'] + hdf['num_negative_mols']
    # hdf['lab_counts'] = hdf['positive_counts'] + hdf['negative_counts']
    # noinspection PyUnusedLocal
    _ = X = y = None

    # Add other counts
    print('Adding counts in other datasets')
    for dset in ('amb', 'unl', 'scr'):
        _, _, X, _ = rdkhash_feature_matrix(fpt=fpt, dset=dset, recompute=False)
        counts = np.array(X.sum(axis=0)).ravel()
        hdf['%s_counts' % dset] = (hdf['i'].
                                   apply(lambda i: counts[i] if i < len(counts) else 0).
                                   astype(np.uint32))
        hdf['num_%s_mols' % dset] = (hdf['i'].
                                     apply(lambda i: 1 if i < len(counts) and counts[i] > 0 else 0).
                                     astype(np.uint32))
        # noinspection PyUnusedLocal
        _ = X = None

    # Now go for the duplicated columns in the training (design) matrix
    print('Adding "representatives in train"')
    for transductive in (True, False):
        _, representatives, _ = find_rdkhash_dataset_duplicates(fpt=fpt, transductive=transductive)
        col_name = 'representative_%s' % ('transductive' if transductive else 'train')
        hdf[col_name] = (hdf['i'].
                         apply(lambda i: representatives[i] if i < len(representatives) else -1).
                         astype(np.int32))
        # noinspection PyStatementEffect
        _ = representatives = None

    print('Reorganizing columns')
    # Reorganize
    column_order = [
        'rdkhash',
        'i',
        'representative_train',
        'representative_transductive',
        'radius',
        'num_sub',
        'num_positive_mols',
        'num_negative_mols',
        'positive_counts',
        'negative_counts',
        # 'num_lab_mols',
        # 'lab_counts',
        'num_amb_mols',
        'amb_counts',
        'num_unl_mols',
        'unl_counts',
        'num_scr_mols',
        'scr_counts',
    ]
    hdf = hdf[column_order].rename(columns={'i': 'column',
                                            'representative': 'representative_column',
                                            'num_sub': 'num_smarts'})

    # Save to feather
    feather.write_dataframe(hdf, cache_feather)

    return hdf[columns] if columns is not None else hdf


# hdf = munge_rdk_hashes_df(fpt='ecfp', nthreads=4, recompute=True)
# hdf = munge_rdk_hashes_df(fpt='fcfp', nthreads=4, recompute=True)
# exit(0)


if __name__ == '__main__':

    # --- Analysis with rdkit hashes
    hdf = munge_rdk_hashes_df(fpt='ecfp', nthreads=4, recompute=False)
    hdf['num_lab_mols'] = hdf['num_positive_mols'] + hdf['num_negative_mols']
    print('Labelled and in train (not ambiguous/failed):', len(hdf.query('num_lab_mols > 0')))
    print('%s_6 and in train:' % 'ecfp', len(hdf.query('num_lab_mols > 0 and radius <= 3')))
    print(len(hdf.query('column >= 0')))
    print(len(hdf.query('num_lab_mols > 0')['rdkhash'].unique()))
    exit(0)

    # --- Analysis with smarts features

    df = merge_all_dfs(fpt='ecfp')
    df.info()

    print('Max radius used to generate a cansmi', df.radius.max())
    print('Max number or radii generating a sub:', df['num_radii'].max())
    print('Subs generated by more than one radius:\n', df.query('num_radii > 1'))

    print('Number of subs not in train:', len(df.query('lab_unique_mols_counts == 0')))
    print('Number of subs not in train (radius <= 3):',
          len(df.query('lab_total_counts == 0 and radius <=3')))
    print('Number of subs not in train (radius <= 2):',
          len(df.query('lab_total_counts == 0 <=2')))

    print('Number of subs not in train, but in unl:',
          len(df.query('lab_total_counts == 0 and unl_total_counts > 0')))
    print('Number of subs not in train, but in unl (radius <= 3):',
          len(df.query('lab_total_counts == 0 and radius <=3 and unl_total_counts > 0')))
    print('Number of subs not in train, but in unl (radius <= 2):',
          len(df.query('lab_total_counts == 0 and radius <=2 and unl_total_counts > 0')))

    print('Number of subs not in train, but in scr:',
          len(df.query('lab_total_counts == 0 and scr_total_counts > 0')))
    print('Number of subs not in train, but in unl (radius <= 3):',
          len(df.query('lab_total_counts == 0 and radius <=3 and scr_total_counts > 0')))
    print('Number of subs not in train, but in unl (radius <= 2):',
          len(df.query('lab_total_counts == 0 and radius <=2 and scr_total_counts > 0')))

    rdkhashes = list(chain.from_iterable(df['rdkhashes']))
    unique_rdkhashes = set(rdkhashes)
    colliding_rdkhashes = set(h for h, count in Counter(rdkhashes).items() if count > 1)
    df['num_rdkhashes'] = df.rdkhashes.apply(len)
    print('Max num of rdkhases collapsed to a cansmi:', df['num_rdkhashes'].max())
    print('Number of rdkhashes distribution:\n', df.groupby('num_rdkhashes').size())
    print('Number of rdkhashes:', len(rdkhashes))
    print('Number of unique rdkhashes:', len(unique_rdkhashes))
    print('Number of colliding rdkhashes:', len(colliding_rdkhashes))

    # Invert map from rdkit hashes to substructure smiles
    h2s = defaultdict(list)
    for sub, subh in df[['sub', 'rdkhashes']].itertuples(index=False, name=None):
        for h in subh:
            if h in colliding_rdkhashes:
                h2s[h].append(sub)

    for h, ss in take(2000, h2s.items()):
        print(ss)
        print(list(map(cansmartssmi, ss)))
        print('-----')

    # Perhaps here we should also already check for sets of hashes
    print('Number of cansmi features:', len(df))
    print('Number of rdkhashes:', df['rdkhashes'].apply(len).sum())
    print('Number of cansmi features (radius 3):', len(df.query('radius <= 3')))
    print('Number of rdkhashes (radius 3):', df.query('radius <= 3')['rdkhashes'].apply(len).sum())

    print('Cansmi features radius counts\n', df.groupby('radius').size())
    print('Single-atom features:', list(df.query('radius == 0')['sub']))

    exit(0)

    root = ensure_dir(DATA_DIR, 'substructure_voluntary_collisions_analysys')

    collapsers = (
        # partial(build_collapsing_features_info_df,
        #         matcher=has_query_query_match,
        #         mols_instantiator=unsanitized_mol_from_smiles,
        #         pattern_instantiator=None,
        #         dest_file=op.join(root, 'qqm_ums_None.pickle')),
        #
        # partial(build_collapsing_features_info_df,
        #         matcher=has_non_recursive_query_query_match,
        #         mols_instantiator=unsanitized_mol_from_smiles,
        #         pattern_instantiator=None,
        #         dest_file=op.join(root, 'nqqm_ums_None.pickle')),

        # partial(build_collapsing_features_info_df,
        #         matcher=has_substruct_match,
        #         mols_instantiator=unsanitized_mol_from_smiles,
        #         pattern_instantiator=None,
        #         dest_file=op.join(root, 'sm_ums_None.pickle')),

        # partial(build_collapsing_features_info_df,
        #         matcher=has_query_query_match,
        #         mols_instantiator=unsanitized_mol_from_smiles,
        #         pattern_instantiator=mol_from_smarts,
        #         dest_file=op.join(root, 'qqm_ums_smarts.pickle')),
        #
        # partial(build_collapsing_features_info_df,
        #         matcher=has_non_recursive_query_query_match,
        #         mols_instantiator=unsanitized_mol_from_smiles,
        #         pattern_instantiator=mol_from_smarts,
        #         dest_file=op.join(root, 'nqqm_ums_smarts.pickle')),
        #
        # partial(build_collapsing_features_info_df,
        #         matcher=has_substruct_match,
        #         mols_instantiator=unsanitized_mol_from_smiles,
        #         pattern_instantiator=mol_from_smarts,
        #         dest_file=op.join(root, 'sm_ums_smarts.pickle')),
        #

        partial(build_collapsing_features_info_df,
                matcher=has_query_query_match,
                mols_instantiator=mol_from_smarts,
                pattern_instantiator=mol_from_smarts,
                dest_file=op.join(root, 'qqm_smarts_smarts.pickle')),

        #
        # partial(build_collapsing_features_info_df,
        #         matcher=has_non_recursive_query_query_match,
        #         mols_instantiator=mol_from_smarts,
        #         pattern_instantiator=mol_from_smarts,
        #         dest_file=op.join(root, 'nqqm_smarts_smarts.pickle')),
        #
        # partial(build_collapsing_features_info_df,
        #         matcher=has_substruct_match,
        #         mols_instantiator=mol_from_smarts,
        #         pattern_instantiator=mol_from_smarts,
        #         dest_file=op.join(root, 'sm_smarts_smarts.pickle')),
    )

    for collapser in collapsers:
        print('=' * 80)
        print('START', collapser.keywords['dest_file'])
        df, num_atoms_distribution = collapser()
        df['num_total_mols'] = df['num_lab_mols'] + df['num_amb_mols'] + df['num_unl_mols'] + df['num_scr_mols']
        print('%d features (substructures) in lab + amb (remember, we removed amb mols from training)' % 2351460)
        print('%d features were used' % len(df))
        print('%d did not appear in a single non-ambiguous lab example (so were not used either)'
              % len(df.query('num_lab_mols == 0')))
        print('%d did appear only in amb (and not in not amb)' %
              len(df.query('num_lab_mols == 0 and num_amb_mols > 0')))
        print('%d did not appear in a single amb example' % len(df.query('num_amb_mols == 0')))
        print('%d did not appear in a single unl example' % len(df.query('num_unl_mols == 0')))
        print('%d did not appear in a single scr example' % len(df.query('num_scr_mols == 0')))
        print('%d only happened in a single molecule' % len(df.query('num_total_mols == 1')))
        print('%d features are unique (no collapse of identical)' % len(df.query('num_substructures == 1')))
        print('%d features are unique (after collapsing identical)' % len(df.query('num_different_substructures == 1')))
        print('%d groups have "total disjoint" collisions' % len(df.query('num_groups > 1')))
        # But better analysis can be carried; it is intrincate (look at roots and "intermediate roots" and leaves)
        print(df.columns)
        print('END', collapser.keywords['dest_file'])

        # --- Let's quickly explore the number of atoms / feature length distribution
        num_atoms_distribution = num_atoms_distribution[0:last_non_zero(num_atoms_distribution) + 1]
        print('Max substructure length: %d' % len(num_atoms_distribution))
        mean, std = mean_std_from_counts(num_atoms_distribution)
        expanded = expand(num_atoms_distribution)
        print(np.mean(expanded), np.std(expanded))
        print('Mean substructure length: %.2f +/- %.2f' % (mean, std))
        sns.distplot(np.arange(len(num_atoms_distribution)),
                     bins=100,
                     hist_kws={'weights': num_atoms_distribution})
        plt.show()

    print('DONE')

#
#  2329566 In lab
#    21894 In amb but not in lab
#  2351460 In lab + amb (we used this as the vector, 0-ing "all dupes but 1";
#                        also amb-only were, obviously, 0, as we did not use these mols on training)
#  1265410 Used (not removing amb, but zeroing all features with equal "column vector in all mols matrix"
#                except for one representative of each group)
#    10597 In unl but not in lab + amb
#  2362057 In lab + amb + unl
# 39209611 In scr but not in lab + amb + unl
# 41571668 In lab+amb+unl+scr
#

# Remember this nifty blogpost by Bulatov, include relevant graph theory comments
#   https://medium.com/@yaroslavvb/fitting-larger-networks-into-memory-583e3c758ff9

#
# TODO: great fast streaming + random access via something like jagged
#   - molids apart
#   - jagged for sparse + counts
#   - compute stats online
#
