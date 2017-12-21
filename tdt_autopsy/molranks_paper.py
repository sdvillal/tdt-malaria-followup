from __future__ import print_function, division
from future.utils import string_types

import os.path as op

import pandas as pd
from feather import write_dataframe, read_dataframe
import numpy as np
from scipy.stats import rankdata

from ccl_malaria.rdkit_utils import to_rdkit_mol
from rdkit.Chem import MolToSmiles
from tdt_autopsy.config import (STJUDE_FLO_GREG_ANNOTATED_CSV,
                                CCL_EXPERIMENTS_DIR,
                                PAPER_NOT_INTRAIN_CSV,
                                PAPER_INTRAIN_CSV)
from tdt_autopsy.eval import read_ccl_final_scores, read_sg_scores


# --- Canonical smiles used
# Note that there might be divergences as we are using later versions of rdkit than these originally used

def cansmi(smi):
    mol = to_rdkit_mol(smi, molid=None, sanitize=True, toPropertyMol=False, to2D=False, to3D=False)
    return MolToSmiles(mol,
                       isomericSmiles=True,
                       kekuleSmiles=False,
                       canonical=True,
                       allBondsExplicit=False,
                       allHsExplicit=False)


# --- Read the tables as shown in the paper

def read_paper_table():

    # --- Read table 5 (assay compounds not in train + reference compounds)

    not_in_train_columns = [
        'identifier',            # 'Identifier'
        'ec50',                  # 'EC 50 \n[μM]'
        'assay_label',           # 'Score'
        'proposed_by_workflow',  # 'Rank \n(1000)Proposed  \nby Workflow 1'
        'rank10000wf1',          # 'Rank (top \n10000) \nWorkflow 1'
        'rank1000wf2',           # 'Rank (top \n1000) \nWorkflow 2'
        'known_datasets'         # 'Known Datasets'
    ]
    not_in_train_df = pd.read_csv(PAPER_NOT_INTRAIN_CSV, names=not_in_train_columns, skiprows=1, index_col=None)

    # Fix proposed_by_workflow (originally strikenthrough)
    def fix_proposed_by_workflow_table5(x):
        x = str(x).lower().strip()
        if x == '2':
            return '2'
        if x.endswith(',2'):
            return '1,2'
        return '1'
    not_in_train_df['proposed_by_workflow'] = (not_in_train_df['proposed_by_workflow'].
                                               apply(fix_proposed_by_workflow_table5))

    # --- Read supplementary 1 (assay compounds in train)

    in_train_columns = [
        'identifier',            # 'Identifier'
        'ec50',                  # 'EC50\n[μM]'
        'assay_label',           # 'Score
        'proposed_by_workflow',  # 'Proposed by Workflow'
        'rank10000wf1',          # 'Rank \n(top 10000)\nWorkflow \n1'
        'rank1000wf2',           # 'Rank \n(top 1000)\nWorkflow \n2'
        'train_label',           # 'HTS screen',  => HTS is the bioassay to test many compounds, not to confuse with VS
        'known_datasets'         # 'Known Datasets'
    ]
    in_train_df = pd.read_csv(PAPER_INTRAIN_CSV, names=in_train_columns, skiprows=1, index_col=None)

    # --- Merge

    # Does the paper report this compound to be in train?
    # Somewhat redundant with "train_label"
    not_in_train_df['paper_in_train'] = False
    in_train_df['paper_in_train'] = True

    # Concat
    paper_table = pd.concat([not_in_train_df, in_train_df])

    # Remove hyphen in ambi-guous
    def fix_train_label(label):
        if not isinstance(label, string_types):
            return label
        if label.lower().startswith('ambi'):
            return 'ambiguous'
        return label.lower()
    paper_table['train_label'] = paper_table['train_label'].apply(fix_train_label)

    # Remove spurious new lines in "known_datasets"
    paper_table['known_datasets'] = paper_table['known_datasets'].str.replace('\n', '')

    # Normalize identifier
    paper_table['identifier'] = paper_table['identifier'].apply(lambda x: x.strip())

    # Amodiaquine is a reference compound that actually was in train
    paper_table.loc[paper_table['identifier'] == 'SJ000110703', 'paper_in_train'] = True

    # Normalize missing rankings
    def normalize_missing_rankings(x):
        try:
            return int(x)
        except ValueError:
            return None

    paper_table['rank10000wf1'] = paper_table['rank10000wf1'].apply(normalize_missing_rankings)
    paper_table['rank1000wf2'] = paper_table['rank1000wf2'].apply(normalize_missing_rankings)

    order = [
        'identifier',
        'paper_in_train',
        'proposed_by_workflow',
        'rank10000wf1',
        'rank1000wf2',
        'train_label',
        'assay_label',
        'ec50',
        'known_datasets',
    ]

    return paper_table[order]


# --- Read the assay file (biological tests results)


def read_assay_table():

    # --- Read the CSV
    columns = [
        ('identifier', 'SAMPLE'),
        ('cansmi_original', 'Molecule_RDKitCanonical'),
        ('rank_ccl', 'RankSantiagoFloriane'),
        ('rank_sg', 'RankSereina'),
        ('assay_label_original', 'cScore'),
        ('rSquared', 'rSquared'),
        ('ec50_original', 'ec50'),
        ('ec50_l', 'ec50_l'),
        ('ec50_u', 'ec50_u'),
        ('emolecules_id', 'EMOLECULES_ID'),
        ('supplier', 'supplier'),
        ('catalog_number', 'CATALOG_NUMBER'),
        ('chemblid', 'CHEMBLID'),
        ('flo_labelled_active_chembl21', 'Malaria (as of ChEMBL_21)'),
        ('flo_in_train', 'In training'),
        ('flo_similarity_to_train', 'Similar to training'),
        ('flo_interesting', 'Interesting'),
    ]

    assay_file = STJUDE_FLO_GREG_ANNOTATED_CSV
    assay_df = pd.read_csv(assay_file, names=[c for c, _ in columns], index_col=None, skiprows=1)

    # Normalize identifier
    assay_df['identifier'] = assay_df['identifier'].apply(lambda x: x.strip().split('-')[0])

    # Make in_training bool (plus Flo did not check some, essentially the inactive ones)
    def normalize_in_train(x):
        if not isinstance(x, string_types):
            return None
        return x.strip().lower() == 'yes'
    assay_df['flo_in_train'] = assay_df['flo_in_train'].apply(normalize_in_train)

    # --- Quick check of canonical smiles idempotence
    these_would_be_a_problem = assay_df['cansmi_original'] != assay_df['cansmi_original'].apply(cansmi)
    # noinspection PyUnresolvedReferences
    assert not these_would_be_a_problem.any(), 'canonical roundtrip fails'

    # Order and filter out columns
    column_order = [
        'identifier',
        'cansmi_original',
        'rank_ccl',
        'rank_sg',
        'assay_label_original',
        'ec50_original',
        'chemblid',
        'flo_labelled_active_chembl21',
        'flo_in_train',
        'flo_similarity_to_train',
        'flo_interesting',
    ]

    return assay_df[column_order]


def read_full_submission_screening_scores():

    cache_path = op.join(CCL_EXPERIMENTS_DIR,
                         'fusion201712',
                         'cache-screening-non_calibrated-onlylastfold-linr.feather')

    try:
        return read_dataframe(cache_path)
    except IOError:

        # Read CSV
        submissions_df = read_ccl_final_scores(
            fusion201712=True,    # New stacking (may change due to different linear regression)
            screening=True,       # For the screening dataset (i.e.)
            calibrated=False,     # Not calibrated
            average_folds=False,  # With bug
            linr_stacker=True     # Linear regression stacker
        )
        submissions_df = submissions_df.rename(columns={
            'smiles': 'isosmiles',
            'id': 'original_id',
            'score': 'score',
        })

        # Compute rankings
        tie_breaking_methods = ['average']  # , 'min', 'max', 'dense', 'ordinal'
        for method in tie_breaking_methods:
            submissions_df['ranking_%s' % method] = rankdata(-submissions_df['score'], method=method)

        # Compute canonical SMILES (N.B. this shows a warning here and there)
        from tqdm import tqdm
        tqdm.pandas(desc='canonical smiles computation')
        submissions_df['cansmi_from_isosmiles'] = submissions_df['isosmiles'].progress_apply(cansmi)
        submissions_df = submissions_df.set_index('cansmi_from_isosmiles', drop=False)
        # Remove redundant isosmiles
        del submissions_df['isosmiles']

        # Write cache
        write_dataframe(submissions_df, cache_path)

        # Done
        return submissions_df


if __name__ == '__main__':

    # --- Merge original table and paper table

    paper_df = read_paper_table()
    assay_df = read_assay_table()

    assay_df = pd.merge(paper_df, assay_df, how='inner', on='identifier')
    assert len(assay_df) == 114

    columns = [
        'identifier',
        'proposed_by_workflow',
        'rank10000wf1', 'rank_sg',                             # These two cannot coincide
                                                               # (we can use rank_sg later to check correct merging)
        'rank1000wf2', 'rank_ccl',                             # These two must coincide
        'train_label', 'assay_label', 'assay_label_original',  # These must not be contradictory
        'ec50', 'ec50_original',                               # These two must coincide
        'known_datasets',
        # Canonical smiles reported in the original assay file
        'cansmi_original',
        # Flo exploring manually chembl and pubchem
        'chemblid',
        'flo_labelled_active_chembl21',
        'paper_in_train', 'flo_in_train',  # These two must coincide (or at least flo be a strict subsets)
        'flo_similarity_to_train',
        'flo_interesting'
    ]

    assay_df = assay_df[columns]

    np.testing.assert_array_equal(assay_df['rank1000wf2'], assay_df['rank_ccl'])
    del assay_df['rank1000wf2']

    np.testing.assert_array_equal(assay_df['ec50'], assay_df['ec50_original'])
    del assay_df['ec50_original']

    flo_checked_in_train = assay_df.query('flo_in_train in [True, False]')
    np.testing.assert_array_equal(flo_checked_in_train['paper_in_train'],
                                  flo_checked_in_train['flo_in_train'])
    del assay_df['paper_in_train']  # Did they double check these that Flo did not check (they just wrote False)?

    # --- Prepare our submissions dataframe

    submissions_df = read_full_submission_screening_scores()
    columns = ['original_id', 'score', 'ranking_average', 'cansmi_from_isosmiles']

    #
    # We can merge:
    #
    #  - Using canonical smiles, but I do not trust it fully, as we know for a fact there are mistakes
    #    (we are using a different rdkit version than the original).
    #
    #  - Using the rankings: more involved, but well done is trustworthier.
    #      1. Get the original submissions, find rankings, map to original screening IDs
    #      2. Merge with the rankings in the assays table
    #  The following code prepares to merge using the reported rankings, by adding the
    #  original rankings to the submissions dataframe.
    #

    # Unfortunately this is not working with Sereina & Gregs submission...
    # ... I won't look for the cause and assume there is no error in the paper tables
    sg_df = read_sg_scores(screening=True, modern=False)
    sg_df['ranking_sg_submission'] = rankdata(-sg_df['score'], method='ordinal')
    submissions_df = pd.merge(submissions_df, sg_df[['id', 'ranking_sg_submission']],
                              left_on='original_id', right_on='id', how='left')

    ccl_df = read_ccl_final_scores(fusion201712=False,
                                   screening=True,
                                   calibrated=False,
                                   average_folds=False,
                                   linr_stacker=True)
    ccl_df['ranking_ccl_submission'] = rankdata(-ccl_df['score'], method='ordinal')
    submissions_df = pd.merge(submissions_df, ccl_df[['id', 'ranking_ccl_submission']],
                              left_on='original_id', right_on='id', how='left')

    # Merge all
    merged_df = pd.merge(assay_df, submissions_df,
                         left_on='cansmi_original', right_on='cansmi_from_isosmiles',
                         how='left')

    columns = [
        'identifier',
        'proposed_by_workflow',
        'ranking_average',
        'rank10000wf1',
        'rank_sg',
        'ranking_sg_submission',
        'rank_ccl',
        'ranking_ccl_submission',
        'train_label',
        'assay_label',
        'assay_label_original',
        'ec50',
        'known_datasets',
        'cansmi_original',
        'cansmi_from_isosmiles',
        'chemblid',
        'flo_labelled_active_chembl21',
        'flo_in_train',
        'flo_similarity_to_train',
        'flo_interesting',
        'score',
        'original_id',
        'id_x',
        'id_y',
    ]
    merged_df = merged_df[columns]
    assert(len(merged_df) == 114)
