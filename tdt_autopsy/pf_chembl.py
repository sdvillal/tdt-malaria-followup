# coding=utf-8
"""Queries to chembl related to Plasmodium falciparum, activities munging."""
from __future__ import print_function, division
from future.utils import string_types

import gzip

import os.path as op
import pandas as pd
from functools import partial

from ccl_malaria.molscatalog import MalariaCatalog
from rdkit import Chem
from minioscail.common.misc import ensure_dir
from whatami import what2id
from tdt_autopsy.config import CHEMBL_DEST_DIR, STJUDE_CSV, STJUDE_FLO_GREG_ANNOTATED_CSV
from sqlalchemy import create_engine

# Select all full-organism (aka functional?) assays with plasmodium
# falciparum as target. We can decide what was a hit later on.
#
# Note that most probably we are not interested in most of them,
# and could reduce this to strictly match "plasmodium falciparum".
#
# This should be alike to querying by organism using chembl web/REST
#   https://www.ebi.ac.uk/chembl/
#   "falciparum" -> target -> click -> export bioactivities
#
# FAQ with further examples: https://www.ebi.ac.uk/chembl/faq

_PF_ACTIVITIES_QUERY = """
SELECT act.*,
       cpds.standard_inchi, cpds.standard_inchi_key, cpds.canonical_smiles
FROM assays AS ass,
     activities as act,
     compound_structures as cpds,
     target_dictionary as td
WHERE ass.assay_organism ILIKE '%%plasmodium falciparum%%' AND
      td.tid = ass.tid AND td.target_type = 'ORGANISM' AND
      ass.assay_id = act.assay_id AND
      cpds.molregno = act.molregno
"""

_CHEMBL_VERSIONS = ('17',    # ChEMBL during the competition
                    '20',    # ChEMBL during in-vitro testing
                    '22_1')  # Current


def _query_chembl_for_activities_against_plasmodium_falciparum(chembl_version='20'):
    chembl_version = chembl_version.split('_')[0]
    url = 'postgresql://chembl:chembl@localhost:5432/chembl_%s' % chembl_version
    return pd.read_sql(_PF_ACTIVITIES_QUERY, con=create_engine(url))


def pf_activities(src_dir=CHEMBL_DEST_DIR, chembl_version='22_1', force=False):
    """
    Returns a pandas dataframe with all bioactivities for assays against plasmodium falciparum in chembl.

    The dataframe has these columns (refer to the ChEMBL schema is something is not clear):
      'activity_id', 'assay_id', 'doc_id', 'record_id', 'molregno',
      'standard_relation', 'published_value', 'published_units',
      'standard_value', 'standard_units', 'standard_flag',
      'standard_type', 'activity_comment', 'published_type',
      'data_validity_comment', 'potential_duplicate', 'published_relation',
      'pchembl_value', 'standard_inchi', 'standard_inchi_key', 'canonical_smiles'
    """
    csv = op.join(ensure_dir(src_dir), 'pf_bioactivities_chembl_%s.csv.gz' % chembl_version)
    if not op.isfile(csv) or force:
        df = _query_chembl_for_activities_against_plasmodium_falciparum(chembl_version=chembl_version)
        df.to_csv(csv, index=False, compression='gzip')
    return pd.read_csv(csv, low_memory=False)


# --- Labellers (conversion assay -> active/inactive)
#
# Very ad-hoc.
#
# Standard units with more than 100 activities in Chembl 22_1
# ----------
#   st = pf_activities().groupby('standard_type').size().sort_values(ascending=False)
#   print(st[st > 100])
# standard_type
# Potency              302218
# Inhibition            43114
# IC50                  37004
# EC50                  21100
# XC50                  13547
# Ratio IC50             1790
# Activity               1776
# IC90                   1489
# ED50                   1208
# MIC                     743
# Ratio                   616
# AbsAC35                 469
# Growth Inhibition       400
# GI                      367
# Ratio EC50              339
# Relative activity       263
# Log RA                  211
# PGI                     156
# ----------


def potency_labeller(df, threshold_nM=2000):
    df = df.query('standard_type == "Potency"').copy()
    df['active'] = df.standard_value < threshold_nM
    return df


def inhibition_labeller(df, threshold_pct=50):
    df = df.query('standard_type == "Inhibition" and '
                  'standard_units == standard_units').copy()
    # df.standard_units.unique() -> ['%' nan]
    df['active'] = df.standard_value > threshold_pct
    return df


def ic50_labeller(df, threshold_nM=2000):
    df = df.query('standard_type == "IC50"').copy()
    df['active'] = df.standard_value < threshold_nM
    return df


def ec50_labeller(df, threshold_nM=2000):
    df = df.query('standard_type == "EC50" and '
                  'standard_units == "nM"').copy()
    # print(df.standard_units.unique())
    # => ['nM' 'ug.mL-1' nan '(mg/kg)/day' 'ug']
    df['active'] = df.standard_value < threshold_nM
    return df


def xc50_labeller(df, threshold_nM=2000):
    # A 2uM threshold is proposed in the original assay.
    #   https://pubchem.ncbi.nlm.nih.gov/bioassay/2305#section=Data-Table
    xc50 = df.query('standard_type == "XC50" and '
                    'standard_units == standard_units').copy()
    xc50['active'] = xc50.standard_value < threshold_nM
    return xc50


LABELLERS = {
    'naive': {
        'Potency': potency_labeller,
        'Inhibition': inhibition_labeller,
        'IC50': ic50_labeller,
        'EC50': ec50_labeller,
        'XC50': xc50_labeller,
    },
    'strict': {
        'Potency': partial(potency_labeller, threshold_nM=1000),
        'Inhibition': partial(inhibition_labeller, threshold_pct=75),
        'IC50': partial(ic50_labeller, threshold_nM=1000),
        'EC50': partial(ec50_labeller, threshold_nM=1000),
        'XC50': partial(xc50_labeller, threshold_nM=1000),
    },
    'relaxed': {
        'Potency': partial(potency_labeller, threshold_nM=10000),
        'Inhibition': inhibition_labeller,
        'IC50': partial(ic50_labeller, threshold_nM=10000),
        'EC50': partial(ec50_labeller, threshold_nM=10000),
        'XC50': partial(xc50_labeller, threshold_nM=10000),
    }
}


def _activities2classes(df, labellers, pct_active=1, pct_inactive=0):
    """Returns a 2-column (standard_inchi, active) dataframe."""
    # Look at cricket chorus for much more sophisticated options,
    # like dubious removal and sources agreement checking

    # Apply each individual labeller, merge
    df = pd.concat([labeller(df) for _, labeller in labellers.items()])

    # Compute the percentage of measurements assays each compound is active
    activeness = df.groupby('standard_inchi')['active'].mean().reset_index()

    # Remove "ambiguous" compounds
    unambiguous = activeness.query(
        'active <= {pct_inactive} or active >= {pct_active}'.format(
            pct_inactive=pct_inactive,
            pct_active=pct_active)
    ).copy()

    # Active -> bool
    unambiguous['active'] = unambiguous['active'] >= pct_active

    return unambiguous


def attribute_active(labellers='naive', pct_active=1, pct_inactive=0, force=False):
    csv_file = op.join(ensure_dir(CHEMBL_DEST_DIR, 'classification', 'labellers=%r' % labellers),
                       'pf_activities.csv.gz')
    if not op.isfile(csv_file) or force:
        if isinstance(labellers, string_types):
            labellers = LABELLERS[labellers]
        df = None
        a2c = partial(_activities2classes,
                      labellers=labellers,
                      pct_active=pct_active, pct_inactive=pct_inactive)
        for chembl_version in _CHEMBL_VERSIONS:
            chembl_df = a2c(pf_activities(chembl_version=chembl_version))
            chembl_df.rename(columns={'active': 'active_' + chembl_version}, inplace=True)
            if df is None:
                df = chembl_df
            else:
                df = pd.merge(df, chembl_df, on='standard_inchi', how='outer')
        df = df.set_index('standard_inchi', drop=True)
        with gzip.open(csv_file, 'wt') as writer:
            df.to_csv(writer)
        with open(op.splitext(csv_file)[0] + '.info', 'wt') as writer:
            writer.write(what2id(a2c) + '\n')
            writer.write('ChEMBLs: %s' % ','.join(_CHEMBL_VERSIONS))
    return pd.read_csv(csv_file, index_col='standard_inchi', low_memory=False)


naive_active = attribute_active
stricter_active = partial(attribute_active, labellers='strict')
relaxed_active = partial(attribute_active, labellers='relaxed')


def label_all(force=False):
    # Generate active / inactive CSVs
    for labellers in LABELLERS:
        attribute_active(labellers=labellers,
                         pct_inactive=0, pct_active=1,
                         force=force)


def nasty_smi2inchi(smi, log_level=4):
    # I do not know if inchi logLevel works as it should, but I feel not
    # Also I feel the logging machinery is not well exposed in python land
    try:
        Chem.inchi.logger.setLevel(log_level)  # 4 => critical?
        return Chem.MolToInchi(Chem.MolFromSmiles(smi))
    finally:
        Chem.inchi.logger.setLevel(0)  # Nasty nasty, how to ask for old loglevel


def nasty_inchi2smi(inchi, log_level=4):
    # I do not know if inchi logLevel works as it should, but I feel not
    # Also I feel the logging machinery is not well exposed in python land
    try:
        Chem.inchi.logger.setLevel(log_level)  # 4 => critical?
        return Chem.MolToSmiles(Chem.MolFromInchi(inchi))
    finally:
        Chem.inchi.logger.setLevel(0)  # Nasty nasty, how to ask for old loglevel


def inchi2inchikey(inchi):
    return Chem.InchiToInchiKey(inchi)


def tidy_stjude_flo():

    # Index(['SAMPLE', 'Molecule_RDKitCanonical', 'RankSantiagoFloriane',
    #    'RankSereina', 'cScore', 'rSquared', 'ec50', 'ec50_l', 'ec50_u',
    #    'EMOLECULES_ID', 'supplier', 'CATALOG_NUMBER', 'CHEMBLID',
    #    'Malaria (as of ChEMBL_21)', 'In training', 'Similar to training',
    #    'Interesting'],

    stj_flo = pd.read_csv(STJUDE_FLO_GREG_ANNOTATED_CSV)
    stj_flo['in_chembl_21'] = stj_flo['Malaria (as of ChEMBL_21)'].apply()


if __name__ == '__main__':

    # df = naive_active()
    df = relaxed_active()
    # df = stricter_active()
    print(df.columns)
    print('%d compounds from ChEMBL (%d active)' % (len(df), df['active_22_1'].sum()))

    stj_df = pd.read_csv(STJUDE_CSV)

    stj_df['standard_inchi'] = stj_df['Molecule_RDKitCanonical'].apply(nasty_smi2inchi)
    stj_df['active'] = stj_df['cScore'].apply(lambda x: x.endswith('Active'))

    stj_df = pd.merge(stj_df, df, left_on='standard_inchi', right_index=True)
    stj_df = stj_df[['standard_inchi', 'Molecule_RDKitCanonical', 'active_17', 'active_20', 'active_22_1', 'active']]
    pd.set_option('display.max_colwidth', 0)
    stj_df.to_html('/home/santi/mola.html')

    exit(22)

    cat = MalariaCatalog()

    print(cat.num_lab())
    cat.lab()
    # print(cat.lab())
    # from joblib import Parallel, delayed
    # inchis = Parallel(n_jobs=8)(delayed(smi2inchi)(cat.molid2smiles(molid))
    #                             for molid in cat.lab())

    mols = [cat.molid2mol(molid) for molid in cat.lab()]
    inchis = [Chem.MolToInchi(mol) for mol in mols if mol is not None]
    inchis_stj = set(stj_df['standard_inchi'])
    print(inchis_stj & set(inchis))
    print(len(inchis))

    with open('/home/santi/inchis.txt', 'wt') as writer:
        for inchi in inchis:
            writer.write(inchi + '\n')

# SJ000243361-2,Clc1ccccc1Nc1nc(NCc2ccccc2)c2ccccc2n1,450,,A Active,1,1.9,1.8,2,3021339,ENAMINE,Z56854609,CHEMBL1198276,yes (StJude),yes,yes,no
# SJ000180705-2,COc1ccc(CCNC(=S)Nc2ccccc2Cl)cc1OC,589,425,A Active,1,0.59,0.55,0.61,2194279,CHEMBRIDGE,7202723,CHEMBL589480,yes (StJude),yes,yes,no
# SJ000171307-4,CCOc1ccc(Nc2nc(NCc3ccccc3)c3ccccc3n2)cc1,310,,A Active,1,0.52,0.47,0.59,6296699,ENAMINE,Z56833777,CHEMBL592582,yes (StJude and Novartis),yes,yes,no
# SJ000011369-3,CCOc1cc(C2C(C(=O)OC(C)C)=C(C)NC3=C2C(=O)CC(c2ccc(Cl)cc2)C3)ccc1O,534,,A Active,0.99,2.3,2.2,2.6,1323284,CHEMBRIDGE,5710333,CHEMBL600641,yes (StJude),yes,yes,no

# https://www.ebi.ac.uk/chembl/compound/inspect/CHEMBL600641

# --- DB stuff
#
#  * Install postgresql:
#    Arch: https://wiki.archlinux.org/index.php/PostgreSQL
#
#   sudo pacman -S postgresql  # Install postgresql
#   sudo passwd postgres       # Change password for new "postgresql" user
#   sudo -i -u postgres        # Switch to postgres user
#   initdb --locale $LANG -E UTF8 -D '/var/lib/postgres/data'  # Initialize DB cluster
#   createuser --interactive  # I created the "admin" user, with admin privileges
#
#  * Prepare CHEMBL databases
#   createdb chembl_17
#   createdb chembl_20
#   createdb chembl_22
#   psql
#   CREATE USER chembl WITH PASSWORD 'chembl';
#   GRANT ALL PRIVILEGES ON DATABASE chembl_17 to chembl;
#   GRANT ALL PRIVILEGES ON DATABASE chembl_20 to chembl;
#   GRANT ALL PRIVILEGES ON DATABASE chembl_22 to chembl;
#   \connect chembl_17
#   GRANT USAGE ON SCHEMA public to chembl;
#   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chembl;
#   \connect chembl_20
#   GRANT USAGE ON SCHEMA public to chembl;
#   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chembl;
#   \connect chembl_22
#   GRANT USAGE ON SCHEMA public to chembl;
#   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chembl;
# See too:
#   http://stackoverflow.com/questions/13497352/error-permission-denied-for-relation-tablename-on-postgres-while-trying-a-selec
#
#  * Download and install chembl:
#    https://www.ebi.ac.uk/chembl/downloads
#    As written in CHEMBL/INSTALL
#      psql --host=HOST --port=PORT --username=USERNAME --password=PASSWORD chembl_22 < chembl_22_1.pgdump.sql
#    Actually:
#      psql --username=chembl --password chembl_17 < chembl_17.pgdump.sql
#      psql --username=chembl --password chembl_20 < chembl_20.pgdump.sql
#      psql --username=chembl --password chembl_22 < chembl_22_1.pgdump.sql
#    (wherever these DB dumps are...)
#    Look at the nice schema documentation in the download page.
#
# N.B. We could also have used anaconda/rdkit instances for postgres
# No need to be root or creating new users...
# It is also very simple, it could be (from the top of my head, not run)
#   conda install postgresql
#   hash -r
#   CHEMBL_DB_PATH=~/postregs-dbs/chembls
#   mkdir -p "${CHEMBL_DB_PATH}"
#   pg_ctl -D "${CHEMBL_DB_PATH}" -l ~/postgresdbs/chembls-logfile start o "-p 5433"
#   initdb --locale $LANG -E UTF8 -D "${CHEMBL_DB_PATH}" -U `whoami`
#   ...keep going with the instructions...


# --- Munging
# Columns:
#   - since_chembl: from which release of chembl was the activity known
#                   (chembl does not store the date of insertion itself,
#                    so this is a very crude approximation to when the activity
#                    was measured).

# --- Interesting resources
# Chembl:
#   - Malaria: https://www.ebi.ac.uk/chembl/malaria/source (not recently updated)
#   - Target types: https://www.dropbox.com/s/4v8wmffpum1b7jg/target_types.pptx?m
