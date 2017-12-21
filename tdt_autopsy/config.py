import os.path as op

FOLLOWUP_DIR = op.dirname(__file__)
FOLLOWUP_INSTALL_DIR = op.abspath(op.join(FOLLOWUP_DIR, '..'))

# --- Submodules pointing to submission commits
SUBMISSIONS_DIR = op.join(FOLLOWUP_INSTALL_DIR, 'submissions')
SEREINA_ORIGINAL_DIR = op.join(SUBMISSIONS_DIR, 'sereina-original')
SEREINA_BUGFIXED_DIR = op.join(SUBMISSIONS_DIR, 'sereina-bugfixed')
SEREINA_MODERN_DIR = op.join(SUBMISSIONS_DIR, 'sereina-modern')
SANTI_ORIGINAL_DIR = op.join(SUBMISSIONS_DIR, 'santi-original')
SANTI_MODERN_DIR = op.join(SUBMISSIONS_DIR, 'santi-modern')

# --- Data
DATA_DIR = op.join(FOLLOWUP_INSTALL_DIR, 'data')

# Here are the initial efforts from Greg and Flo
CHEMBL_FIRST_ANALYSIS_DIR = op.join(DATA_DIR, 'ChEMBL_analysis')

# Queries to ChEMBL and munged ChEMBL data will be stored here
CHEMBL_DEST_DIR = op.join(DATA_DIR, 'chembl-pf')

# Assays results from StJude
STJUDE_CSV = op.join(DATA_DIR, 'StJudeResults2016.csv')
STJUDE_FLO_GREG_ANNOTATED_CSV = op.join(CHEMBL_FIRST_ANALYSIS_DIR, 'hits_malaria_screen_flo.csv')

# Tables as shown in the paper
PAPER_NOT_INTRAIN_CSV = op.join(DATA_DIR, '--paper201712', 'paper-notintraining.csv')  # Table 5
PAPER_INTRAIN_CSV = op.join(DATA_DIR, '--paper201712', 'paper-intraining.csv')         # Supplementary 1

# Ground truth for the competition benchmark dataset
COMPETITION_GT_CSV = op.join(DATA_DIR, 'TDT_Malaria_HTS_ranked.csv')
COMPETITION_GT_TSV = op.join(DATA_DIR, 'tdt2014_challenge1_mar3d7_heldoutdataandresults.txt')

# --- Original munged data/ models / results from Flo and Santi

# Dir with the data as submitted (with minimal changes)
CCL_DATA_DIR = op.join(SANTI_MODERN_DIR, 'data-original')
# Original smiles files
CCL_ORIGINAL_DATA_DIR = op.join(CCL_DATA_DIR, 'original')
# Canonical ordering of molecules
CCL_INDICES_DIR = op.join(CCL_DATA_DIR, 'indices')
# Molecules catalog and features
CCL_MOLCATALOGS_DIR = op.join(CCL_DATA_DIR, 'rdkit', 'mols')
CCL_FINGERPRINTS_DIR = op.join(CCL_DATA_DIR, 'rdkit', 'ecfps')  # not really ECFPS
CCL_FEATURES_DIR = op.join(CCL_DATA_DIR, 'rdkit', 'rdkfs')      # other features
# Models, OOBs & cross-vals, submissions
CCL_EXPERIMENTS_DIR = op.join(CCL_DATA_DIR, 'experiments')
CCL_LOGREGS_EXPERIMENTS_DIR = op.join(CCL_EXPERIMENTS_DIR, 'logregs')
CCL_TREES_EXPERIMENTS_DIR = op.join(CCL_EXPERIMENTS_DIR, 'trees')
# Vowpal Wabbit experiments
CCL_VOWPAL_DIR = op.join(CCL_DATA_DIR, 'vowpal')
