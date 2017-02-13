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

# Ground truth for the competition benchmarl dataset
COMPETITION_GT_CSV = op.join(DATA_DIR, 'tdt2014_challenge1_mar3d7_heldoutdataandresults.txt')

# Original munged data/ models / results from Flo and Santi
CCL_RESULTS_DIR = op.join(SANTI_MODERN_DIR, 'original-data')

