# Generate fingerprints
from joblib import Parallel, delayed

from ccl_malaria.features import morgan
from tdt_autopsy.pf_chembl import pf_activities

df = pf_activities(chembl_version='22_1')


def molid_smiles():
    for molid, smiles in zip(df['molregno'], df['canonical_smiles']):
        yield molid, smiles


# Generate, on parallel, weird-fingerprints files
n_jobs = 8
n_steps = 32
Parallel(n_jobs=n_jobs)(delayed(morgan)
                        (start=start,
                         step=n_steps,
                         mols=list(molid_smiles()),
                         fcfp=True,
                         output_file='/home/santi/mols_%d_of_%d_fcfp.weirdfpt.gz' % (start, n_steps))
                        for start in range(n_steps))

# Merge the weird-fingerprints files, remove centers

# Use only features that had been used a few years ago when training
