#
# --- Just a reminder of other rdkit facilities for working with substructures / fingerprints
#
#  - FragCatalog
#
#  - Other explicit path FPs (e.g. RDKFingerprint, but in bitInfo when nBitsPerHash > 1?)
#    + findAllPathsBlah (see Code/GraphMol/Fingerprints.{cpp|h}
#
#  Morgan/ECFP hashing schemes still have the edge, specially if we integrate better
#  returning SMARTS patterns from them (rework atom invariants to not collapse, do not
#  make the back and forth generation of SMARTS but do inline...)
#
#
# if __name__ == '__main__':
#
#     from collections import defaultdict
#     from rdkit.Chem.rdfragcatalog import FragCatalog
#     from rdkit.Chem.rdmolops import RDKFingerprint
#     from rdkit.Chem import AllChem
#
#     cat = FragCatalog('/home/santi/trans.merged.s2i')
#
#     mol = AllChem.MolFromSmiles('C(=C(C#N)S(=O)(=O)c1ccccc1)c1ccc(OC(C)=O)c(OC)c1')
#     bitInfo = {}
#     atomBits = []
#     RDKFingerprint(mol, minPath=1, maxPath=100, nBitsPerHash=4, fpSize=1000, bitInfo=bitInfo, atomBits=atomBits)
#
#     bits = defaultdict(list)
#
#     for bit, paths in bitInfo.items():
#         for path in paths:
#             bits[tuple(path)].append(bit)
#             print(AllChem.PathToSubmol(mol, path))
#
#     print(max(map(len, bits.values())))
#
#     bonds = mol.GetBonds()
#     AllChem.FragmentOnBonds()
#     AllChem.FindAllPathsOfLengthN()
#     print(bitInfo)
#     print(atomBits)
