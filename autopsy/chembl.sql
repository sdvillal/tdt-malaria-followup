/*
  Select all organism assays with plasmodium falciparum as target.
  We can decide what was a hit later on.

  Note that most probably we are not interested in most of them,
  and could reduce this to strictly match "plasmodium falciparum".

  This should be alike to querying by organism using chembl web/REST
    https://www.ebi.ac.uk/chembl/
    "falciparum" -> target -> click -> export bioactivities
*/

SELECT cpds.standard_inchi, cpds.canonical_smiles,
  act.*
FROM assays AS ass, activities as act, compound_structures as cpds
WHERE ass.assay_organism ILIKE '%plasmodium falciparum%' AND
  ass.assay_id = act.assay_id AND
  cpds.molregno = act.molregno
