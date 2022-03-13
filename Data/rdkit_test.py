import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

smi = "C1ccccC1"
print("smi type : ", type(smi), smi)
mol = Chem.MolFromSmiles(smi)
print("mol1 type : ", type(mol), mol)
mol = Chem.AddHs(mol)
print("mol2 type : ", type(mol), mol)

noa = mol.GetNumAtoms()
print("noa type : ", type(noa), noa)
hnoa = mol.GetNumHeavyAtoms()
print("hnoa type : ", type(hnoa), hnoa)

c_patt = Chem.MolFromSmiles('C')
print("c_patt type : ", type(c_patt), c_patt)
c_patt_sub = mol.GetSubstructMatches(c_patt)
print("c_patt_sub type : ", type(c_patt_sub), c_patt_sub)

print("hello change")