# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
from pathlib import Path

sys.path.append(str(Path(sys.path[0]).parent))
print(sys.path)

import nglview
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, TorsionFingerprints
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdmolops import Kekulize, SanitizeFlags, SanitizeMol
from rdkit.Chem.rdMolTransforms import SetDihedralDeg

import stk
from stk.utilities.utilities import get_plane_normal


def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol

# %%
file = Path.cwd().parent / 'molecules' / 'xor_gate_cache' / 'XORgateNo.mol'
mol = Chem.MolFromMolFile(str(file))
AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=20000)

nonring, ring = TorsionFingerprints.CalculateTorsionLists(mol)
nonring = [list(atoms[0]) for atoms, ang in nonring]


print(nonring)
Chem.rdMolTransforms.SetDihedralDeg(mol.GetConformer(id=0),0,5,10,12, 180.0)
display(Draw.MolToImage(mol_with_atom_index(mol),size=(300,300)))


# %%
file = Path.cwd().parent / 'molecules' / 'xor_gate_cache' / 'XORBuildingBlock.mol'
mol = Chem.MolFromMolFile(str(file))
mol = Chem.rdmolops.RemoveHs(mol)
AllChem.MMFFOptimizeMolecule(mol, confId=0, maxIters=20000)
display(Draw.MolToImage(mol_with_atom_index(mol),size=(700,300)))

# building_block = stk.BuildingBlock('BrCCBr', [stk.BromoFactory()])
c_0, c_1, c_2, c_3, c_4, c_5 = stk.C(6), stk.C(7), stk.C(8), stk.C(3), stk.C(5), stk.C(4)
functional_groups = [stk.GenericFunctionalGroup(atoms=(c_0, c_1, c_2, c_3, c_4, c_5),
                                                bonders=(c_0, c_3), deleters=(c_1, c_2)),
                     stk.GenericFunctionalGroup(atoms=(c_0, c_1, c_2, c_3, c_4, c_5),
                                                bonders=(c_4, c_5), deleters=())]
building_block = stk.BuildingBlock(smiles=Chem.MolToSmiles(mol), functional_groups=functional_groups)
# building_block = stk.BuildingBlock.init_from_rdkit_mol(mol, functional_groups=functional_groups)
# display(Draw.MolToImage(mol_with_atom_index(mol),size=(700,300)))
# building_block = stk.BuildingBlock.init_from_file(str(file), functional_groups=functional_groups)
# reaction = stk.molecular.factories.GenericReactionFactory(
#     bond_orders= {frozenset({stk.GenericFunctionalGroup}): 2}
# )
polymer = stk.ConstructedMolecule(
    topology_graph=stk.polymer.Linear(
        building_blocks=(building_block,),
        repeating_unit='A',
        num_repeating_units=2,
        # reaction_factory=reaction
        # orientations=(0,1)
    )
)
print(Path.cwd())
polymer.write('polymer.mol')
mol = Chem.MolFromMolFile('polymer.mol')
print(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
# mol = polymer.to_rdkit_mol()
mol = Chem.rdmolops.RemoveAllHs(mol, sanitize=True)
mol = Chem.rdmolops.AddHs(mol, addCoords=True)
AllChem.MMFFOptimizeMolecule(mol, confId=0)

display(Draw.MolToImage(mol_with_atom_index(mol),size=(700,300)))
nglview.show_rdkit(mol)

# %%

# JOSH - after creating XorGate class, this cell should no longer be needed

def init_building_block(smiles):
    'construct a building block with hydrogens removed'
    mol = stk.BuildingBlock(smiles=smiles)
    mol = Chem.rdmolops.RemoveHs(mol.to_rdkit_mol(), sanitize=True)
    Kekulize(mol) # convert rdkit aromatic bonds to single and double bonds for portability
    return stk.BuildingBlock.init_from_rdkit_mol(mol)

def make_xor_monomer(position=0):
    # initialize building blocks
    benzene = init_building_block(smiles='C1=CC=CC=C1')
    acetaldehyde = init_building_block(smiles='CC=O')
    benzene = benzene.with_functional_groups([stk.SingleAtom(stk.C(position))])
    acetaldehyde = acetaldehyde.with_functional_groups([stk.SingleAtom(stk.C(1))])

    # construct xor gate monomer
    xor_gate = stk.ConstructedMolecule(
        topology_graph=stk.polymer.Linear(
            building_blocks=(benzene, acetaldehyde),
            repeating_unit='AB',
            num_repeating_units=1
        )
    )

    # construct functional groups for xor gate monomer
    # numbering starts at top and proceeds clockwise
    c_0, c_1, c_2, c_3, c_4, c_5 = stk.C(0), stk.C(1), stk.C(2), stk.C(3), stk.C(4), stk.C(5)
    functional_groups = [stk.GenericFunctionalGroup(atoms=(c_0, c_1, c_2, c_3, c_4, c_5),
                                                    bonders=(c_0, c_3), deleters=(c_4, c_5)),
                        stk.GenericFunctionalGroup(atoms=(c_0, c_1, c_2, c_3, c_4, c_5),
                                                    bonders=(c_1, c_2), deleters=())]
    return stk.BuildingBlock.init_from_molecule(xor_gate,functional_groups=functional_groups)


# construct XOR gate monomers
xor_gate_top = make_xor_monomer(position=0)
xor_gate_bottom = make_xor_monomer(position=3)
# display(Draw.MolToImage(mol_with_atom_index(xor_gate_top.to_rdkit_mol()),size=(700,300)))
# display(Draw.MolToImage(mol_with_atom_index(xor_gate_bottom.to_rdkit_mol()),size=(700,300)))

polymer = stk.ConstructedMolecule(
    topology_graph=stk.polymer.Linear(
        building_blocks=(xor_gate_top,xor_gate_bottom),
        repeating_unit='AAABBB',
        num_repeating_units=3,
        # reaction_factory=reaction
        # orientations=(0,1)
    )
)
# JOSH - RESUME HERE - TRY ROTATING MOLECULE WITH 
# benzene = benzene.with_rotation_about_axis(np.PI / 2, benzene.get_plane_normal(), benzene.get_centroid())

def nonring_torsions():
    nonring = [[1,0,7,8]]
    nonring += [[i+1,i,i+5,i+6] for i in range(9,59,7)]
    nonring += [[i-1,i,i+2,i+3] for i in range(68,125,7)]
    return nonring

mol = polymer.to_rdkit_mol()
for atom in mol.GetAtoms():
    atom.SetNoImplicit(False)
SanitizeMol(mol)
mol = Chem.rdmolops.AddHs(mol, addCoords=True)
conf = mol.GetConformer(id=0)
# display(Draw.MolToImage(mol_with_atom_index(mol),size=(1200,300)))
nglview.show_rdkit(mol)
print(AllChem.MMFFOptimizeMolecule(mol, maxIters=2000, confId=0))
for tor in nonring_torsions():
    # print(f'i:{i+1},j:{i},k:{i+7},l:{i+8}')
    Chem.rdMolTransforms.SetDihedralDeg(conf, tor[0], tor[1], tor[2], tor[3], 90.0)
print(AllChem.MMFFOptimizeMolecule(mol, maxIters=2000, confId=0))
mol = Chem.rdmolops.RemoveHs(mol)
# print(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
display(Draw.MolToImage(mol_with_atom_index(mol),size=(1500,400)))

file = Path.cwd().parent / 'molecules' / 'xor3_gate_6_alt' / 'XOR3gate6_alt.mol'
Chem.MolToMolFile(mol, str(file))
nglview.show_rdkit(mol)

# %%
import ipywidgets as ipw
import nglview as nv
import py3Dmol
from ipywidgets import interact
from rdkit import Chem
from rdkit.Chem import AllChem

print("Hello world")
# %%
# eval, ep and step is hard coded here. Feel free to change those to what actually fit the specification.
path = Path.home() / 'ConformerML' / 'conformer-ml' / 'log'

@interact(eval=(2), ep=(0, 1), step=(0, 199))
def show_mol(eval, ep, step):
    file = path / ('xor_gates/eval' + str(eval) + '/ep' + str(ep) + '/step' + str(step) + '.mol')
    print(file)
    mol = Chem.MolFromMolFile(str(file))
    m = Chem.MolToMolBlock(mol)
    p = py3Dmol.view(width=600, height=600)
    p.removeAllModels()
    p.addModel(m)
    p.setStyle({'stick':{}})
    p.zoomTo()
    # p.show()
    display(nv.show_rdkit(mol,gui=False))
# %%

# playing with using stk to get which building blocks a given torsion comes from
for atom_info in polymer.get_atom_infos():
    atom = atom_info.get_atom()
    # If the atom was added during construction, these will return None.
    building_block = atom_info.get_building_block()
    building_block_id = atom_info.get_building_block_id()
    pass
# %%
