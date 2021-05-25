from rdkit import Chem
from typing import List

class ContinuousActionMixin:

    def _step(self, action: List[float]) -> None:
        conf = self.conf
        for idx, tors in enumerate(self.nonring):
            Chem.rdMolTransforms.SetDihedralDeg(conf, *tors, float(action[idx]))
        Chem.AllChem.MMFFOptimizeMolecule(self.mol, maxIters=10000, confId=self.mol.GetNumConformers() - 1)
    
class DiscreteActionMixin:

    def _step(self, action: List[int]) -> None:
        for idx, tors in enumerate(self.nonring):
            ang = -180 + 60 * action[idx]
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, *tors, float(ang))
        Chem.AllChem.MMFFOptimizeMolecule(self.mol, maxIters=10000, confId=self.mol.GetNumConformers() - 1)