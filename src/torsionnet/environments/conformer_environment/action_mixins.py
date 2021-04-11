from rdkit import Chem

from .conformer_env import ConformerEnv

class DiscreteActionMixin(ConformerEnv):

    def _handle_action(self):
        desired_torsions = []

        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * self.action[idx]
            desired_torsions.append(ang)
            Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], float(ang))
        Chem.AllChem.MMFFOptimizeMolecule(self.molecule, maxIters=50, confId=0)