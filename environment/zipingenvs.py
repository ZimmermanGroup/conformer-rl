from graphenvironments import *
from rdkit import Chem
import gym

class BestGibbs(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, folder_name, gibbs_normalize=False, temp_normal=1.0, sort_by_size=True, ind_select=None):
        super(BestGibbs, self).__init__()
        self.temp_normal = temp_normal
        self.gibbs_normalize = gibbs_normalize
        self.all_files = glob.glob(f'{folder_name}*.json')
        self.folder_name = folder_name

        self.ind_select = ind_select

        if sort_by_size:
            self.all_files.sort(key=os.path.getsize)
        else:
            self.all_files.sort()

        self.choice = -1
        self.episode_reward = 0
        self.choice_ind = 1
        self.num_good_episodes = 0

        if '/' in self.folder_name:
            self.folder_name = self.folder_name.split('/')[0]

        while True:
            obj = self.molecule_choice()

            if 'inv_temp' in obj:
                self.temp_normal = obj['inv_temp']

            self.standard_energy = float(obj['standard'])
            if 'total' in obj and self.gibbs_normalize:
                self.total = obj['total']
            else:
                self.total = 1.0

            if 'mol' in obj:
                self.mol = Chem.MolFromSmiles(obj['mol'])
                self.mol = Chem.AddHs(self.mol)
                res = AllChem.EmbedMultipleConfs(self.mol, numConfs=1)
                if not len(res):
                    continue
                res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol)
                self.conf = self.mol.GetConformer(id=0)

            else:
                self.mol = Chem.MolFromMolFile(os.path.join(self.folder_name, obj['molfile']))
                self.mol = Chem.AddHs(self.mol)
                self.conf = self.mol.GetConformer(id=0)
                res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol)

            break

        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]
        self.delta_t = []
        self.current_step = 0
        self.best_seen = 999.9999
        self.energys = []
        self.zero_steps = 0
        self.repeats = 0

    def load(self, obj):
        pass

    def _get_reward(self):
        current = confgen.get_conformer_energies(self.mol)[0]
        current = current * self.temp_normal

        if current >= self.best_seen:
            print('seen better')
        else:
            self.best_seen = current
            print('current', self.best_seen)

        return np.exp(-1.0 * (self.best_seen - self.standard_energy)) / 20.0

    def _get_obs(self):
        data = Batch.from_data_list([mol2vecstupidsimple(self.mol)])
        return data, self.nonring

    def step(self, action):
        # Execute one time step within the environment
        print("action is ", action)
        if len(action.shape) > 1:
            self.action = action[0]
        else:
            self.action = action
        self.current_step += 1

        begin_step = time.process_time()
        desired_torsions = []

        for idx, tors in enumerate(self.nonring):
            deg = Chem.rdMolTransforms.GetDihedralDeg(self.conf, *tors)
            ang = -180.0 + 60 * self.action[idx]
            desired_torsions.append(ang)
            try:
                Chem.rdMolTransforms.SetDihedralDeg(self.conf, tors[0], tors[1], tors[2], tors[3], float(ang))
            except:
                Chem.MolToMolFile(self.mol, 'debug.mol')
                print('exit with debug.mol')
                exit(0)

        Chem.AllChem.MMFFOptimizeMolecule(self.mol, confId=0)

        rbn = len(self.nonring)
        if rbn == 3:
            done = (self.current_step == 25)
        else:
            done = (self.current_step == 200)

        self.mol_appends(done)

        obs = self._get_obs()
        rew = self._get_reward()
        self.episode_reward += rew

        print("reward is ", rew)
        print ("new state is:")
        print_torsions(self.mol)

        end_step = time.process_time()

        delta_t = end_step-begin_step
        self.delta_t.append(delta_t)

        info = {}
        if done:
            info['repeats'] = self.repeats

        info = self.info(info)

        return obs, rew, done, info

    def info(self, info):
        return info

    def mol_appends(self, done):
        pass

    def change_level(self, up_or_down=True):
        print('level', up_or_down)

    def molecule_choice(self):
        if self.ind_select is not None:
            cjson = self.all_files[self.ind_select]
        else:
            cjson = np.random.choice(self.all_files)

        print(cjson, '\n\n\n\n')

        with open(cjson) as fp:
            obj = json.load(fp)
        return obj

    def reset(self):
        self.best_seen = 999.9999
        self.repeats = 0
        self.current_step = 0
        self.zero_steps = 0
        self.seen = set()
        while True:
            obj = self.molecule_choice()

            if 'inv_temp' in obj:
                self.temp_normal = obj['inv_temp']

            self.standard_energy = float(obj['standard'])
            if 'total' in obj and self.gibbs_normalize:
                self.total = obj['total']
            else:
                self.total = 1.0

            if 'mol' in obj:
                self.mol = Chem.MolFromSmiles(obj['mol'])
                self.mol = Chem.AddHs(self.mol)
                res = AllChem.EmbedMultipleConfs(self.mol, numConfs=1)
                if not len(res):
                    continue
                res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol)
                self.conf = self.mol.GetConformer(id=0)
            else:
                self.mol = Chem.MolFromMolFile(os.path.join(self.folder_name, obj['molfile']))
                self.mol = Chem.AddHs(self.mol)
                self.conf = self.mol.GetConformer(id=0)
                res = Chem.AllChem.MMFFOptimizeMoleculeConfs(self.mol)
            break

        self.episode_reward = 0
        nonring, ring = TorsionFingerprints.CalculateTorsionLists(self.mol)
        self.nonring = [list(atoms[0]) for atoms, ang in nonring]

        obs = self._get_obs()

        print('step time mean', np.array(self.delta_t).mean())
        print('reset called\n\n\n\n\n')
        print_torsions(self.mol)
        return obs

    def render(self, mode='human', close=False):
        print_torsions(self.mol)


class TestBestGibbs(BestGibbs):
    def __init__(self):
        super(TestBestGibbs, self).__init__('diff/')

class BestTestGibbs(BestGibbs):
    metadata = {'render.modes': ['human']}

    def _get_reward(self):
        # indnum = self.ind_select + 4
        # path = os.path.join(self.folder_name, f'{indnum}.mol')

        path = os.path.join(self.folder_name, f'optimal_{self.ind_select}.mol')
        print(path)
        mol = Chem.MolFromMolFile(path)
        current = Chem.rdMolAlign.GetBestRMS(Chem.RemoveHs(self.mol), mol)
        if current >= self.best_seen:
            print('seen better')
        else:
            self.best_seen = current
            print('current', self.best_seen)

        if self.current_step == 200:
            return self.best_seen
        else:
            return 0

class BestTestGibbs2(BestGibbs):
    metadata = {'render.modes': ['human']}

    def _get_reward(self):
        # indnum = self.ind_select + 4
        # path = os.path.join(self.folder_name, f'{indnum}.mol')

        path = os.path.join(self.folder_name, f'optimal_{self.ind_select}.mol')
        print(path)
        mol = Chem.MolFromMolFile(path, removeHs=False)

        selfmolenergy = confgen.get_conformer_energies(self.mol)[0]
        molenergy = confgen.get_conformer_energies(mol)[0]

        current = (selfmolenergy - molenergy)

        if current >= self.best_seen:
            print('seen better')
        else:
            self.best_seen = current
            print('current', self.best_seen)

        if self.current_step == 200:
            return self.best_seen
        else:
            return 0

class BestCurriculaExp(BestGibbs):
    def info(self, info):
        info['num_good_episodes'] = self.num_good_episodes
        info['choice_ind'] = self.choice_ind
        return info

    def molecule_choice(self):
        if self.episode_reward > 7.5:
            self.num_good_episodes += 1
        else:
            self.num_good_episodes = 0

        if self.num_good_episodes >= 10:
            filename = f'{self.choice_ind}'
            self.choice_ind += 1
            self.choice_ind = min(self.choice_ind, len(self.all_files))
            self.num_good_episodes = 0

        if self.choice_ind != 1:
            p = 0.5 * np.ones(self.choice_ind) / (self.choice_ind - 1)
            p[-1] = 0.5
            cjson = np.random.choice(self.all_files[0:self.choice_ind], p=p)
        else:
            cjson = self.all_files[0]

        print(cjson, '\n\n\n\n')

        with open(cjson) as fp:
            obj = json.load(fp)
        return obj

    # def info(self, info):
    #     info['choice_ind'] = self.choice_ind
    #     return info

    # def molecule_choice(self):
    #     self.choice_ind = min(self.choice_ind, len(self.all_files))

    #     if self.choice_ind != 1:
    #         p = 0.5 * np.ones(self.choice_ind) / (self.choice_ind - 1)
    #         p[-1] = 0.5
    #         cjson = np.random.choice(self.all_files[0:self.choice_ind], p=p)
    #     else:
    #         cjson = self.all_files[0]

    #     print(cjson, '\n\n\n\n')

    #     with open(cjson) as fp:
    #         obj = json.load(fp)
    #     return obj

    # def change_level(up_or_down):
    #     if up_or_down:
    #         self.choice_ind += 1

    #     elif self.choice_ind != 1:
    #         self.choice_ind -= 1

class TChainTrain(BestCurriculaExp):
    def __init__(self):
        super(TChainTrain, self).__init__('transfer_test_t_chain/')

class TChainTest(BestTestGibbs):
    def __init__(self, **kwargs):
        super(TChainTest, self).__init__('transfer_test_t_chain/', **kwargs)

    def mol_appends(self, done):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

        if done:
            import pickle
            with open('test_mol.pickle', 'wb') as fp:
                pickle.dump(self.backup_mol, fp)

class TChainTest2(BestGibbs):
    def __init__(self, **kwargs):
        super(TChainTest2, self).__init__('transfer_test_t_chain/', **kwargs)

    def mol_appends(self, done):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

        if done:
            import pickle
            with open('test_mol.pickle', 'wb') as fp:
                pickle.dump(self.backup_mol, fp)

class TChainTest3(BestTestGibbs2):
    def __init__(self, **kwargs):
        super(TChainTest3, self).__init__('transfer_test_t_chain/', **kwargs)

    def mol_appends(self, done):
        if self.current_step == 1:
            self.backup_mol = Chem.Mol(self.mol)
            return

        c = self.mol.GetConformer(id=0)
        self.backup_mol.AddConformer(c, assignId=True)

        if done:
            import pickle
            with open('test_mol.pickle', 'wb') as fp:
                pickle.dump(self.backup_mol, fp)
