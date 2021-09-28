"""
generate_lignin
===============
"""
import numpy as np

# Chemical Drawing
from rdkit.Chem import MolFromMolBlock
from rdkit import Chem

# Lignin-KMC functions and global variables used in this notebook
from ligninkmc.kmc_functions import run_kmc, generate_mol
from ligninkmc.create_lignin import (calc_rates, create_initial_monomers, create_initial_events,
                                     create_initial_state, analyze_adj_matrix, adj_analysis_to_stdout)
from ligninkmc.kmc_common import (DEF_E_BARRIER_KCAL_MOL, ADJ_MATRIX, MONO_LIST, MONOMER, OX, GROW, Monomer, Event)

# Calculate the rates of reaction in 1/s (or 1/monomer-s if biomolecular) at the specified temp
temp = 298.15  # K
rxn_rates = calc_rates(temp, ea_kcal_mol_dict=DEF_E_BARRIER_KCAL_MOL)

def generate_lignin(num_monomers: int = 1) -> Chem.Mol:
        """Generates lignin molecule.

        parameters
        ----------
        num_monomers : int
                Number of monomers in lignin molecule.
        """
        # Set the percentage of S
        sg_ratio = 0
        pct_s = sg_ratio / (1 + sg_ratio)

        # Set the initial and maximum number of monomers to be modeled.
        ini_num_monos = 1
        max_num_monos = num_monomers

        # Maximum time to simulate, in seconds
        t_max = 1  # seconds
        mono_add_rate = 1e4  # monomers/second

        # Use a random number and the given sg_ratio to determine the monolignol types to be initially modeled
        monomer_draw = np.random.rand(ini_num_monos)
        initial_monomers = create_initial_monomers(pct_s, monomer_draw)

        # Initially allow only oxidation events. After they are used to determine the initial state, add 
        #     GROW to the events, which allows additional monomers to be added to the reaction at the 
        #     specified rate and with the specified ratio
        initial_events = create_initial_events(initial_monomers, rxn_rates)
        initial_state = create_initial_state(initial_events, initial_monomers)
        initial_events.append(Event(GROW, [], rate=mono_add_rate))

        # simulate lignin creation
        result = run_kmc(rxn_rates, initial_state,initial_events, n_max=max_num_monos, t_max=t_max, sg_ratio=sg_ratio)
        # using RDKit
        nodes = result[MONO_LIST]
        adj = result[ADJ_MATRIX]
        block = generate_mol(adj, nodes)
        mol = MolFromMolBlock(block)
        mol = Chem.AddHs(mol)

        return mol
