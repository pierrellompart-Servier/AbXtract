import io
from typing import Union, List
from urllib.request import urlopen
import numpy as np
from scipy import optimize
from anarci import anarci
from Bio import SeqIO
import os
from Bio.PDB import *
import tempfile
import zipfile
from tqdm import tqdm
import time
from tqdm.notebook import tqdm
import time



import numpy as np
import numpy.typing as npt

from scipy import spatial

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import pairwise2

from Bio.SeqUtils import seq1
from Bio.PDB.Polypeptide import is_aa
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, NeighborSearch, Selection, DSSP
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.ResidueDepth import get_surface, min_dist as bio_min_dist
from collections import defaultdict
from glob import glob
from Bio import PDB
import multiprocessing
from Bio.PDB.ResidueDepth import ResidueDepth
import subprocess
import os
    
    
import io
from typing import Union, List
from urllib.request import urlopen

import numpy as np
from scipy import optimize

from anarci import anarci
from Bio import SeqIO
import pandas as pd
import numpy as np


import pandas as pd
import numpy as np



import pandas as pd

from numba import njit, prange
import numpy as np

from openmm.app import *
from openmm import *
from Bio.PDB import *
from pdbfixer import PDBFixer
   
import freesasa
from Bio.PDB import *

import numpy as np

from sklearn.cluster import DBSCAN
import numpy as np


from sklearn.cluster import DBSCAN
import numpy as np


from sklearn.cluster import DBSCAN
import numpy as np



from sklearn.cluster import DBSCAN
import numpy as np


from ..utils.constants import (
    main_chain,
    three_to_one,
    hydrophobic_residues,
    HYDROPHOBIC_RESIDUES,
    HYDROPHOBIC_AA,
    pKa_dict,
    AA_CHARGE,
    AA_MW,
    AA_VOLUME,
    AA_CHARGE,
    AA_PI,
    AA_CLASSES,
    AA_LIST,
    KABAT_SCHEME,
    CHOTHIA_SCHEME,
    IMGT_SCHEME,
    HYD_SCALES,
    KD_SCALE,
    EISENBERG_SCALE,
    CRIPPEN_PARAMS,
    PDB_TO_CRIPPEN,
    get_uniprot_seq,
    HC_SEQS,
    LC_SEQS,
    HINGE_REGIONS,
)



class SeqFeaturizer:
    """This class exposes the API for calculating sequence-derived features.
    """
    def __init__(self, seqs_type: tuple, seqs: tuple, is_fv: bool, isotype: str, lc_type: str, pH: float=7.4) -> None:
        """Constructor of SeqFeaturizer, which exposes the API for calculating sequence-derived 
        features.

        Parameters
        ----------
        seqs : tuple
            A pair of heavy and light chain sequences.
        is_fv : bool
            Whether the given pair of sequences are Fv domain only.
        isotype : str
            Isotype of the heavy chain.
        lc_type : str
            Type of the light chain.
        pH : float, optional
            The pH condiation at which to calculate charge related features, by default 7.4

        """
        self.seqs = seqs
        self.isotype = isotype
        self.lc_type = lc_type
        self.pH = pH
        self.seqs_type = seqs_type
        
        if len(seqs) == 2:
            
            if is_fv:

                self.vh_seq, self.vl_seq = seqs

                # create full sequence for the heavy chain
                if self.isotype in HC_SEQS:
                    self.h_full_seq = self.vh_seq + HC_SEQS[self.isotype]
                elif self.isotype.lower() == 'igg1':
                    self.h_full_seq = self.vh_seq + HC_SEQS['IGHG1']
                elif self.isotype.lower() == 'igg2':
                    self.h_full_seq = self.vh_seq + HC_SEQS['IGHG2']
                elif self.isotype.lower() == 'igg4':
                    self.h_full_seq =  self.vh_seq + HC_SEQS['IGHG4']
                else:
                    raise ValueError(f'Unknown heavy chain isotype {self.isotype}!')

                # create full sequence for the light chain
                if self.lc_type in LC_SEQS:
                    self.l_full_seq = self.vl_seq + LC_SEQS[self.lc_type]
                elif self.lc_type.lower() == 'kappa':
                    self.l_full_seq = self.vl_seq + LC_SEQS['IGKC']
                elif self.lc_type.lower() == 'lambda':
                    self.l_full_seq = self.vl_seq + LC_SEQS['IGLC2']
                else:
                    raise ValueError(f'Unknown light chain type {self.lc_type}!')
            else:
                self.h_full_seq, self.l_full_seq = seqs
                self.vh_seq = extract_fv_seq(self.h_full_seq)
                self.vl_seq = extract_fv_seq(self.l_full_seq)
        
        else:
            if "HC" in seqs:
                if is_fv:

                    self.vh_seq = seqs[0]

                    # create full sequence for the heavy chain
                    if self.isotype in HC_SEQS:
                        self.h_full_seq = self.vh_seq + HC_SEQS[self.isotype]
                    elif self.isotype.lower() == 'igg1':
                        self.h_full_seq = self.vh_seq + HC_SEQS['IGHG1']
                    elif self.isotype.lower() == 'igg2':
                        self.h_full_seq = self.vh_seq + HC_SEQS['IGHG2']
                    elif self.isotype.lower() == 'igg4':
                        self.h_full_seq =  self.vh_seq + HC_SEQS['IGHG4']
                    else:
                        raise ValueError(f'Unknown heavy chain isotype {self.isotype}!')

                    # create full sequence for the light chain
                    self.l_full_seq = None
                else:
                    self.h_full_seq=  seqs[0]
                    self.l_full_seq=  None
                    self.vh_seq = extract_fv_seq(self.h_full_seq)
                    self.vl_seq = None

            else:
                
                if is_fv:

                    self.vl_seq = seqs[0]

                    # create full sequence for the light chain
                    if self.lc_type in LC_SEQS:
                        self.l_full_seq = self.vl_seq + LC_SEQS[self.lc_type]
                    elif self.lc_type.lower() == 'kappa':
                        self.l_full_seq = self.vl_seq + LC_SEQS['IGKC']
                    elif self.lc_type.lower() == 'lambda':
                        self.l_full_seq = self.vl_seq + LC_SEQS['IGLC2']
                    else:
                        raise ValueError(f'Unknown light chain type {self.lc_type}!')
                else:
                    self.l_full_seq = seqs[0]
                    self.h_full_seq = None
                    
                    self.vh_seq = None
                    self.vl_seq = extract_fv_seq(self.l_full_seq)
            

        # pH aware charged residues
        if self.pH < 4.0 or self.pH > 10.0:
            raise ValueError(
                f'Invalid pH value {self.pH}!'
                'Please choose a pH in [4.0, 10.0]. Values outside this range may lead to' 
                'incorrect features, especially those related to charges.'
            )
        if self.pH < pKa_dict['H'][0]:
            self.charged_aas = 'DEHKR'
        else:
            self.charged_aas = 'DEKR'
 
    def n_charged_res(self, seqs) -> int:
        """Counts the number of charged residues for one pair of heavy and light chains.

        Returns
        -------
        int
            Number of charged residues.
        """
        if len(seqs) == 2:
            return np.sum([
            aa in self.charged_aas for aa in self.h_full_seq + self.l_full_seq
            ])
        else:
            if "HC" in seqs:
                return np.sum([
                    aa in self.charged_aas for aa in self.h_full_seq
                ])
            else:
                return np.sum([
                    aa in self.charged_aas for aa in  self.l_full_seq
                ])

    def n_charged_res_fv(self, seqs) -> int:
        """Counts the number of charged residues in the Fv domain (only one arm).

        Returns
        -------
        int
            Number of charged residues in the Fv domain.
        """
        if len(seqs) == 2:
            return np.sum([
                aa in self.charged_aas for aa in self.vh_seq + self.vl_seq
                ])
        else:
            if "HC" in seqs:
                return np.sum([
                    aa in self.charged_aas for aa in self.vh_seq 
                ])
            else:
                return np.sum([
                        aa in self.charged_aas for aa in self.vl_seq
                    ])

    def vh_charge(self) -> float:
        """Calculates the charge of the VH domain.

        Returns
        -------
        float
            The charge of the VH domain
        """
        return calculate_seq_charge(self.vh_seq, self.pH)

    def vl_charge(self) -> float:
        """Calculates the charge of the VL domain.

        Returns
        -------
        float
            The charge of the VL domain.
        """

        return calculate_seq_charge(self.vl_seq, self.pH)

    def fv_charge(self, seqs) -> float:
        """Calcualtes the charge of the Fv domain.

        Returns
        -------
        float
            The charge of the Fv domain.
        """
        if len(seqs) == 2:
            return self.vh_charge() + self.vl_charge()
        else:
            if "HC" in seqs:
                return self.vh_charge()
            else:
                return self.vl_charge()

    def fv_csp(self, seqs) -> float:
        """Calculates the charge separation of the Fv domain.

        Returns
        -------
        float
            The charge separation of the Fv domain.
        """
        if len(seqs) == 2:
            return self.vh_charge() * self.vl_charge()
        else:
            return None

    def theoretical_pi(self, seqs) -> float:
        """Calculates the theoretical pI of the antibody (full sequence, both arms)

        Returns
        -------
        float
            The theoretical pI of the antibody.
        """
        if len(seqs) == 2:
            return calculate_pi([self.h_full_seq] * 2 + [self.l_full_seq] * 2)
        else:
            if "HC" in seqs:
                return calculate_pi([self.h_full_seq] * 2)
            else:
                return calculate_pi([self.l_full_seq] * 2)

    def fab_charge(self, seqs) -> float:
        """Calculates the charge of the Fab domain.

        Returns
        -------
        float
            The charge of the Fab domain.
        """
        
        # get the sequence for the Fab domain
        if self.isotype in HINGE_REGIONS:
            fab_end = HINGE_REGIONS[self.isotype][0] - 1
        elif self.isotype.lower() == 'igg1':
            fab_end = HINGE_REGIONS['IGHG1'][0] - 1
        elif self.isotype.lower() == 'igg2':
            fab_end = HINGE_REGIONS['IGHG2'][0] - 1
        elif self.isotype.lower() == 'igg3':
            fab_end = HINGE_REGIONS['IGHG3'][0] - 1
        elif self.isotype.lower() == 'igg4':
            fab_end = HINGE_REGIONS['IGHG4'][0] - 1
        else:
            raise ValueError(f'Unknown heavy chain isotype {self.isotype}!')

        if len(seqs) == 2:
                        
            ch_seq = self.h_full_seq[len(self.vh_seq):]    
            fab_seq = self.vh_seq + ch_seq[:fab_end] + self.l_full_seq

            return calculate_seq_charge(fab_seq)
        else:
            if "HC" in seqs:
                ch_seq = self.h_full_seq[len(self.vh_seq):]    
                fab_seq = self.vh_seq + ch_seq[:fab_end]

                return calculate_seq_charge(fab_seq)
            else:   
                fab_seq = self.l_full_seq
                return calculate_seq_charge(fab_seq)
        
    def fc_charge(self, seqs) -> float:
        """Calculates the charge of the Fc domain.

        Returns
        -------
        float
            The charge of the Fc domain.
        """

        # get the sequence for the Fc domain
        if self.isotype in HINGE_REGIONS:
            fc_start = HINGE_REGIONS[self.isotype][1]
        elif self.isotype.lower() == 'igg1':
            fc_start = HINGE_REGIONS['IGHG1'][1]
        elif self.isotype.lower() == 'igg2':
            fc_start = HINGE_REGIONS['IGHG2'][1]
        elif self.isotype.lower() == 'igg3':
            fc_start = HINGE_REGIONS['IGHG3'][1]
        elif self.isotype.lower() == 'igg4':
            fc_start = HINGE_REGIONS['IGHG4'][1]
        else:
            raise ValueError(f'Unknown heavy chain isotype {self.isotype}!')
            
        if "HC" in seqs:
            ch_seq = self.h_full_seq[len(self.vh_seq):]
            fc_seq = ch_seq[fc_start:]

            return calculate_seq_charge(fc_seq)
        else:
            return None
        
        
    def fab_fc_csp(self, seqs) -> float:
        """Calculates the charge separation parameter (CSP) between the Fab domain and Fc domain.

        Returns
        -------
        float
            The charge separation parameter between the Fab domain and the Fc domain.
        """
        if len(seqs) == 2:
            try:
                return self.fab_charge(seqs) * self.fc_charge(seqs)
            except:
                return self.fab_charge(seqs)
        else:
            return None
    
def get_all_seq_features(
    heavy_seq: str, 
    light_seq: str, 
    is_fv: bool,
    isotype: str, 
    lc_type: str, 
    pH: float=7.4
) -> dict:
    """Calculates all currently implemented sequence-based features for the given antibody.

    Parameters
    ----------
    heavy_seq : str
        Amino acid sequence of the heavy chain.
    light_seq : str
        Amino acid sequence of the light chain.
    is_fv : bool
        Is the given sequence only the Fv region.
    isotype : str
        Isotype of the antibody. Select one from ['igg1', 'igg2', 'igg4'].
    lc_type : str
        Type of the light chain, either kappa or lambda.

    Returns
    -------
    dict
        Sequence features of the given antibody as a dictionary keyed by feature names.
    """
    if heavy_seq :
        seqs = (heavy_seq)
        seqs_type = ("HC")
        
    if light_seq :
        seqs = (light_seq)
        seqs_type = ("LC")
        
    if heavy_seq and light_seq:
        seqs = (heavy_seq, light_seq)
        seqs_type = ("HC","LC")
        
    seq_featurizer = SeqFeaturizer(
        seqs_type=seqs_type, seqs=seqs, is_fv=is_fv, isotype=isotype, 
        lc_type=lc_type, pH=pH
    )
    

    seq_features = {
        'theoretical_pi': seq_featurizer.theoretical_pi(seqs),
        'n_charged_res': seq_featurizer.n_charged_res(seqs),
        'n_charged_res_fv': seq_featurizer.n_charged_res_fv(seqs),
        'fv_charge': seq_featurizer.fv_charge(seqs),
        'fv_csp': seq_featurizer.fv_csp(seqs),
        'fc_charge': seq_featurizer.fc_charge(seqs),
        'fab_fc_csp': seq_featurizer.fab_fc_csp(seqs)
    }

    return seq_features

from tqdm.auto import tqdm


def pdb_from_file(pdb_code,pdb_file_location=None):
    """ get a Bio.PDB.Structure for a pdb from file  
    
    parameters:
    -----------
    pdb_file_location: (string) The pdb code file location
    clean_non_ca: (bool, optional) remove residues without defined C_alpha atom
    
    returns:
    ---------
    structure: (Bio.PDB.Structure) pdb structure file
    """
    parser = PDBParser()
    structure = parser.get_structure('bound', os.path.join(pdb_file_location,pdb_code))
    return structure

        
    

def save_structure(structure,pdb_out_path):
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save(pdb_out_path)


class RipleyK:
    def __init__(
        self, obs_coords: npt.ArrayLike, allowed_coords: npt.ArrayLike,
        distance: float = 8., p: int = 2, n: int = 1000
    ):
        """

        Parameters
        ----------
        obs_coords
        distance
        """
        self.obs_coords = np.asarray(obs_coords)
        self.allowed_coords = np.asarray(allowed_coords)
        self.distance = distance
        self.p = p
        self.n = n
        self._ripley_k = None

    @property
    def ripley_k(self) -> float:
        """

        Returns
        -------

        """
        if self._ripley_k is None:
            feature_size = self.obs_coords.shape[0]
            denominator = feature_size * (feature_size - 1)
            k_o = self.get_number_of_pairs(self.obs_coords, self.distance) / denominator
            rng = np.random.default_rng()
            k_e_null = []
            for _ in range(self.n):
                new_locations = rng.choice(self.allowed_coords.shape[0], size=feature_size)
                new_coords = self.allowed_coords[new_locations]
                k_e_null.append(
                    self.get_number_of_pairs(new_coords, self.distance) / denominator
                )
            k_e = np.mean(k_e_null)

            self._ripley_k = k_o / k_e
        return self._ripley_k

    @staticmethod
    def get_number_of_pairs(coords: npt.ArrayLike, distance):
        """Computes the number of neighboring pairs.

        Parameters
        ----------
        coords : npt.ArrayLike
            Cartesian coordinates of the features.

        Returns
        -------
        float
            The the number of neighboring pairs.

        """
        kd_tree = spatial.KDTree(coords)
        neighbor_pairs = kd_tree.query_pairs(r=distance)
        return len(neighbor_pairs)


class AverageNearestNeighbor:
    def __init__(
        self, feature_coords: npt.ArrayLike, allowed_coords: npt.ArrayLike,
        p: int = 2, n: int = 1000
    ):
        """

        Parameters
        ----------
        coords
        p : int
            Which Minkowski p-norm to use.

        n : int
            Number of permutations to do in deriving the expected mean distance.
        """
        self.feature_coords = np.asarray(feature_coords)
        self.allowed_coords = np.asarray(allowed_coords)
        self.p = p
        self.n = n
        self._ann_index = None

    @property
    def ann_index(self):
        """Computes the Average Nearest Neighbor index feature.
        The Average Nearest Neighbor index feature is defined as the ratio of observed mean distance
        to the expected mean distance.

        Returns
        -------
        float
            The Average Nearest Neighbor inex feature.

        """
        if self._ann_index is None:
            # observed mean distance each point and its nearest neighbor
            d_o = self.compute_nn_mean_distance(self.feature_coords)

            # expected mean distance for the features given a "random" pattern
            feature_size = self.feature_coords.shape[0]
            rng = np.random.default_rng()
            d_e_null = []
            for _ in range(self.n):
                new_locations = rng.choice(self.allowed_coords.shape[0], size=feature_size)
                new_coords = self.allowed_coords[new_locations]
                d_e_null.append(
                    self.compute_nn_mean_distance(new_coords)
                )
            d_e = np.mean(d_e_null)
            self._ann_index = d_o / d_e
        return self._ann_index

    @staticmethod
    def compute_nn_mean_distance(coords: npt.ArrayLike) -> float:
        """Computes the mean distance of nearest neighbors.

        Parameters
        ----------
        coords : npt.ArrayLike
            Cartesian coordinates of the features.

        Returns
        -------
        float
            The mean distance of nearest neighbors.

        """
        nn_distances = []
        kd_tree = spatial.KDTree(coords)
        for point in coords:
            # point is contained in the tree, so its nearest neighbor
            # should exclude itself, hence k=2
            dists, _ = kd_tree.query(point, k=2)
            nn_distances.append(dists[1])
        return np.mean(nn_distances)
    
import numpy as np

import openmm.app as openmm_app

class StructFeaturizer:
    def __init__(self, pdb_file: str):
        """Encapsulation of functionalities for computing structure-based features.

        Parameters
        ----------
        pdb_file : str
            A PDB file storing the structure.
        """
        self.pdb_file = pdb_file
        self.openmm_pdb = openmm_app.PDBFile(pdb_file)
        self._struct = None
        self._residue_sasa = None
        self._atom_sasa =None
        self._atom_charges = self.atom_charges= None
        self._atoms = None
        self._residues = None
        self._system_and_topology = None
        self.seq_HC = None
        self.seq_LC = None

    @property
    def struct(self):
        if self._struct is None:
            self._struct = load_structure(self.pdb_file)
        return self._struct

    @property
    def atoms(self):
        if self._atoms is None:
            self._atoms = list(self.openmm_pdb.topology.atoms())
        return self._atoms

    @property
    def residues(self):
        if self._residues is None:
            self._residues = list(self.openmm_pdb.topology.residues())
        return self._residues

    @property
    def residue_sasa(self):
        """Residue solvent accessible surface area.

        Returns
        -------
        Nested dict
            First level key is chain ID, second level key is residue number.
        """
        if self._residue_sasa is None:
            sasa_results = apply_sasa(self.pdb_file)
            self._residue_sasa = sasa_results.residueAreas()
        return self._residue_sasa


    @property
    def system_and_topology(self):
        if self._system_and_topology is None:
            self._system_and_topology = simple_system(self.pdb_file)
        return self._system_and_topology

    @property
    def atom_sasa(self):
        """Atom solvent accessible surface area.

        Returns
        -------

        """
        if self._atom_sasa is None:
            sasa_results = apply_sasa(self.pdb_file, inc_hydrogen=True)
            atom_sasas = np.array([
                sasa_results.atomArea(i) for i in range(sasa_results.nAtoms())
            ])
            atom_chain_ids = [a.full_id[2] for a in self.struct.get_atoms()]
            atom_sasa_dict = {}
            for chain_id, s in zip(atom_chain_ids, atom_sasas):
                if chain_id not in atom_sasa_dict:
                    atom_sasa_dict[chain_id] = [s]
                else:
                    atom_sasa_dict[chain_id].append(s)
            self._atom_sasa = atom_sasa_dict
        return self._atom_sasa
    
    
    def net_charge(self) -> dict:
        if self._atom_charges is None:
            system, topology = self.system_and_topology
            atom_charges = {}
            chain_ids = [chain.id for chain in self.struct.get_chains()]
            for chain_id in chain_ids:
                atom_charges[chain_id] = np.array([
                    c._value for c in get_partial_charges_system(system, topology, chains=chain_id)
                ])
            self._atom_charges = atom_charges

        from tqdm import tqdm
        charges_per_chain = {}

        for chain_id, chain_atom_charges in self._atom_charges.items():
            charges_per_chain[chain_id] = [float(c) for c in chain_atom_charges]

        return charges_per_chain



    
    
    def exposed_net_charge(self) -> float: 
        """The net charge of atoms at the surface.

        Returns
        -------
        float
            Total charge exposed at the solvent exposed surface.

        """
        total_exposed_charge = 0.
        chain_ids = self._atom_charges.keys()
        for chain_id in chain_ids:
            for c, s in zip(self._atom_charges[chain_id], self._atom_sasa[chain_id]):
                if s > 0:
                    total_exposed_charge += c
        return total_exposed_charge

    def net_charge_cdr(self, numbering_scheme: str = 'IMGT', chain_ids: str = None, exposed: bool = False) -> dict:
        """Returns the list of atomic charges in CDRs (optionally solvent-exposed), per chain."""
        print("> Compute atom charges and SASA")

        from tqdm import tqdm
        
        if chain_ids is None:
            chain_ids = self._atom_charges.keys()

        self._atom_sasa = self.atom_sasa()
        cdr_charges_per_chain = {}

        for chain_id in tqdm(chain_ids, desc="cdr_charge"):
            atoms = list(self.struct[0][chain_id].get_atoms())
            charges = self._atom_charges[chain_id]
            sasas = self._atom_sasa.get(chain_id, [0.0] * len(charges))

            chain_cdr_charges = []
            cdr_res_ids = cdr_residues.get(chain_id, set())

            for atom, c, s in zip(atoms, charges, sasas):
                if exposed and s == 0:
                    continue
                res = atom.get_parent()
                res_id = res.id[1]  # numerical residue number
                if res_id in cdr_res_ids:
                    chain_cdr_charges.append(float(c))

            cdr_charges_per_chain[chain_id] = chain_cdr_charges

        return cdr_charges_per_chain


    
    def dipole_moment(self) -> float:
        """Compute the dipole moment of the given structure in Debye unit.

        The dipole moment is computed as the vector sum of a cloud of point charges,
        i.e. mu = sum(q_i * r_i) where q_i and r_i are the partial charge and the
        position vector of atom i respectively, and i runs over all atoms of the protein.

        References
        https://academic.oup.com/nar/article/35/suppl_2/W512/2922221
        https://www.cell.com/fulltext/S0006-3495(95)80001-9

        Returns
        -------
        float
            The magnitude of the dipole moment.
        """
        if self._atom_charges is None:
            system, topology = self.system_and_topology
            atom_charges = {}
            chain_ids = [chain.id for chain in self.struct.get_chains()]
            for chain_id in chain_ids:
                atom_charges[chain_id] = np.array([
                    c._value for c in get_partial_charges_system(system, topology, chains=chain_id)
                ])
            self._atom_charges = atom_charges

            
        # atom_coords = self.openmm_pdb.getPositions(asNumpy=True)
        atom_coords = []
        for atom in self.struct.get_atoms():
            atom_coords.append(atom.coord)
        atom_coords = np.array(atom_coords)
        center_coords = np.mean(atom_coords, axis=0)
        centered_atom_coords = atom_coords - center_coords

        all_atom_charges = np.concatenate(list(self._atom_charges.values()))
        atom_charges = all_atom_charges.reshape((-1, 1))
        dipole_vector = np.sum(centered_atom_coords * atom_charges, axis=0)
        return np.linalg.norm(4.803 * dipole_vector)

    def hyd_moment(self, hyd_scale: str = 'kd') -> float:
        """Computes the first-order hydrophobic moment of the given structure.

        The hydrophobic moment is computed as the vector sum of a cloud of hydrophobicity
        points, i.e. mu = sum(h_j * r_j) where h_j and r_j are the hydrophobicity and the
        position vector of residue j respectively, and j runs over all residues of the protein.

        References
        https://www.pnas.org/doi/10.1073/pnas.081086198

        Returns
        -------
        float
            The magnitude of the hydrophobic moment.
        """
        # first compute center of geometry for each residue
        residue_cog_all = []
        residue_hyd_all = []
        for residue in self.struct.get_residues():
            residue_atom_coords = np.array([
                a.coord for a in residue.get_atoms()
            ])
            residue_cog_all.append(np.mean(residue_atom_coords, axis=0))
            if hyd_scale == 'kd':
                residue_hyd_all.append(
                    KD_SCALE[residue.resname]
                )
            else:
                residue_hyd_all.append(
                    EISENBERG_SCALE[residue.resname]
                )

        residue_cog_all = np.array(residue_cog_all)
        residue_hyd_all = np.array(residue_hyd_all).reshape((-1, 1))
        hyd_vector = np.sum(residue_cog_all * residue_hyd_all, axis=0)
        return np.linalg.norm(hyd_vector)

    def fv_chml(self) -> float:
        """Formal charge of the VH minus the formal charge of the VL domains.
        """
        vh_charge = np.sum(self._atom_charges['H'])
        if "L" in self._atom_charges.keys():
            vl_charge = np.sum(self._atom_charges['L'])
        else:
            vl_charge = 0
        return vh_charge - vl_charge

    def exposed_fv_chml(self) -> float:
        """Formal charge of the VH minus the formal charge of the VL domains.
        """

        if self._atom_sasa is None:
            print("> Apply SASA calculation")
            sasa_results = apply_sasa(self.pdb_file, inc_hydrogen=True)

            print("> Compute per-atom SASA values")
            from tqdm import tqdm
            atom_sasas = [
                sasa_results.atomArea(i) for i in tqdm(range(sasa_results.nAtoms()), desc="atom_area")
            ]

            print("> Get chain IDs from structure")
            atom_chain_ids = [a.full_id[2] for a in self.struct.get_atoms()]
            if len(atom_chain_ids) != len(atom_sasas):
                raise ValueError(f"Mismatch: {len(atom_chain_ids)} chain IDs vs {len(atom_sasas)} SASA values")

            print("> Group SASA values by chain")
            atom_sasa_dict = {}
            for chain_id, s in tqdm(zip(atom_chain_ids, atom_sasas), desc="group_sasa", total=len(atom_sasas)):
                atom_sasa_dict.setdefault(chain_id, []).append(s)

            self._atom_sasa = atom_sasa_dict
            print("> Finished SASA per chain")

        exposed_vh_charge = []
        for h_c, h_s in zip(self._atom_charges['H'], self._atom_sasa['H']):
            if h_s > 0:
                exposed_vh_charge.append(h_c)
        exposed_vl_charge = []
        for l_c, l_s in zip(self._atom_charges['L'], self._atom_sasa['L']):
            if l_s > 0:
                exposed_vl_charge.append(l_c)
        return exposed_vh_charge, exposed_vl_charge

    def hyd_asa(self) -> float:
        """Total hydrophobic accessible surface area.
        """
        total_hyd_asa = 0.
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                total_hyd_asa += residue_area.apolar
        return total_hyd_asa

    def hph_asa(self) -> float:
        """Total hydrophilic accessible surface area.
        """
        total_hph_asa = 0.
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                total_hph_asa += residue_area.polar
        return total_hph_asa

    def aromatic_asa(self) -> float:
        """Count the total number of exposed aromatic residues.

        Parameters
        ----------
        rsa_cutoff : float
            Cutoff for the relative solvent accessible surface area above which a residue
            will considered exposed.

        Returns
        -------
        int
            The total number of aromatic residues considered exposed.
        """
        total_asa_aromatic = 0
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                res_name = residue_area.residueType
                total_asa = residue_area.total
                if res_name in ['PHE', 'TYR', 'TRP']:
                    total_asa_aromatic += total_asa
        return total_asa_aromatic

    
    def pos_asa(self) -> float:
        """Total positively charged accessible surface area (ARG, LYS, HIS).

        Returns
        -------
        float
            Sum of ASA for positively charged residues.
        """
        total_pos_asa = 0.
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                if residue_area.residueType in ['ARG', 'LYS', 'HIS']:
                    total_pos_asa += residue_area.total
        return total_pos_asa

    def neg_asa(self) -> float:
        """Total negatively charged accessible surface area (ASP, GLU).

        Returns
        -------
        float
            Sum of ASA for negatively charged residues.
        """
        total_neg_asa = 0.
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                if residue_area.residueType in ['ASP', 'GLU']:
                    total_neg_asa += residue_area.total
        return total_neg_asa

    
    
    def cdr_length(self, cdr: str = 'H3', numbering_scheme: str = 'IMGT') -> int:
        """Count the number of residues in the specified CDR.

        Parameters
        ----------
        cdr: str
            Name of the CDR region. Choose among [H1, H2, H3, L1, L2, L3].
        numbering_scheme : str
            Numbering scheme for the Fv region.

        Returns
        -------
        int
            Length of the specified CDR.
        """
        if numbering_scheme.upper() == 'IMGT':
            cdr_boundaries = IMGT_SCHEME
        elif numbering_scheme.upper() == 'KABAT':
            cdr_boundaries = KABAT_SCHEME
        else:
            cdr_boundaries = CHOTHIA_SCHEME

        # Get the correct chain
        chain_id = cdr.upper()[0]
        correct_chain = None
        for chain in self.struct.get_chains():
            if chain.id == chain_id:
                correct_chain = chain
                break

        if correct_chain is None:
            return 0

        # Get all residues in the chain
        residues = list(correct_chain.get_residues())
        residue_numbers = [res.id[1] for res in residues]

        # Check if we're using sequential numbering (like 0,1,2... or 120,121,122...)
        min_num = min(residue_numbers)
        max_num = max(residue_numbers)
        expected_sequential = list(range(min_num, max_num + 1))
        is_sequential = (sorted(residue_numbers) == expected_sequential)

        if is_sequential:
            # Use position-based boundaries instead of residue numbers
            # These are approximate positions where CDRs typically occur in the sequence
            POSITION_BASED_CDR = {
                'H1': (27, 35),   # Roughly positions 25-35 in heavy chain
                'H2': (50, 58),   # Roughly positions 50-58 in heavy chain  
                'H3': (95, 110),  # Roughly positions 95-110 in heavy chain
                'L1': (23, 35),   # Roughly positions 23-35 in light chain
                'L2': (49, 56),   # Roughly positions 49-56 in light chain
                'L3': (89, 100)   # Roughly positions 89-100 in light chain
            }

            if cdr.upper() not in POSITION_BASED_CDR:
                return 0

            cdr_start, cdr_end = POSITION_BASED_CDR[cdr.upper()]

            # Count residues by their position in the chain sequence
            count = 0
            for i, residue in enumerate(residues):
                position_in_chain = i + 1  # Convert to 1-based position
                if cdr_start <= position_in_chain <= cdr_end:
                    count += 1

            return count

        else:
            # Use standard numbering boundaries (original approach)
            cdr_start, cdr_end = cdr_boundaries[cdr.upper()]

            # Count residues by their actual residue numbers
            count = 0
            for residue in residues:
                res_number = residue.id[1]
                if cdr_start <= res_number <= cdr_end:
                    count += 1

            return count


    def aromatic_cdr(self) -> int:
        """Count the number of aromatic residues (F, Y, W) in CDR sequences."""
        # Requires self.seq_HC and self.seq_LC to be defined as strings
        if self.seq_HC:
            seq_HC = SeqAnnotation(seq=self.seq_HC)
            cdr_seq_H = seq_HC.get_cdr_seq("H1") + seq_HC.get_cdr_seq("H2") + seq_HC.get_cdr_seq("H3")
            total_cdr_seq = cdr_seq_H
            
        if self.seq_LC:
            seq_LC = SeqAnnotation(seq=self.seq_LC)
            cdr_seq_L = seq_LC.get_cdr_seq("L1") + seq_LC.get_cdr_seq("L2") + seq_LC.get_cdr_seq("L3")
            total_cdr_seq =  cdr_seq_L
            
        if self.seq_HC and self.seq_LC:
            seq_LC = SeqAnnotation(seq=self.seq_LC)
            seq_HC = SeqAnnotation(seq=self.seq_HC)
            cdr_seq_H = seq_HC.get_cdr_seq("H1") + seq_HC.get_cdr_seq("H2") + seq_HC.get_cdr_seq("H3")
            cdr_seq_L = seq_LC.get_cdr_seq("L1") + seq_LC.get_cdr_seq("L2") + seq_LC.get_cdr_seq("L3")
            total_cdr_seq = cdr_seq_H + cdr_seq_L
            
        aromatic_aas = {'F', 'Y', 'W'}

        return sum(1 for aa in total_cdr_seq if aa in aromatic_aas)



    def exposed_aromatic(self, rsa_cutoff: float = 0.05) -> int:
        """Count the total number of exposed aromatic residues.

        Parameters
        ----------
        rsa_cutoff : float
            Cutoff for the relative solvent accessible surface area above which a residue
            will be considered exposed.

        Returns
        -------
        int
            The total number of aromatic residues considered exposed.
        """
        total_exposed_aromatic = 0
        for _, chain_area in self.residue_sasa.items():
            for _, residue_area in chain_area.items():
                res_name = residue_area.residueType
                total_rsa = residue_area.relativeSideChain
                if total_rsa >= rsa_cutoff and res_name in ['PHE', 'TYR', 'TRP']:
                    total_exposed_aromatic += 1
        return total_exposed_aromatic

    def ann_index(
        self, prop: str = 'pos', rsa_cutoff: float = 0.05, n: int = 1000
    ) -> float:
        """Calculates the Average Nearest Neighbor (ANN) statistic for the given property.

        Parameters
        ----------
        prop : str
            Property (amino acid) type for which the ANN statistic is calculated.
            Distance cutoff at/shorter than which two residues are considered neighbors.
        rsa_cutoff : float
            Relative solvent accessibility cutoff above which a residue is considered 
            solvent exposed, i.e. at the surface.
        n : int
            Number of permutation runs for simulating the null distribution.

        Returns
        -------
        float
            The ANN statistic for the given property.

        """
        if prop.lower()[:3] == 'neg':
            prop_res_names = ['ASP', 'GLU']
        elif prop.lower()[:3] == 'aro':
            prop_res_names = ['PHE', 'TYR', 'TRP']
        else:
            prop_res_names = ['ARG', 'LYS', 'HIS']

        prop_ca_coords = []
        all_ca_coords = []
        for chain_id, chain_area in self.residue_sasa.items():
            chain_ca_coords = [
                r['CA'].coord for r in self.struct[0][chain_id].get_residues()
            ]
            for ca_coord, residue_area in zip(chain_ca_coords, chain_area.values()):
                res_name = residue_area.residueType
                total_rsa = residue_area.relativeSideChain
                if total_rsa >= rsa_cutoff:
                    all_ca_coords.append(ca_coord)
                    if res_name in prop_res_names:
                        prop_ca_coords.append(ca_coord)
        ann = AverageNearestNeighbor(
            feature_coords=prop_ca_coords, allowed_coords=all_ca_coords, n=n
        )
        return ann.ann_index

    def ripley_k(
        self, prop: str = 'pos', distance: float = 8.0,
        rsa_cutoff: float = 0.05, n: int = 1000
    ) -> float:
        """Calculates a variant of the Ripley's K statistic for the given property.

        Parameters
        ----------
        prop : str
            Property (amino acid) type for which the Ripley's K statistic is calculated.
        distance : float
            Distance cutoff at/shorter than which two residues are considered neighbors.
        rsa_cutoff : float
            Relative solvent accessibility cutoff above which a residue is considered 
            solvent exposed, i.e. at the surface.
        n : int
            Number of permutation runs for simulating the null distribution.

        Returns
        -------
        float
            The Ripley's K statistic for the given property.

        """
        if prop.lower()[:3] == 'neg':
            prop_res_names = ['ASP', 'GLU']
        elif prop.lower()[:3] == 'aro':
            prop_res_names = ['PHE', 'TYR', 'TRP']
        else:
            prop_res_names = ['ARG', 'LYS', 'HIS']

        prop_ca_coords = []
        all_ca_coords = []
        for chain_id, chain_area in self.residue_sasa.items():
            chain_ca_coords = [
                r['CA'].coord for r in self.struct[0][chain_id].get_residues()
            ]
            for ca_coord, residue_area in zip(chain_ca_coords, chain_area.values()):
                res_name = residue_area.residueType
                total_rsa = residue_area.relativeSideChain
                if total_rsa >= rsa_cutoff:
                    all_ca_coords.append(ca_coord)
                    if res_name in prop_res_names:
                        prop_ca_coords.append(ca_coord)
        this_ripley_k = RipleyK(
            obs_coords=prop_ca_coords, allowed_coords=all_ca_coords,
            distance=distance, n=n
        )
        return this_ripley_k.ripley_k
    

    
# Function to calculate Solvent Accessible Surface Area (SASA) and Relative Accessible Surface Area (RASA)
def calculate_SASA_RASA(model):
    # Get all chain IDs in the model
    chains = [chain.get_id() for chain in model.get_chains()]
    sr = ShrakeRupley()          # Initialize SASA calculator
    sr.compute(model, level="R") # Compute SASA at the residue level
    data_ASA = []

    for chain_id in chains:
        chain = model[chain_id]
        for residue in chain:
            # Get residue number and store SASA data
            residue_num = str(residue.get_id()[1]) + residue.get_id()[2]
            residue_num = residue_num.strip()
            data_ASA.append((chain_id, residue_num, residue.get_resname(), round(residue.sasa, 2)))

    # Create DataFrame with SASA data
    df_SASA = pd.DataFrame(data_ASA, columns=['chain', 'residue_num', 'residue_name', 'SASA'])

    # Define maximum ASA values for amino acids
    asa_max = {"ALA": 107.24, "ARG": 233.01, "ASN": 150.85, "ASP": 144.06,
               "CYS": 131.46, "GLN": 177.99, "GLU": 171.53, "GLY": 80.54,
               "HIS": 180.93, "ILE": 173.40, "LEU": 177.87, "LYS": 196.14,
               "MET": 186.80, "PHE": 200.93, "PRO": 133.78, "SER": 115.30,
               "THR": 136.59, "TRP": 240.12, "TYR": 213.21, "VAL": 149.34}

    # Calculate RASA and add it to the DataFrame
    df_SASA_RASA = df_SASA.copy()
    df_SASA_RASA['RASA'] = round((df_SASA_RASA['SASA'] / df_SASA_RASA['residue_name'].str[:3].map(asa_max).fillna(1)) * 100, 3)
    
    return df_SASA_RASA

# Function to calculate Half-Sphere Exposure (HSE) values
def calculate_HSE(model):
    chains = [chain.get_id() for chain in model.get_chains()]
    data_hse = []
    RADIUS = 12.0 # Define radius for HSE calculation
    hse_CA = PDB.HSExposure.HSExposureCA(model, RADIUS)
    hse_CB = PDB.HSExposure.HSExposureCB(model, RADIUS)


    for chain_id in chains:
        chain = model[chain_id]
        for residue in chain:
            residue_num = str(residue.get_id()[1]) + residue.get_id()[2]
            residue_num = residue_num.strip()
            xse = residue.xtra # Retrieve HSE data from residue
            entry = {
                'chain': chain_id,
                'residue_num': residue_num ,
                'EXP_HSE_B_U': xse.get('EXP_HSE_B_U', None),
                'EXP_HSE_B_D': xse.get('EXP_HSE_B_D', None),
                'EXP_HSE_A_U': xse.get('EXP_HSE_A_U', None),
                'EXP_HSE_A_D': xse.get('EXP_HSE_A_D', None)
            }
            data_hse.append(entry)
    
    # Convert HSE data to a DataFrame
    return pd.DataFrame(data_hse)

# Function to calculate hydrophobicity based on a predefined scale
def calculate_HDR(model):
    chains = [chain.get_id() for chain in model.get_chains()]
    data_HDR = []

    hdr_scale = {'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
                 'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
                 'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLN': -3.5,
                 'ASN': -3.5, 'GLU': -3.5, 'ASP': -3.5, 'LYS': -3.9, 'ARG': -4.0}

    for chain_id in chains:
        chain = model[chain_id]
        for residue in chain:
            residue_num = str(residue.get_id()[1]) + residue.get_id()[2]
            residue_num = residue_num.strip()
            data_HDR.append([chain_id, residue_num, residue.get_resname()])

    df_HDR = pd.DataFrame(data_HDR, columns=['chain', 'residue_num', 'residue_name'])
    df_HDR['hydrophobicity'] = df_HDR['residue_name'].map(hdr_scale)
    df_HDR = df_HDR.drop(columns=['residue_name'])

    return df_HDR

# Function to calculate residue depth (DPX) for each residue in the structure
def calculate_dpx(structure):
    data_dpx = []
    rd = ResidueDepth(structure[0]) # Calculate residue depth for the first model

    for chain in structure[0]:
        for residue in chain:
            residue_num = str(residue.get_id()[1]) + residue.get_id()[2]
            residue_num = residue_num.strip()
            residue_depth = rd[chain.id, residue.id]
            residue_depth = round(residue_depth[0], 3)

            data_dpx.append({
                'chain': chain.get_id(),
                'residue_num': residue_num,
                'mean_dpx': residue_depth,
            })

    return pd.DataFrame(data_dpx)

# Function to calculate the protrusion index (IP) for residues
def calculate_IP(model):
    data_IP = []
    results = defaultdict(list)
    atoms = Selection.unfold_entities(model, 'A') # Get all atoms in the model
    ns = NeighborSearch(atoms) # Initialize neighbor search for proximity analysis

    for atom in atoms:
        close_atoms = ns.search(atom.coord, 10) # Search atoms within 10 Å radius
        parent = atom.get_parent()              # Get residue containing the atom
        residue_name = parent.get_resname()
        residue_df = str(parent.get_id()[1]) + parent.get_id()[2]
        residue_id = residue_df.strip()
        chain_id = parent.get_full_id()[2]
        key = (chain_id, residue_id, residue_name)
        results[key].append(len(close_atoms))

    for key, value in results.items():
        data_residues = (list(key) + value)
        chain = data_residues[0]
        residue_num = data_residues[1]
        residue_name = data_residues[2]
        values = data_residues[3:]
        mean_contacts = sum(values) / len(values)
        max_contacts = max(values)
        min_contacts = min(values)
        mean_value = indice_protrusion(mean_contacts)
        value_max = indice_protrusion(max_contacts)
        value_min = indice_protrusion(min_contacts)
        data_IP.append([chain, residue_num, mean_value, value_max, value_min])

    return pd.DataFrame(data_IP, columns=['chain', 'residue_num', 'mean_IP', 'max_IP', 'min_IP'])

# Function to calculate the Cα (alpha carbon) coordinates for each residue in a PDB model
def calculate_CA(model, pdb_file):
    chains = [chain.get_id() for chain in model.get_chains()]
    data_DSSP = []
    data_CA = []
    parser = PDBParser()

    for chain_id in chains:
        chain = model[chain_id]
        for residue in chain:
            residue_num = str(residue.get_id()[1]) + residue.get_id()[2]
            residue_num = residue_num.strip()
            res_id = residue.get_id()

            # Check if the residue contains a Cα atom
            if residue.has_id("CA"):
                ca_atom = residue["CA"]
                data_CA.append({
                    'chain': chain.id,
                    'residue_num': residue_num,
                    'CA_x': ca_atom.coord[0],
                    'CA_y': ca_atom.coord[1],
                    'CA_z': ca_atom.coord[2]
                })

    df_CA = pd.DataFrame(data_CA)

    # Return the Cα DataFrame
    return df_CA

# Helper function to calculate the protrusion index (CX)
def indice_protrusion(num_contatos):
    radius = 10.0
    mean_atomic_volume = 20.1
    volume_int = num_contatos * mean_atomic_volume
    volume_sphere = (4 / 3) * np.pi * (radius ** 3)
    volume_ext = volume_sphere - volume_int
    cx_value = round(volume_ext / volume_int, 3)

    # Return the protrusion index
    return cx_value

# Function to calculate DSSP values for secondary structure
def calculate_DSSP(pdb_file):
    filename = f'{pdb_file[:-4]}'
    subprocess.run(f"mkdssp {pdb_file} --output-format dssp > {filename}.tbl", shell=True)
    dssp_file = f'{filename}.tbl'

    # Parse the DSSP file and write relevant data to a CSV
    with open(f'{filename}_dssp.csv', "w") as output_file:
        subprocess.run(
            f"grep -A500000000000 'RESIDUE AA' {dssp_file} | "
            f"awk -F '' '{{print $6$7$8$9$10$11\",\"$12\",\"$17\",\"$104$105$106$107$108$109\",\"$110$111$112$113$114$115$116}}' | "
            f"tr -d ' '",
            shell=True,
            stdout=output_file
        )
    
    # Update the header of the CSV file
    with open(f'{filename}_dssp.csv', 'r+') as file:
        lines = file.readlines()
        lines[0] = 'residue_num,chain,secondary_structure,phi,psi\n'
        file.seek(0) 
        file.writelines(lines)
    
    df_DSSP = pd.read_csv(f'{filename}_dssp.csv')
    df_DSSP['secondary_structure'].fillna('L', inplace=True) # Fill missing values with 'L' (loop)

    # Return the DSSP DataFrame
    return df_DSSP

# Function to process a single PDB file
def process_pdb_file(pdb_file):
    # Get the structure pdb
    parser = PDBParser()
    structure = parser.get_structure("estrutura", pdb_file)
    model = structure[0]

    # Calculate various descriptors
    df_SASA_RASA = calculate_SASA_RASA(model)
    df_HSE = calculate_HSE(model)
    df_HDR = calculate_HDR(model)
    df_IP = calculate_IP(model)
    df_dpx = calculate_dpx(structure)
    df_CA = calculate_CA(model, pdb_file)

    # Merge all calculated DataFrames
    df_final = df_SASA_RASA.merge(df_HSE, on=['chain', 'residue_num'], how='left')
    df_final = df_final.merge(df_HDR[['chain', 'residue_num', 'hydrophobicity']], on=['chain', 'residue_num'], how='left')
    df_final = df_final.merge(df_IP, on=['chain', 'residue_num'], how='left')
    df_final = df_final.merge(df_dpx, on=['chain', 'residue_num'], how='left')
    df_final = df_final.merge(df_CA, on=['chain', 'residue_num'], how='left')
    df_final['residue_num'] = df_final['residue_num'].astype(str)
    

    # Save the final DataFrame to a CSV file
    return(df_final)




class NumberScheme:
    def __init__(self, scheme):
        self.scheme = scheme.lower()
        assert self.scheme in ['kabat', 'chothia', 'imgt']

    def get_range(self, domain=None):
        valid_domains = ['L1', 'L2', 'L3', 'H1', 'H2', 'H3']
        if domain not in valid_domains:
            raise ValueError(f'Given domain must be one of {valid_domains}')
        if self.scheme == 'kabat':
            return KABAT_SCHEME[domain]
        if self.scheme == 'chothia':
            return CHOTHIA_SCHEME[domain]
        if self.scheme == 'imgt':
            return IMGT_SCHEME[domain]


def number_sequence(input_seq: str, scheme: str = None) -> tuple:
    numbering, chain_type = number(sequence=input_seq, scheme=scheme)
    numbered_seq_dict = {
        ''.join(str(x) for x in t).strip(): aa for t, aa in numbering
    }
    return numbered_seq_dict, chain_type




def get_anarci_numbers(seq: str, scheme: str='imgt') -> list:
    """Number the given sequence using ANARCI and return the numbers in a list.

    Parameters
    ----------
    seq : str
        Amino acid sequence.

    scheme : str, optional
        Immunoprotein sequence numbering scheme, by default 'imgt'

    Returns
    -------
    list
        Residue numbers as numbered by ANARCI, in str type.
    """
    numbering, _, _ = anarci(
        [('input_seq', seq)], scheme=scheme, output=False
    )
    # sequence, domain, domain numbering
    domain_numbering = numbering[0][0][0]
    seq_numbers = []
    for res_num, res in domain_numbering:
        if res != '-':
            res_num_str = ''.join([str(x) for x in res_num]).strip()
            seq_numbers.append(res_num_str)
    return seq_numbers


def seq_to_gapped_seq_imgt(seq: str, imgt_numbers: list) -> str:
    """Insert gaps into the given amino acid sequence.
    A gap is inserted wherever there is no amino acid for an IMGT residue number.

    Parameters
    ----------
    seq : str
        Amino acid sequence.
    imgt_numbers : list
        Residue numbers of the given sequence in the IMGT numbering scheme.

    Returns
    -------
    str
        Amino acid sequence with gaps inserted.

    Raises
    ------
    ValueError
        Raised when the lengths of seq and imgt_numbers do not match.
    """
    if len(seq) != len(imgt_numbers):
        raise ValueError(
            'Inputs must be of the same length! ' 
            f'seq length: {len(seq)}, imgt_numbers length: {len(imgt_numbers)}'
        )
        
    ordered_allowed_imgt = [
        # residues 1 through 111, inclusive
        '{:d}'.format(i) for i in range(1, 112) 
    ] + [
        # residues 111A through 111M, inclusive
        '111' + chr(65 + i) for i in range(13)
    ] + [
        # residues 112M through 112A inclusive
        '112' + chr(65 + i) for i in range(13)
    ][::-1] + [
        # residues 112 through 128 inclusive
        '{:d}'.format(i) for i in range(112, 129)
    ]

    imgt_dict = {name: idx for idx, name in enumerate(ordered_allowed_imgt)}
    out_seq = ['-'] * len(imgt_dict)
    for aa, imgt_n in zip(seq, imgt_numbers):
        out_seq[imgt_dict[imgt_n]] = aa

    return ''.join(out_seq)


def onehot_encode(gapped_seq: str, flatten: bool=False) -> np.ndarray:
    """Onehot-encode the given sequence.

    Parameters
    ----------
    gapped_seq : str
        Amino acid sequence.
    flatten : bool, optional
        Whether to flatten the one-hot matrix into a vector, by default False.

    Returns
    -------
    np.ndarray
        A vector if flatten is True, otherwise a matrix.
    """
    aa_to_idx = {
        aa: idx for idx, aa in enumerate('ACDEFGHIKLMNPQRSTVWY' + '-')
    }
    
    one_hot = np.zeros(shape=(len(gapped_seq), 21))
    for i, aa in enumerate(gapped_seq):
        one_hot[i, aa_to_idx[aa]] = 1.
    
    if flatten:
        # by default the order is 'C', i.e. row-major
        return one_hot.flatten()
    return one_hot


def calculate_pi(seq: Union[str, List[str]]) -> float:
    """Calculates the theoretical pI based on the given sequence.

    Based on formulas from this link: http://isoelectric.org/theory.html

    Based on `scipy.optimize.minimize_scalar`. It is more robust than a grid search for
    the pH value that results in a net charge within a certain tolerance around 0. In testing,
    this implmentation fixed all four cases where the grid search-based method failed.

    Note that for antibodies you can either pass sequences of the four subunits as a list
    of length four, or concatenate the four sequences into a single string and pass the
    concatenated string.

    Parameters
    ----------
    seq : Union[str, List[str]]
        Amino acid sequence of the protein. Can be a single string for monomers or a list
        of sequences for multimers.

    Returns
    -------
    float
        Theoretical isoelectric point (pI) of the antibody, calucated based on the sequence.
        Returns np.nan if scipy.optimize.minimize_scalar() failed.
    """
    if isinstance(seq, str):
        N_term_count = 1
        C_term_count = 1
        full_seq = seq
    else:
        N_term_count = len(seq)
        C_term_count = len(seq)
        full_seq = ''.join(seq)

    aa_counts = {
        'N_term': N_term_count,
        'C_term': C_term_count,
        'D': 0,
        'E': 0,
        'H': 0,
        'Y': 0,
        'K': 0,
        'R': 0
    }

    for aa in full_seq:
        if aa in aa_counts:
            aa_counts[aa] += 1

    def net_charge(x):
        neg_charge = 0
        pos_charge = 0
        for aa in pKa_dict.keys():
            aa_pKa, aa_charge = pKa_dict[aa]
            aa_count = aa_counts[aa]
            if aa_charge < 0:
                neg_charge += aa_count * (-1 / (1 + np.power(10, aa_pKa - x)))
            else:
                pos_charge += aa_count * (1 / (1 + np.power(10, x - aa_pKa)))
        net_charge = neg_charge + pos_charge
        return net_charge ** 2

    # pH, especially protein pI, rarely falls outside of the (0, 14) range
    opt_results = optimize.minimize_scalar(net_charge, bounds=(0, 14))

    if opt_results.success:
        return opt_results.x
    else:
        return np.nan


def extract_fv_seq(seq: str, scheme: str='imgt') -> str:
    """Uses ANARCI to extract sequence of the Fv domain.

    Parameters
    ----------
    seq : str
        Amino acid sequence.
    scheme : str, optional
        Sequence numbering scheme, by default 'imgt'

    Returns
    -------
    str
        Extracted sequence of the Fv domain.
    """
    numberings, _, _ = anarci(
        [('tmp', seq)], scheme=scheme, output=False
    )
    numbering = numberings[0]
    fv_seq = ''.join([a for _, a in numbering[0][0] if a != '-'])
    return fv_seq


def calculate_seq_charge(seq: str, pH: float=7.4) -> float:
    """Calculates the charge at the given pH based on sequence.

    Parameters
    ----------
    seq : str
        Amino acid sequence.
    pH : float, optional
        pH condition, by default 7.4

    Returns
    -------
    float
        Charge at the given pH.

    """
    if pH < pKa_dict['H'][0]:
        return seq.count('H') + seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')
    else:
        return seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')

    
from anarci import number

class SeqAnnotation:
    def __init__(self, seq: str, scheme: str = 'imgt'):
        self.seq = seq
        self.number_scheme = NumberScheme(scheme)
        numbered_seq_dict, chain_type = number_sequence(seq, scheme)
        self.numbered_seq_dict = numbered_seq_dict
        self.chain_type = chain_type

    def get_cdr_seq(self, cdr: str) -> str:
        start, end = self.number_scheme.get_range(domain=cdr.upper())
        return ''.join(
            self.numbered_seq_dict[str(i)] for i in range(start, end + 1)
        )

    
def get_charge(residue):
    return AA_CHARGE.get(residue, 0.0)

def compute_chain_charge_asymmetry(df):
    df['charge'] = df['residue_name'].map(get_charge)
    surface_df = df[df['RASA'] > 20]
    charge_H = surface_df[surface_df['chain'] == 'H']['charge'].sum()
    charge_L = surface_df[surface_df['chain'] == 'L']['charge'].sum()
    asymmetry = abs(charge_H - charge_L) / (abs(charge_H) + abs(charge_L) + 1e-6)
    return asymmetry

def compute_dipole_vector_symmetry(df):
    df['charge'] = df['residue_name'].map(get_charge)
    surface_df = df[df['RASA'] > 20]
    
    def dipole_vector(sub_df):
        coords = sub_df[['CA_x', 'CA_y', 'CA_z']].values
        charges = sub_df['charge'].values
        return np.sum(charges[:, None] * coords, axis=0)

    v_H = dipole_vector(surface_df[surface_df['chain'] == 'H'])
    v_L = dipole_vector(surface_df[surface_df['chain'] == 'L'])
    
    norm_H = np.linalg.norm(v_H)
    norm_L = np.linalg.norm(v_L)
    if norm_H == 0 or norm_L == 0:
        return np.nan  # avoid division by zero
    cos_theta = np.dot(v_H, v_L) / (norm_H * norm_L)
    return cos_theta

def compute_plane_symmetry_charge_imbalance(df):
    df['charge'] = df['residue_name'].map(get_charge)
    surface_df = df[df['RASA'] > 20]
    scm_symmetry_score = np.sum(surface_df['charge'] * surface_df['CA_x'])
    return scm_symmetry_score

def compute_charge_entropy(df):
    df['charge'] = df['residue_name'].map(get_charge)
    surface_df = df[df['RASA'] > 20]

    def assign_octant(row):
        x, y, z = row['CA_x'], row['CA_y'], row['CA_z']
        return (int(x > 0), int(y > 0), int(z > 0))

    surface_df['octant'] = surface_df.apply(assign_octant, axis=1)
    grouped = surface_df.groupby('octant')['charge'].sum()
    total_charge = grouped.abs().sum()
    if total_charge == 0:
        return 0
    probs = (grouped.abs() / total_charge).values
    entropy = -np.sum([p * np.log(p) for p in probs if p > 0])
    return entropy







# Functions
def compute_molecular_weight(df):
    return df.groupby('chain')['residue_name_U'].apply(lambda x: sum(AA_MW.get(res, 0) for res in x)).to_dict()

def compute_total_molecular_weight(df):
    return sum(AA_MW.get(res, 0) for res in df['residue_name_U'])

def compute_average_residue_volume(df):
    return df.groupby('chain')['residue_name_U'].apply(lambda x: sum(AA_VOLUME.get(res, 0) for res in x) / len(x)).to_dict()

def compute_total_volume_per_chain(df):
    return df.groupby('chain')['residue_name_U'].apply(lambda x: sum(AA_VOLUME.get(res, 0) for res in x)).to_dict()

def compute_net_charge_at_ph(df, ph=7.4):
    return df.groupby('chain')['residue_name_U'].apply(lambda x: sum(AA_CHARGE.get(res, 0) for res in x)).to_dict()

def compute_average_pI(df):
    return df.groupby('chain')['residue_name_U'].apply(lambda x: sum(AA_PI.get(res, 0) for res in x) / len(x)).to_dict()




# Function to compute compactness
def compute_compactness(df):
    coords = df[['CA_x', 'CA_y', 'CA_z']].values
    centroid = coords.mean(axis=0)
    distances = np.linalg.norm(coords - centroid, axis=1)
    return distances.mean()

# Surface patch score: SASA-weighted mean hydrophobicity
def compute_surface_patch_score(df, rasa_threshold=20.0):
    surface = df[df['RASA'] > rasa_threshold]
    weighted = surface['SASA'] * surface['hydrophobicity']
    return weighted.sum() / surface['SASA'].sum() if surface['SASA'].sum() > 0 else np.nan

# Amino acid composition stats
def compute_aa_composition(df):
    total = len(df)
    residue_names = df['residue_name'].str.upper()
    hydrophobic = residue_names.isin(AA_CLASSES['hydrophobic']).sum() / total
    apolar = residue_names.isin(AA_CLASSES['apolar']).sum() / total
    charged = residue_names.isin(AA_CLASSES['charged']).sum() / total
    return {'%hydrophobic': hydrophobic, '%apolar': apolar, '%charged': charged}

# Aromatic residue count
def compute_aromatic_count(df):
    return df['residue_name'].str.upper().isin(AA_CLASSES['aromatic']).sum()

# 20-dim AA frequency vector
def compute_aa_frequency_vector(df):
    aa_counts = df['residue_name'].str.upper().value_counts()
    freq_vector = np.array([aa_counts.get(aa, 0) for aa in AA_LIST])
    freq_vector = freq_vector / freq_vector.sum()
    return freq_vector



# Threshold for surface classification
SURFACE_RASA_THRESHOLD = 20.0

def hydrophobicity_skew(df):
    return df['hydrophobicity'].max() - df['hydrophobicity'].min()

def hydrophobic_surface_fraction(df, threshold=SURFACE_RASA_THRESHOLD):
    surface = df[df['RASA'] > threshold]
    if not surface.empty:
        return surface['hydrophobicity'].mean()
    else:
        return float('nan')

def ip_stats(df):
    return {
        "mean_IP_mean": df['mean_IP'].mean(),
        "mean_IP_max": df['mean_IP'].max(),
        "mean_IP_min": df['mean_IP'].min()
    }

def mean_dpx(df):
    return df['mean_dpx'].mean()

def surface_vs_buried_ip_diff(df, threshold=SURFACE_RASA_THRESHOLD):
    surface = df[df['RASA'] > threshold]
    buried = df[df['RASA'] <= threshold]
    if not surface.empty and not buried.empty:
        return surface['mean_IP'].mean() - buried['mean_IP'].mean()
    else:
        return float('nan')

def compute_antibody_descriptors(df):
    results = {}

    # Mean/Total SASA
    results["total_SASA"] = df["SASA"].sum()
    results["mean_SASA"] = df["SASA"].mean()

    # Mean/Total RASA
    results["total_RASA"] = df["RASA"].sum()
    results["mean_RASA"] = df["RASA"].mean()

    # % of residues with RASA > 20% (exposed)
    exposed = df["RASA"] > 20
    results["percent_exposed_residues"] = 100 * exposed.sum() / len(df)

    # SASA ratio heavy/light chain
    sasa_H = df[df["chain"] == "H"]["SASA"].sum()
    sasa_L = df[df["chain"] == "L"]["SASA"].sum()
    results["SASA_ratio_H_to_L"] = sasa_H / sasa_L if sasa_L != 0 else np.nan

    # Mean EXP_HSE_B_U / B_D / A_U / A_D
    for col in ["EXP_HSE_B_U", "EXP_HSE_B_D", "EXP_HSE_A_U", "EXP_HSE_A_D"]:
        results[f"mean_{col}"] = df[col].mean(skipna=True)

    # % buried residues (e.g. EXP_HSE_B_U + EXP_HSE_B_D < 15)
    hse_sum = df["EXP_HSE_B_U"].fillna(0) + df["EXP_HSE_B_D"].fillna(0)
    results["percent_buried_residues"] = 100 * (hse_sum < 15).sum() / len(df)

    # Asymmetry HSE score: (U - D) / (U + D)
    asymmetry = (df["EXP_HSE_B_U"] - df["EXP_HSE_B_D"]) / (
        df["EXP_HSE_B_U"] + df["EXP_HSE_B_D"]
    ).replace(0, np.nan)
    results["mean_HSE_asymmetry"] = asymmetry.mean(skipna=True)

    # Mean hydrophobicity
    results["mean_hydrophobicity"] = df["hydrophobicity"].mean()

    return results



def get_forcefield(kind='amber'):
    if kind=='amber':
        return ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    else:
        return ForceField('charmm36.xml','charmm36/tip3p-pme-b.xml')

def simple_system(path):
    
    pdb = PDBFile(path)
    forcefield = get_forcefield(kind='charmm')
    system = forcefield.createSystem(
        pdb.topology
    )
    return system, pdb.topology


def clean_structure(
    pdb_path,
    chains_keep=None,
    ph=6.0,
    ionic=0.015,
    remove_water=True,
    add_solvent=False,
    solvent_box=None
):
    fix_name = pdb_path.split('.')[0]+'_fixed_{:.2f}.pdb'.format(ph)
        
    fixer = PDBFixer(filename=pdb_path)
    
    if chains_keep is None:
        chains_keep = ['H','L']
    if isinstance(chains_keep,list):
        chains_keep = set(chains_keep)
        
    chains_remove_dict = {c.id: c.index for c in fixer.topology.chains()}
    chains_remove = list(set([c.id for c in fixer.topology.chains()]) - chains_keep )
    chain_id_remove = [chains_remove_dict[k] for k in chains_remove]
    
    fixer.removeChains(chain_id_remove)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    het_arg = not remove_water
    fixer.removeHeterogens(het_arg) #False removes all hetatm inc. water, can be necessary just because of pdb chain formatting. True remove hetatm, but not water.
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    fix_name_no_h = pdb_path.split('.')[0]+'_fixed_noH.pdb'
    PDBFile.writeFile(fixer.topology, fixer.positions, open(fix_name_no_h, 'w'))
    
    fixer.addMissingHydrogens(ph)

    PDBFile.writeFile(fixer.topology, fixer.positions, open(fix_name, 'w'))

    maxSize=0*unit.angstrom
    if add_solvent:
        if solvent_box is None:
            fixer.addSolvent(fixer.topology.getUnitCellDimensions(), ionicStrength=ionic*unit.molar)
        else:
            maxSize = max(max((pos[i] for pos in fixer.positions))-min((pos[i] for pos in fixer.positions)) for i in range(3))
            print('maxSize = ', maxSize)
            maxSize += solvent_box*unit.angstrom
            boxSize = maxSize*Vec3(1, 1, 1)
            #fixer.topology.setUnitCellDimensions(boxSize)
            #fixer.addSolvent(fixer.topology.getUnitCellDimensions(), ionicStrength=ionic*unit.molar)
            fixer.addSolvent(boxSize,  ionicStrength=ionic*unit.molar)
            
        fix_name_solv = pdb_path.split('.')[0]+'_fixed_{:.2f}_solvent.pdb'.format(ph)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(fix_name_solv, 'w'))
    return fix_name,maxSize

def setup_minimization_system(
    path,
    field='charmm',
    ionic=0.015,
    padding=1.2,
    bond_cutoff=1.2,
    temp=300,
    dt=0.001,
    
):
    pdb = PDBFile(path)

    forcefield = get_forcefield(kind='charmm')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, padding=padding*unit.nanometers, ionicStrength=ionic*unit.molar)
    
    system = forcefield.createSystem(
        modeller.topology, 
        nonbondedMethod=PME, # ideally construct own PME and addd switching distance to that obj, then pass in here
        nonbondedCutoff=bond_cutoff*unit.nanometer, 
        constraints=None 
    )
    
    integrator = LangevinMiddleIntegrator(temp*unit.kelvin, 1/unit.picosecond, dt*unit.picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    return modeller, system, simulation

def get_partial_charges_system(system,topology,chains='AB'):
    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    atoms = [a for a in topology.atoms()]
    charges = []
    for i in range(system.getNumParticles()):
        if atoms[i].residue.chain.id in chains:
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charges.append(charge)
    return charges

def get_partial_charges(
    pdb_path,
    forcefield,
    chains='AB'
):
    """ Get partial charges on all atoms
    
    Will skip waters if solvent added to structure
    
    Assumes two chain structure with chain_ids A and B in that order. 
     - How pdbfixer will output for two chains selected
    
    """
    pdb = PDBFile(pdb_path)
    system = forcefield.createSystem(pdb.topology)
    nonbonded = [f for f in system.getForces() if isinstance(f, NonbondedForce)][0]
    atoms = [a for a in pdb.topology.atoms()]
    charges = []
    for i in range(system.getNumParticles()):
        if atoms[i].residue.chain.id in chains:
            charge, sigma, epsilon = nonbonded.getParticleParameters(i)
            charges.append(charge)
    return charges

def minimize(
    pdb_path,
    forcefield,
    nb_cutoff=12,
    temp=300,
    max_iter=200,
):
    pdb = PDBFile(pdb_path)
    system = forcefield.createSystem(
    pdb.topology, 
        nonbondedMethod=PME, 
        nonbondedCutoff=(nb_cutoff/10.)*nanometer, 
        constraints=None 
    )
    integrator = LangevinMiddleIntegrator(temp*kelvin, 1/picosecond, 0.004*picoseconds)
    simulation = Simulation(pdb.topology, system, integrator)
    simulation.context.setPositions(pdb.positions)
    
 
def apply_sasa(pdb_path,join_models=True,inc_hydrogen=False,n_slices=None):
    sasa_options = {
        'hetatm' : False,          # False: skip HETATM
                                   # True: include HETATM

        'hydrogen' : inc_hydrogen,        # False: ignore hydrogens
                                   # True: include hydrogens

        'join-models' : join_models,     # False: Only use the first MODEL
                                   # True: Include all MODELs

        'skip-unknown' : False,    # False: Guess radius for unknown atoms
                                   #     based on element
                                   # True: Skip unknown atoms

        'halt-at-unknown' : False  # False: set radius for unknown atoms,
                                   #    that can not be guessed to 0.
                                   # True: Throw exception on unknown atoms.
    }
    
    structure = freesasa.Structure(pdb_path,None,sasa_options)
    if n_slices is not None:
        calc_params = {
            'algorithm' : freesasa.LeeRichards,
            'n-slices' : 100
        }
        result = freesasa.calc(
            structure,
            freesasa.Parameters(calc_params)
        )
    else:
        result = freesasa.calc(structure)
    return result

def write_sasa_to_structure(structure,result,sasa_type='sideChain'):
    for chain in structure[0].get_chains():
        areas = result.residueAreas()[chain._id]
        for r in chain.get_residues():
            r_name = (str(r._id[1])+r._id[2]).strip()
            for atom in r:
                if sasa_type=='total':
                    atom.bfactor = areas[r_name].total #sideChain
                elif sasa_type=='sideChain':
                    atom.bfactor = areas[r_name].sideChain
    return structure


@njit(parallel=True)
def get_scm_numba(a_mainchain,a_sasa,a_coords,charge_values,d_cutoff,sasa_cutoff):
    """ SCM per atom in structure, with precalculated charges in MD forcefield
    """
    scms = np.zeros((len(a_mainchain),1))
    for i in prange(len(a_mainchain)):
        tmp_scm = 0
        for j in np.arange(len(a_mainchain)):
            if not a_mainchain[j] and i!=j:
                d = np.sqrt(np.sum(np.square(a_coords[i,:]-a_coords[j,:])))
                sasa = a_sasa[j]
                if d<d_cutoff and sasa>sasa_cutoff:
                    tmp_scm += charge_values[j]
        scms[i] = tmp_scm
    return scms

def get_scm(structure, charges, d_cutoff=10, sasa_cutoff=10):
    """ SCM per atom in structure, with precalculated charges in MD forcefield
    
    internally uses numba for faster calculation
    """
    a_mainchain = np.array([a.id in main_chain for a in structure.get_atoms()])
    a_sasa = np.array([a.get_bfactor() for a in structure.get_atoms()]).astype('float32')
    a_coords = np.array([a.coord for a in structure.get_atoms()])
    a_charges = np.array([c._value for c in charges])
    
    scms = get_scm_numba(a_mainchain,a_sasa,a_coords,a_charges,d_cutoff,sasa_cutoff)
    return scms

def score_from_scms(scms):
    """ SCM score from scm values per atom"""
    return abs(np.sum(scms[scms<0]))

def scm_score(structure, charges, d_cutoff=10, sasa_cutoff=10):
    """ for a structure and charges per atom, get final SCM score for structure """
    scms = get_scm(structure, charges, d_cutoff=d_cutoff, sasa_cutoff=sasa_cutoff)
    return score_from_scms(scms)

def raw_pdb_scm_scoring(pdb_path):
    """ Requires PDB with Hydrogens already added
    
    Will not perform any MD, will use CHARMM for potential
    Expects chains named H and L for heavy and light chains
    
    parameters
    -----------
    pdb_path: string
        path to pdb file to get SCM for. must have hydrogens already.
        must have heavy and ligh chains named H and L.
        
    returns
    ---------
    SCMs: Tuple of (scms,scm_score)
        scms: scm values per atom, scm_score, scm_score for whole structure
    """
    structure_ = load_structure(pdb_path)
    sasa_result = apply_sasa(pdb_path)
    sasa_structure = write_sasa_to_structure(structure_,sasa_result)
    
    system,topology = simple_system(pdb_path)
    charges = get_partial_charges_system(system,topology,chains='HL')
    
    scms = get_scm(sasa_structure, charges)
    scm_score = score_from_scms(scms)
    return scms, scm_score

def load_structure(path,name='tmp'):
    parser = PDBParser()
    structure = parser.get_structure(name, path)
    return structure

def scm_pos_neg(scms):
    """
    Compute SCM+ (positive SCM sum) and SCM− (negative SCM absolute sum).

    Parameters
    ----------
    scms : np.ndarray
        SCM values per atom. Shape: (N, 1) or (N,)

    Returns
    -------
    dict
        {
            'SCM_pos': float,
            'SCM_neg': float
        }
    """
    scms = scms.flatten()  # ensure 1D
    scm_pos = np.sum(scms[scms > 0])
    scm_neg = np.abs(np.sum(scms[scms < 0]))
    return {
        'SCM_pos': float(scm_pos),
        'SCM_neg': float(scm_neg)
    }




from sklearn.cluster import DBSCAN

def scm_patch_count(structure, scms, threshold=-1.5, eps=6.0, min_samples=2):
    coords = np.array([a.coord for a in structure.get_atoms()])
    patch_coords = coords[scms.flatten() < threshold]
    if len(patch_coords) == 0:
        return 0
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(patch_coords)
    return len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)




def scm_patch_stats(structure, scms, threshold=-1.5, eps=4.0, min_samples=2):
    """
    Cluster atoms by spatial proximity and SCM sign.
    Return counts of total patches, positive, negative, and hydrophobic ones.
    """
    atom_coords = np.array([a.coord for a in structure.get_atoms()])
    atom_scm = scms.flatten()
    atom_residues = [a.get_parent() for a in structure.get_atoms()]
    atom_residues = np.asarray(atom_residues, dtype=object)  # <-- FIXED

    patch_types = np.where(atom_scm < threshold, 'neg',
                   np.where(atom_scm > -threshold, 'pos', 'none'))

    valid_mask = patch_types != 'none'
    valid_coords = atom_coords[valid_mask]
    valid_types = patch_types[valid_mask]
    valid_residues = atom_residues[valid_mask]

    if len(valid_coords) == 0:
        return {'patch_total': 0, 'patch_neg': 0, 'patch_pos': 0, 'patch_hydrophobic': 0}

    # Cluster patches
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(valid_coords)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    patch_neg = 0
    patch_pos = 0
    patch_hydrophobic = 0

    for label in set(labels):
        if label == -1:
            continue  # noise
        indices = np.where(labels == label)[0]
        patch_type_counts = {'neg': 0, 'pos': 0}

        for idx in indices:
            patch_type_counts[valid_types[idx]] += 1

        # Majority vote: neg vs pos
        if patch_type_counts['neg'] > patch_type_counts['pos']:
            patch_neg += 1
        else:
            patch_pos += 1

        # Check if patch contains a hydrophobic residue
        patch_resnames = set(valid_residues[idx].resname for idx in indices)
        if any(r in HYDROPHOBIC_RESIDUES for r in patch_resnames):
            patch_hydrophobic += 1

    return {
        'patch_neg': patch_neg,
        'patch_pos': patch_pos,
        'patch_hydrophobic': patch_hydrophobic
    }


def scm_asymmetry_index(structure, scms):
    from scipy.spatial.distance import euclidean

    atom_coords = np.array([a.coord for a in structure.get_atoms()])
    scm_vals = scms.flatten()
    neg_coords = atom_coords[scm_vals < 0]
    if len(neg_coords) == 0:
        return 0.0
    neg_center = np.mean(neg_coords, axis=0)
    overall_center = np.mean(atom_coords, axis=0)
    return float(euclidean(neg_center, overall_center))


def computeDescriptors(is_fv, isotype= 'igg1', lc_type= 'kappa', 
                       pH= 7.4, HC_chain_ID = "H", LC_chain_ID = "L",
                       pdb_file = "test.pdb", HC = "", LC = ""):
    
    # Format - Precompute
    pdb_file = str(pdb_file)
    featurizer = StructFeaturizer(pdb_file)
    df_final_AA = process_pdb_file(pdb_file)
    seq_features = get_all_seq_features(heavy_seq=HC, light_seq=LC, is_fv=is_fv, isotype=isotype, lc_type=lc_type, pH=pH)
    SCM_atom_list, SCM_score = raw_pdb_scm_scoring(pdb_file)
    structure_ = load_structure(pdb_file)
    scms, _ = raw_pdb_scm_scoring(pdb_file)
    scm_scores = scm_pos_neg(SCM_atom_list)
    desc = scm_patch_stats(structure_, scms)
    scm_asym = scm_asymmetry_index(structure_, scms)
    net_charge = featurizer.net_charge()
    df_final_AA['residue_name_U'] = df_final_AA['residue_name'].str.upper()

    if LC == '':
        LC_chain_ID=None

    if HC == '':
        HC_chain_ID=None


    if HC_chain_ID:
        vh_seq = extract_fv_seq(HC)
        seq_HC = SeqAnnotation(seq = HC)
        featurizer.seq_HC = HC
        
    if LC_chain_ID:
        vl_seq = extract_fv_seq(LC)
        seq_LC = SeqAnnotation(seq = LC)
        featurizer.seq_LC = LC
    
    # Residue-level descriptors
    mw_chain = compute_molecular_weight(df_final_AA)
    mw_total = compute_total_molecular_weight(df_final_AA)
    avg_volume = compute_average_residue_volume(df_final_AA)
    total_volume = compute_total_volume_per_chain(df_final_AA)
    # net_charge = compute_net_charge_at_ph(df_final_AA)
    avg_pI = compute_average_pI(df_final_AA)
    compactness = compute_compactness(df_final_AA)
    surface_patch_score = compute_surface_patch_score(df_final_AA)
    aa_composition = compute_aa_composition(df_final_AA)
    aromatic_count = compute_aromatic_count(df_final_AA)
    aa_frequency_vector = compute_aa_frequency_vector(df_final_AA)
    hydrophobicity_skew_desc = hydrophobicity_skew(df_final_AA)
    surface_vs_buried_ip_diff_desc = surface_vs_buried_ip_diff(df_final_AA)
    mean_dpx_desc = mean_dpx(df_final_AA)
    hydrophobic_surface_fraction_desc = hydrophobic_surface_fraction(df_final_AA)
    asymmetry = compute_chain_charge_asymmetry(df_final_AA)
    dipole_cos = compute_dipole_vector_symmetry(df_final_AA)
    symmetry_score = compute_plane_symmetry_charge_imbalance(df_final_AA)
    entropy = compute_charge_entropy(df_final_AA)
    exposed = df_final_AA["RASA"] > 20

    results = {}
    # Mean/Total SASA
    results["total_SASA"] = df_final_AA["SASA"].sum()
    results["mean_SASA"] = df_final_AA["SASA"].mean()

    # Mean/Total RASA
    results["total_RASA"] = df_final_AA["RASA"].sum()
    results["mean_RASA"] = df_final_AA["RASA"].mean()

    # % of residues with RASA > 20% (exposed)
    results["percent_exposed_residues"] = 100 * exposed.sum() / len(df_final_AA)

    # Mean EXP_HSE_B_U / B_D / A_U / A_D
    for col in ["EXP_HSE_B_U", "EXP_HSE_B_D", "EXP_HSE_A_U", "EXP_HSE_A_D"]:
        results[f"mean_{col}"] = df_final_AA[col].mean(skipna=True)

    # % buried residues (e.g. EXP_HSE_B_U + EXP_HSE_B_D < 15)
    hse_sum = df_final_AA["EXP_HSE_B_U"].fillna(0) +df_final_AA["EXP_HSE_B_D"].fillna(0)
    results["percent_buried_residues"] = 100 * (hse_sum < 15).sum() / len(df_final_AA)

    # Asymmetry HSE score: (U - D) / (U + D)
    asymmetry = (df_final_AA["EXP_HSE_B_U"] - df_final_AA["EXP_HSE_B_D"]) / (
        df_final_AA["EXP_HSE_B_U"] + df_final_AA["EXP_HSE_B_D"]).replace(0, np.nan)
    results["mean_HSE_asymmetry"] = asymmetry.mean(skipna=True)

    # Mean hydrophobicity
    results["mean_hydrophobicity"] = df_final_AA["hydrophobicity"].mean()
    
    # Sequence descriptors
    results = {}
    for descriptor, val in seq_features.items():
        results[f"{descriptor}_pH{pH}"] = val
    
    # Moments
    results["dipole_moment"] = featurizer.dipole_moment()
    results["hyd_moment"] = featurizer.hyd_moment()
    results["RM_dipole_to_hydrophobic"] = results["dipole_moment"] / results["hyd_moment"]

    # ASA
    results["asa_aromatic"] = featurizer.aromatic_asa()
    results["exposed_aromatic"] = featurizer.exposed_aromatic()

    # Surface areas
    results["asa_hyd"] = featurizer.hyd_asa()
    results["asa_hph"] = featurizer.hph_asa()
    results["pos_asa"] = featurizer.pos_asa()
    results["neg_asa"] = featurizer.neg_asa()
    results["ratio_charged_to_hydrophobic_surface_area"] = (results["neg_asa"] + results["pos_asa"]) / results["asa_hph"]

    # ANN
    results["pos_ann_index"] = featurizer.ann_index(prop='pos', rsa_cutoff=0.05, n=1000)
    results["neg_ann_index"] = featurizer.ann_index(prop='neg', rsa_cutoff=0.05, n=1000)
    results["aromatic_ann_index"] = featurizer.ann_index(prop='aro', rsa_cutoff=0.05, n=1000)

    # Ripley K
    results["pos_ripley_k"] = featurizer.ripley_k(prop='pos', distance=6.0, n=1000)
    results["neg_ripley_k"] = featurizer.ripley_k(prop='neg', distance=6.0, n=1000)
    results["aromatic_ripley_k"] = featurizer.ripley_k(prop='aro', distance=6.0, n=1000)

    # Formal VH-VL charge diff
    results["Fv_chml"] = featurizer.fv_chml()

    # Asymmetry and imbalance and surface
    results["sum_HSE_asymmetry"] = asymmetry.sum(skipna=True)
    results["Dipole_Vector_Cosine"] = dipole_cos
    results["Plane_Symmetry_Imbalance"] = symmetry_score
    results["Charge_Distribution_Entropy"] = entropy
    results["compactness"] = compactness
    results["surface_patch_score"] = surface_patch_score
    
    # ratio and count
    results["aromatic_count"] = aromatic_count
    results["aa_%hydrophobic"] = aa_composition["%hydrophobic"]
    results["aa_%apolar"] = aa_composition["%apolar"]
    results["aa_%charged"] = aa_composition["%charged"]

    # hydrophobicity
    results["hydrophobicity_skew"] = hydrophobicity_skew_desc
    results["surface_vs_buried_ip_diff"] = surface_vs_buried_ip_diff_desc
    results["mean_dpx"] = mean_dpx_desc
    results["hydrophobic_surface_fraction"] = hydrophobic_surface_fraction_desc

    # patch
    results["patch_neg"] = desc["patch_neg"]
    results["patch_pos"] = desc["patch_pos"]
    results["patch_hydrophobic"] = desc["patch_hydrophobic"]
    
    # scm
    results["scm_asymmetry_index"] = scm_asym
    results["SCM_pos"] = scm_scores["SCM_pos"]
    results["SCM_neg"] = scm_scores["SCM_neg"]

    # CDRs
    results["cdr_aromatic"] = featurizer.aromatic_cdr()
    results["total_MW"] = mw_total

    # if HC 
    if HC_chain_ID:
        # SASA ratio heavy/light chain
        sasa_H = df_final_AA[df_final_AA["chain"] == HC_chain_ID]["SASA"].sum()
        results["net_charge_H"] = sum(net_charge[HC_chain_ID])
        h1 = seq_HC.get_cdr_seq("H1")
        h2 = seq_HC.get_cdr_seq("H2")
        h3 = seq_HC.get_cdr_seq("H3")
        results["Length_CDRH1"] = len(h1.replace("-",""))
        results["Length_CDRH2"] = len(h2.replace("-",""))
        results["Length_CDRH3"] = len(h3.replace("-",""))
        results["molecular_weight_VH"] = mw_chain[HC_chain_ID]
        results["total_volume_VH"] = avg_volume[HC_chain_ID]
        results["net_charge_pH7.4_VH"] = total_volume[HC_chain_ID]
        results["average_pI_VH"] = avg_pI[HC_chain_ID]
        results["net_charge_H_mean"] = np.mean(net_charge[HC_chain_ID])
    else:
        results["net_charge_VH"] = None
        results["Length_CDRH1"] = None
        results["Length_CDRH2"] = None
        results["Length_CDRH3"] = None
        results["molecular_weight_VH"] = None
        results["total_volume_VH"] = None
        results["net_charge_pH7.4_VH"] = None
        results["average_pI_VH"] = None
        results["net_charge_H_mean"] = None
        sasa_H = None
        h1 = None
        h2 = None
        h3 = None
        
    # if LC 
    if LC_chain_ID:
        sasa_L = df_final_AA[df_final_AA["chain"] == LC_chain_ID]["SASA"].sum()
        results["net_charge_L"] = sum(net_charge[LC_chain_ID]) 
        l1 = seq_LC.get_cdr_seq("L1")
        l2 = seq_LC.get_cdr_seq("L2")
        l3 = seq_LC.get_cdr_seq("L3")
        results["Length_CDRL1"] = len(l1.replace("-",""))
        results["Length_CDRL2"] = len(l2.replace("-",""))
        results["Length_CDRL3"] = len(l3.replace("-",""))
        results["molecular_weigddht_VL"] = mw_chain[LC_chain_ID]
        results["total_volume_VL"] = avg_volume[LC_chain_ID]
        results["net_charge_pH7.4_VL"] = total_volume[LC_chain_ID]
        results["average_pI_VL"] = avg_pI[LC_chain_ID]
        results["net_charge_L_mean"] = np.mean(net_charge[LC_chain_ID])

    else:
        results["net_charge_L"] = None
        results["Length_CDRL1"] = None
        results["Length_CDRL2"] = None
        results["Length_CDRL3"] = None
        results["molecular_weight_VL"] = None
        results["total_volume_VL"] = None
        results["net_charge_pH7.4_VL"] = None
        results["average_pI_VL"] = None
        results["net_charge_L_mean"] = None

        sasa_L = None
        l1 = None
        l2 = None
        l3 = None

    # if LC and HC
    if LC_chain_ID and LC_chain_ID:
        results["SASA_ratio_H_to_L"] = sasa_H / sasa_L if sasa_L != 0 else np.nan
        results["net_charge_sum"] = results["net_charge_H"] + results["net_charge_L"]
        a = h1+ h2+ h3+ l1+ l2+ l3
        a = a.replace("-","")
        results["Total_CDR_Length"] = len(a)
    else:
        results["SASA_ratio_H_to_L"] = None
        results["net_charge_sum"] = None
        results["Total_CDR_Length"] = None
        a = None
        
    # Convert to DataFrame
    df_summary = pd.DataFrame([results])
    return(df_summary, df_final_AA)

