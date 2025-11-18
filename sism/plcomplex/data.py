import os
from itertools import zip_longest
from typing import Sequence, Union
import pickle
import numpy as np
import torch
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import ChemicalFeatures
import rdkit

from torch import Tensor
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_geometric.data import Data

import lightning as L

from sism.plcomplex.stats import Statistics, compute_all_statistics

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(array, path, exist_ok=True):
    if exist_ok:
        with open(path, "wb") as f:
            pickle.dump(array, f)
    else:
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(array, f)

IndexType = Union[slice, Tensor, np.ndarray, Sequence]

full_atom_encoder = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
}

atom_decoder = {v: k for k, v in full_atom_encoder.items()}

fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

RDLogger.DisableLog("rdApp.*")

x_map = {
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
    "hybridization": [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP2D,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ],
    "is_h_donor": [False, True],
    "is_h_acceptor": [False, True],
}


def mol_to_torch_geometric(
    mol,
    atom_encoder,
    smiles=None,
    remove_hydrogens: bool = False,
    cog_proj: bool = True,
    add_ad=True,
    add_pocket=False,
    **kwargs,
):
    if remove_hydrogens:
        # mol = Chem.RemoveAllHs(mol)
        mol = Chem.RemoveHs(
            mol
        )  # only remove (explicit) hydrogens attached to molecular graph
        Chem.Kekulize(mol, clearAromaticFlags=True)

    if smiles is None:
        smiles = Chem.MolToSmiles(mol)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    if remove_hydrogens:
        assert max(bond_types) != 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    if cog_proj:
        pos = pos - torch.mean(pos, dim=0, keepdim=True)
    atom_types = []
    all_charges = []
    is_aromatic = []
    is_in_ring = []
    sp_hybridization = []

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(
            atom.GetFormalCharge()
        )  # TODO: check if implicit Hs should be kept
        is_aromatic.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        is_in_ring.append(x_map["is_in_ring"].index(atom.IsInRing()))
        sp_hybridization.append(x_map["hybridization"].index(atom.GetHybridization()))

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    is_aromatic = torch.Tensor(is_aromatic).long()
    is_in_ring = torch.Tensor(is_in_ring).long()
    hybridization = torch.Tensor(sp_hybridization).long()
    if add_ad:
        # hydrogen bond acceptor and donor
        feats = factory.GetFeaturesForMol(mol)
        donor_ids = []
        acceptor_ids = []
        for f in feats:
            if f.GetFamily().lower() == "donor":
                donor_ids.append(f.GetAtomIds())
            elif f.GetFamily().lower() == "acceptor":
                acceptor_ids.append(f.GetAtomIds())

        if len(donor_ids) > 0:
            donor_ids = np.concatenate(donor_ids)
        else:
            donor_ids = np.array([])

        if len(acceptor_ids) > 0:
            acceptor_ids = np.concatenate(acceptor_ids)
        else:
            acceptor_ids = np.array([])
        is_acceptor = np.zeros(mol.GetNumAtoms(), dtype=np.uint8)
        is_donor = np.zeros(mol.GetNumAtoms(), dtype=np.uint8)
        if len(donor_ids) > 0:
            is_donor[donor_ids] = 1
        if len(acceptor_ids) > 0:
            is_acceptor[acceptor_ids] = 1

        is_donor = torch.from_numpy(is_donor).long()
        is_acceptor = torch.from_numpy(is_acceptor).long()
    else:
        is_donor = is_acceptor = None

    additional = {}
    if "wbo" in kwargs:
        wbo = torch.Tensor(kwargs["wbo"])[edge_index[0], edge_index[1]].float()
        additional["wbo"] = wbo
    if "mulliken" in kwargs:
        mulliken = torch.Tensor(kwargs["mulliken"]).float()
        additional["mulliken"] = mulliken
    if "grad" in kwargs:
        grad = torch.Tensor(kwargs["grad"]).float()
        additional["grad"] = grad

    if add_pocket:
        pocket = {
            "pos_pocket": m.positions_pocket,
            "x_pocket": m.atom_types_pocket,
            "pocket_ca_mask": [],
        }
        additional.update(pocket)

    data = Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        charges=all_charges,
        smiles=smiles,
        is_aromatic=is_aromatic,
        is_in_ring=is_in_ring,
        is_h_donor=is_donor,
        is_h_acceptor=is_acceptor,
        hybridization=hybridization,
        mol=mol,
        **additional,
    )

    return data

class LigandPocketDataset(InMemoryDataset):
    def __init__(
        self,
        split,
        root,
        with_docking_scores=False,
        remove_hs=True,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.remove_hs = remove_hs
        self.with_docking_scores = with_docking_scores

        self.compute_bond_distance_angles = True
        self.atom_encoder = full_atom_encoder

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])),
            valencies=load_pickle(self.processed_paths[5]),
            bond_lengths=load_pickle(self.processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[7])),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[8])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[10])).float(),
            is_h_donor=torch.from_numpy(np.load(self.processed_paths[11])).float(),
            is_h_acceptor=torch.from_numpy(np.load(self.processed_paths[12])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[13])

    @property
    def raw_file_names(self):
        d = "_dock" if self.with_docking_scores else ""
        if self.split == "train":
            return [f"train{d}.npz"]
        elif self.split == "val":
            return [f"val{d}.npz"]
        else:
            return [f"test{d}.npz"]

    def processed_file_names(self):
        h = "noh" if self.remove_hs else "h"
        d = "_dock" if self.with_docking_scores else ""
        if self.split == "train":
            return [
                f"train_{h}{d}.pt",
                f"train_n_{h}{d}.pickle",
                f"train_atom_types_{h}{d}.npy",
                f"train_bond_types_{h}{d}.npy",
                f"train_charges_{h}{d}.npy",
                f"train_valency_{h}{d}.pickle",
                f"train_bond_lengths_{h}{d}.pickle",
                f"train_angles_{h}{d}.npy",
                f"train_is_aromatic_{h}{d}.npy",
                f"train_is_in_ring_{h}{d}.npy",
                f"train_hybridization_{h}{d}.npy",
                f"train_is_h_donor_{h}{d}.npy",
                f"train_is_h_acceptor_{h}{d}.npy",
                f"train_smiles{d}.pickle",
            ]
        elif self.split == "val":
            return [
                f"val_{h}{d}.pt",
                f"val_n_{h}{d}.pickle",
                f"val_atom_types_{h}{d}.npy",
                f"val_bond_types_{h}{d}.npy",
                f"val_charges_{h}{d}.npy",
                f"val_valency_{h}{d}.pickle",
                f"val_bond_lengths_{h}{d}.pickle",
                f"val_angles_{h}{d}.npy",
                f"val_is_aromatic_{h}{d}.npy",
                f"val_is_in_ring_{h}{d}.npy",
                f"val_hybridization_{h}{d}.npy",
                f"val_is_h_donor_{h}{d}.npy",
                f"val_is_h_acceptor_{h}{d}.npy",
                f"val_smiles{d}.pickle",
            ]
        else:
            return [
                f"test_{h}{d}.pt",
                f"test_n_{h}{d}.pickle",
                f"test_atom_types_{h}{d}.npy",
                f"test_bond_types_{h}{d}.npy",
                f"test_charges_{h}{d}.npy",
                f"test_valency_{h}{d}.pickle",
                f"test_bond_lengths_{h}{d}.pickle",
                f"test_angles_{h}{d}.npy",
                f"test_is_aromatic_{h}{d}.npy",
                f"test_is_in_ring_{h}{d}.npy",
                f"test_hybridization_{h}{d}.npy",
                f"test_is_h_donor_{h}{d}.npy",
                f"test_is_h_acceptor_{h}{d}.npy",
                f"test_smiles{d}.pickle",
            ]

    def download(self):
        raise ValueError(
            "Download and preprocessing is manual. If the data is already downloaded, "
            f"check that the paths are correct. Root dir = {self.root} -- raw files {self.raw_paths}"
        )

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        data_list_lig = []
        all_smiles = []

        with np.load(self.raw_paths[0], allow_pickle=True) as f:
            data = {key: val for key, val in f.items()}

        # split data based on mask
        mol_data = {}
        docking_scores = []
        kiba_scores = []
        ic50s = []
        for k, v in data.items():
            if k == "names" or k == "receptors" or k == "lig_mol" or k == "lig_smiles":
                mol_data[k] = v
                continue

            if k == "docking_scores":
                docking_scores = torch.from_numpy(v)
                continue

            if k == "ic50s":
                ic50s = torch.tensor([float(i) for i in v]).float()
                continue

            if k == "kiba_scores":
                kiba_scores = torch.tensor([float(i) for i in v]).float()
                continue

            sections = (
                np.where(np.diff(data["lig_mask"]))[0] + 1
                if "lig" in k
                else np.where(np.diff(data["pocket_mask"]))[0] + 1
            )
            if k == "lig_atom" or k == "pocket_atom":
                mol_data[k] = [
                    torch.tensor([full_atom_encoder[a] for a in atoms])
                    for atoms in np.split(v, sections)
                ]
            else:
                if k == "pocket_one_hot":
                    pocket_one_hot_mask = data["pocket_mask"][data["pocket_ca_mask"]]
                    sections = np.where(np.diff(pocket_one_hot_mask))[0] + 1
                    mol_data["pocket_one_hot_mask"] = [
                        torch.from_numpy(x)
                        for x in np.split(pocket_one_hot_mask, sections)
                    ]
                else:
                    mol_data[k] = [torch.from_numpy(x) for x in np.split(v, sections)]
            # add number of nodes for convenience
            if k == "lig_mask":
                mol_data["num_lig_atoms"] = torch.tensor(
                    [len(x) for x in mol_data["lig_mask"]]
                )
            elif k == "pocket_mask":
                mol_data["num_pocket_nodes"] = torch.tensor(
                    [len(x) for x in mol_data["pocket_mask"]]
                )
            elif k == "pocket_one_hot":
                mol_data["num_resids_nodes"] = torch.tensor(
                    [len(x) for x in mol_data["pocket_one_hot_mask"]]
                )

        for i, (
            mol_lig,
            coords_lig,
            atoms_lig,
            mask_lig,
            coords_pocket,
            atoms_pocket,
            mask_pocket,
            pocket_ca_mask,
            name,
            docking_score,
            kiba_score,
            ic50,
        ) in enumerate(
            tqdm(
                zip_longest(
                    mol_data["lig_mol"],
                    mol_data["lig_coords"],
                    mol_data["lig_atom"],
                    mol_data["lig_mask"],
                    mol_data["pocket_coords"],
                    mol_data["pocket_atom"],
                    mol_data["pocket_mask"],
                    mol_data["pocket_ca_mask"],
                    mol_data["names"],
                    docking_scores,
                    kiba_scores,
                    ic50s,
                    fillvalue=None,
                ),
                total=len(mol_data["lig_mol"]),
            )
        ):
            try:
                # atom_types = [atom_decoder[int(a)] for a in atoms_lig]
                # smiles_lig, conformer_lig = get_mol_babel(coords_lig, atom_types)
                smiles_lig = Chem.MolToSmiles(mol_lig)
                data = mol_to_torch_geometric(
                    mol_lig,
                    full_atom_encoder,
                    smiles_lig,
                    remove_hydrogens=self.remove_hs,
                    cog_proj=False,
                )
            except Exception:
                print(f"Ligand {i} failed")
                continue
            data.pos_lig = coords_lig
            data.x_lig = atoms_lig
            data.pos_pocket = coords_pocket
            data.x_pocket = atoms_pocket
            data.lig_mask = mask_lig
            data.pocket_mask = mask_pocket
            data.pocket_ca_mask = pocket_ca_mask
            data.pocket_name = name
            data.docking_scores = docking_score
            data.kiba_score = kiba_score
            data.ic50 = ic50
            all_smiles.append(smiles_lig)
            data_list_lig.append(data)

        center = False
        if center:
            for i in range(len(data_list_lig)):
                mean = (
                    data_list_lig[i].pos.sum(0) + data_list_lig[i].pos_pocket.sum(0)
                ) / (len(data_list_lig[i].pos) + len(data_list_lig[i].pos_pocket))
                data_list_lig[i].pos = data_list_lig[i].pos - mean
                data_list_lig[i].pos_pocket = data_list_lig[i].pos_pocket - mean

        torch.save(self.collate(data_list_lig), self.processed_paths[0])
        print("Finished processing.\nCalculating statistics")

        statistics = compute_all_statistics(
            data_list_lig,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
            additional_feats=True,
            # do not compute bond distance and bond angle statistics due to time and we do not use it anyways currently
        )
        save_pickle(statistics.num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        np.save(self.processed_paths[8], statistics.is_aromatic)
        np.save(self.processed_paths[9], statistics.is_in_ring)
        np.save(self.processed_paths[10], statistics.hybridization)
        np.save(self.processed_paths[11], statistics.is_h_donor)
        np.save(self.processed_paths[12], statistics.is_h_acceptor)

        save_pickle(set(all_smiles), self.processed_paths[13])

    def __getitem__(
        self,
        idx: Union[int, np.integer, IndexType],
    ):
        if (
            isinstance(idx, (int, np.integer))
            or (isinstance(idx, Tensor) and idx.dim() == 0)
            or (isinstance(idx, np.ndarray) and np.isscalar(idx))
        ):
            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            if hasattr(data, "docking_scores"):
                data.docking_scores = data.docking_scores.clamp(max=100.)
            return data

        else:
            return self.index_select(idx)


class LigandPocketDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.datadir = cfg.dataset_root
        self.pin_memory = True
        self.test_batch_size = 1
        self.cfg = cfg
    
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = LigandPocketDataset(
                split="train", root=self.datadir, remove_hs=self.cfg.remove_hs, with_docking_scores=True
            )
            self.val_dataset = LigandPocketDataset(
                split="val", root=self.datadir, remove_hs=self.cfg.remove_hs, with_docking_scores=True
            )
            self.test_dataset = LigandPocketDataset(
                split="test", root=self.datadir, remove_hs=self.cfg.remove_hs, with_docking_scores=True
            )
            
    def get_dataloader(self, dataset, stage):
        if stage == "train":
            batch_size = self.cfg.batch_size
            shuffle = True
        elif stage in ["val", "test"]:
            batch_size = self.cfg.batch_size
            shuffle = False

        dl = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            follow_batch=["pos", "pos_pocket"],
        )

        return dl
    
    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, "train")
    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, "val")
    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset, "test")
       
if __name__ == "__main__":
    from collections import namedtuple
    
    d = {"dataset_root": "/scratch1/e3moldiffusion/data/crossdocked/crossdocked_noH_cutoff5_dock_new/",
         "joint_property_prediction": True,
         "dataset": "crossdocked",
         "regression_property": "docking_score",
         "property_training": False,
         "remove_hs": True,
         "batch_size": 128,
         "num_workers": 4,
         "inference_batch_size": 1
         }
    
    config = namedtuple('x', d.keys())(*d.values())
    print(config)
    datamodule = LigandPocketDataModule(config)
    loader = datamodule.val_dataloader()
    data = next(iter(loader))
    print(data)
    