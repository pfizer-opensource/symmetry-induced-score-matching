import os
import os.path as osp
from typing import Any, Sequence
import numpy as np
import pandas as pd
import torch

from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from tqdm import tqdm
from torch_geometric.loader import DataLoader

import lightning as L

def mol_to_torch_geometric(
    mol,
    atom_encoder,
    smiles=None,
    remove_hydrogens: bool = False,
    cog_proj: bool = True,
):
    if remove_hydrogens:
        mol = Chem.RemoveHs(
            mol
        )
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

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
      
    atom_types = torch.Tensor(atom_types).long()

    data = Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        smiles=smiles,
        mol=mol,
    )

    return data


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


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


class QM9Dataset(InMemoryDataset):
    raw_url = (
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
        "molnet_publish/qm9.zip"
    )
    raw_url2 = "https://ndownloader.figshare.com/files/3195404"
    processed_url = "https://data.pyg.org/datasets/qm9_v3.zip"

    def __init__(
        self,
        split,
        root,
        remove_h: bool,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.split = split
        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h

        self.atom_encoder = full_atom_encoder
        if remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["gdb9.sdf", "gdb9.sdf.csv", "uncharacterized.txt"]

    @property
    def split_file_name(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        if self.split == "train":
            return [
                f"train_{h}.pt",
            ]
        elif self.split == "val":
            return [
                f"val_{h}.pt",
            ]
        else:
            return [
                f"test_{h}.pt",
            ]

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa

            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(
                osp.join(self.raw_dir, "3195404"),
                osp.join(self.raw_dir, "uncharacterized.txt"),
            )
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(
            dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train]
        )

        train.to_csv(os.path.join(self.raw_dir, "train.csv"))
        val.to_csv(os.path.join(self.raw_dir, "val.csv"))
        test.to_csv(os.path.join(self.raw_dir, "test.csv"))

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=["mol_id"], inplace=True)

        with open(self.raw_paths[-1]) as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split("\n")[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)
        data_list = []
        all_smiles = []
        num_errors = 0
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue
            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            if smiles is None:
                num_errors += 1
            else:
                all_smiles.append(smiles)

            data = mol_to_torch_geometric(mol, full_atom_encoder, smiles, cog_proj=True)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class QM9DataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.datadir = cfg.dataset_root
        self.cfg = cfg
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = QM9Dataset(
                split="train", root=self.datadir, remove_h=self.cfg.remove_hs
            )
            self.val_dataset = QM9Dataset(split="val", root=self.datadir, remove_h=self.cfg.remove_hs)
            self.test_dataset = QM9Dataset(split="test", root=self.datadir, remove_h=self.cfg.remove_hs)
            
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
        )

        return dl
    
    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, "train")
    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, "val")
    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset, "test")

if __name__ == "__main__":
    
    
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    hparams = {'dataset_root': '/Users/tuanle/Desktop/projects/symmetry-induced-score-matching/data/qm9',
               'remove_hs': False,
               'batch_size': 32,
               'num_workers': 0
               }
    hparams = dotdict(hparams)
    datamodule = QM9DataModule(cfg=hparams)
    datamodule.setup(stage="fit")
    
    print(datamodule.train_dataset[0])