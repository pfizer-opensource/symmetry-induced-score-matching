from typing import Tuple, Dict

import torch

from torch import Tensor

from torch_geometric.utils import remove_self_loops, sort_edge_index, to_dense_batch
from torch_scatter import scatter_mean
from torch_sparse import coalesce
    
def get_so3_cartesian_axes(theta, axis: str):
    assert theta.ndim == 1
    n = len(theta)
    R = torch.zeros((n, 3, 3), 
                    device=theta.device
                   )
    assert axis in ["x", "y", "z"]
    if axis == "x":
        R[:, 0, 0] = 1.0
        R[:, 1, 1] = torch.cos(theta)
        R[:, 2, 2] = torch.cos(theta)
        R[:, 1, 2] = -torch.sin(theta)
        R[:, 2, 1] = torch.sin(theta)
    elif axis == "y":
        R[:, 1, 1] = 1.0
        R[:, 0, 0] = torch.cos(theta)
        R[:, 2, 2] = torch.cos(theta)
        R[:, 2, 0] = -torch.sin(theta)
        R[:, 0, 2] = torch.sin(theta)
    else:
        R[:, 2, 2] = 1.0
        R[:, 0, 0] = torch.cos(theta)
        R[:, 1, 1] = torch.cos(theta)
        R[:, 1, 0] = torch.sin(theta)
        R[:, 0, 1] = -torch.sin(theta)
    return R

def get_mu_nu_mask(batch: Tensor) -> Dict[str, Tensor]:
    
    batch_num_nodes = batch.bincount()
    ptr0 = torch.concat((torch.zeros(1,
                                     device=batch.device, 
                                     dtype=torch.long),
                         batch_num_nodes.cumsum(0)[:-1]))
    ptr1 = ptr0 + 1
    ptr1_mu = torch.concat([ptr0, ptr1]).sort()[0]
    mu_mask = torch.ones(len(batch), dtype=torch.float32, device=batch.device)
    nu_mask = torch.ones_like(mu_mask)
    mu_mask[ptr0] = 0.0
    nu_mask[ptr1_mu] = 0.0
    
    return {"mu": mu_mask, "nu": nu_mask, "ptr0": ptr0, "ptr1": ptr1}


def compute_PCs(xs: Tensor) -> Tuple[Tensor, Tensor]:

    # we make sure we are in the center of mass
    mean = torch.mean(xs, dim=0)
    centered_xs = xs - mean

    covariance_matrix = torch.matmul(centered_xs.T, centered_xs)

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    sorted_indices = torch.argsort(eigenvalues, descending=True)
    pc1 = eigenvectors[:, sorted_indices[0]]
    pc2 = eigenvectors[:, sorted_indices[1]]

    return pc1, pc2


def compute_PCs_batch(xs: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
    com = scatter_mean(xs, batch, dim=0)
    xs = xs - com[batch]

    xs, _ = to_dense_batch(xs, batch)  # (B, N, 3)

    covariance_matrix = torch.einsum('bij,bik->bjk', xs, xs)  # (B, 3, 3)

    _, eigenvectors = torch.linalg.eigh(covariance_matrix)  # (B, 3), (B, 3, 3)

    # torch.linalg.eigh returns eigvalues in ascending order
    pc1 = eigenvectors[:, :, -1]
    pc2 = eigenvectors[:, :, -2]

    return pc1, pc2


def coalesce_edges(
    edge_index: Tensor, bond_edge_index: Tensor, bond_edge_attr: Tensor, n: int
) -> Tuple[Tensor, Tensor]:

    edge_attr = torch.full(
        size=(edge_index.size(-1),),
        fill_value=0,
        device=edge_index.device,
        dtype=torch.long,
    )
    edge_index = torch.cat([edge_index, bond_edge_index], dim=-1)
    edge_attr = torch.cat([edge_attr, bond_edge_attr], dim=0)
    edge_index, edge_attr = coalesce(
        index=edge_index, value=edge_attr, m=n, n=n, op="max"
    )
    return edge_index, edge_attr


def concat_ligand_pocket(
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    x_ligand: Tensor,
    x_pocket: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    batch_ctx = torch.cat([batch_ligand, batch_pocket], dim=0)

    mask_ligand = torch.cat(
        [
            torch.ones([batch_ligand.size(0)], device=batch_ligand.device).bool(),
            torch.zeros([batch_pocket.size(0)], device=batch_pocket.device).bool(),
        ],
        dim=0,
    )
    pos_ctx = torch.cat([pos_ligand, pos_pocket], dim=0)
    x_ctx = torch.cat([x_ligand, x_pocket], dim=0)

    return pos_ctx, x_ctx, batch_ctx, mask_ligand


def get_ligand_pocket_edges(
    batch_lig: Tensor,
    batch_pocket: Tensor,
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    cutoff_p: float,
    cutoff_lp: float,
) -> Tensor:

    # ligand-ligand is fully-connected
    adj_ligand = batch_lig[:, None] == batch_lig[None, :]
    adj_pocket = batch_pocket[:, None] == batch_pocket[None, :]
    adj_cross = batch_lig[:, None] == batch_pocket[None, :]

    with torch.no_grad():
        D_pocket = torch.cdist(pos_pocket, pos_pocket)
        D_cross = torch.cdist(pos_ligand, pos_pocket)

    # pocket-pocket is not fully-connected
    # but selected based on distance cutoff
    adj_pocket = adj_pocket & (D_pocket <= cutoff_p)
    # ligand-pocket is not fully-connected
    # but selected based on distance cutoff
    adj_cross = adj_cross & (D_cross <= cutoff_lp)

    adj = torch.cat(
        (
            torch.cat((adj_ligand, adj_cross), dim=1),
            torch.cat((adj_cross.T, adj_pocket), dim=1),
        ),
        dim=0,
    )
    edges = torch.stack(torch.where(adj), dim=0)  # COO format (2, n_edges)
    return edges


def get_joint_edge_attrs(
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
    edge_attr_ligand: Tensor,
    num_bond_classes: int,
    cutoff_p: float = 5.0,
    cutoff_lp: float = 5.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    assert num_bond_classes == 5
    device = edge_attr_ligand.device
    edge_index_global = get_ligand_pocket_edges(
        batch_ligand,
        batch_pocket,
        pos_ligand,
        pos_pocket,
        cutoff_p=cutoff_p,
        cutoff_lp=cutoff_lp,
    )
    edge_index_global = sort_edge_index(edge_index=edge_index_global, sort_by_row=False)
    edge_index_global, _ = remove_self_loops(edge_index_global)
    edge_attr_global = torch.zeros(
        (edge_index_global.size(1), num_bond_classes),
        dtype=torch.float32,
        device=device,
    )

    edge_mask_ligand = (edge_index_global[0] < len(batch_ligand)) & (
        edge_index_global[1] < len(batch_ligand)
    )
    edge_mask_pocket = (edge_index_global[0] >= len(batch_ligand)) & (
        edge_index_global[1] >= len(batch_ligand)
    )
    edge_attr_global[edge_mask_ligand] = edge_attr_ligand

    # placeholder no-bond information
    edge_attr_global[edge_mask_pocket] = (
        torch.tensor([0, 0, 0, 0, 1]).float().to(edge_attr_global.device)
    )

    batch_full = torch.cat([batch_ligand, batch_pocket])
    batch_edge_global = batch_full[edge_index_global[0]]  #
    
    edge_mask_ligand_pocket = (edge_index_global[0] < len(batch_ligand)) & (
        edge_index_global[1] >= len(batch_ligand)
    )
    edge_mask_pocket_ligand = (edge_index_global[0] >= len(batch_ligand)) & (
        edge_index_global[1] < len(batch_ligand)
    )

    # feature for interaction,
    # ligand-ligand, pocket-pocket, ligand-pocket, pocket-ligand
    edge_initial_interaction = torch.zeros(
        (edge_index_global.size(1), 3),
        dtype=torch.float32,
        device=device,
    )

    edge_initial_interaction[edge_mask_ligand] = (
        torch.tensor([1, 0, 0]).float().to(edge_attr_global.device)
    )  # ligand-ligand

    edge_initial_interaction[edge_mask_pocket] = (
        torch.tensor([0, 1, 0]).float().to(edge_attr_global.device)
    )  # pocket-pocket

    edge_initial_interaction[edge_mask_ligand_pocket] = (
        torch.tensor([0, 0, 1]).float().to(edge_attr_global.device)
    )  # ligand-pocket

    edge_initial_interaction[edge_mask_pocket_ligand] = (
        torch.tensor([0, 0, 1]).float().to(edge_attr_global.device)
    )  # pocket-ligand

    return (
        edge_index_global,
        edge_attr_global,
        batch_edge_global,
        edge_mask_ligand,
        edge_mask_pocket,
        edge_initial_interaction,
    )


def combine_protein_ligand_feats(
    pos_ligand: Tensor,
    pos_pocket: Tensor,
    atom_types_ligand: Tensor,
    atom_types_pocket: Tensor,
    batch_ligand: Tensor,
    batch_pocket: Tensor,
    edge_attr_ligand: Tensor,
    num_bond_classes: int,
    cutoff_p: float = 5.0,
    cutoff_lp: float = 5.0,
):
    """Wraps the utils.concat_ligand_pocket and utils.get_joint_edge_attrs
    into one function call
    """

    # get joint node-level features stacked as
    # [ligand, pocket] along the 0-th dimension
    (
        pos_joint,
        atom_types_joint,
        batch_full,
        mask_ligand,
    ) = concat_ligand_pocket(
        pos_ligand,
        pos_pocket,
        atom_types_ligand,
        atom_types_pocket,
        batch_ligand,
        batch_pocket,
    )

    # create protein-ligand complex edge-attrs
    (
        edge_index_global,
        edge_attr_global,
        batch_edge_global,
        edge_mask_ligand,
        _,
        edge_initial_interaction,
    ) = get_joint_edge_attrs(
        pos_ligand,
        pos_pocket,
        batch_ligand,
        batch_pocket,
        edge_attr_ligand,
        num_bond_classes,
        cutoff_p=cutoff_p,
        cutoff_lp=cutoff_lp,
    )

    out = (
        pos_joint,
        atom_types_joint,
        batch_full,
        mask_ligand,
        edge_index_global,
        edge_attr_global,
        edge_mask_ligand,
        edge_initial_interaction,
        batch_edge_global,
    )

    return out
