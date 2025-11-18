import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from torch_scatter import scatter_add, scatter_mean

from sism.modules import GatedEquivBlock, DenseLayer, LayerNorm
from sism.plcomplex.convs import EQGATGlobalEdgeConv


class EQGATEdgeGNN(nn.Module):
    """Equivariant Graph Attention Network
    Reference:
        @inproceedings{
            le2024navigating,
            title={Navigating the Design Space of Equivariant Diffusion-Based Generative Models for De Novo 3D Molecule Generation},
            author={Tuan Le and Julian Cremer and Frank Noe and Djork-Arn{\'e} Clevert and Kristof T Sch{\"u}tt},
            booktitle={The Twelfth International Conference on Learning Representations},
            year={2024},
            url={https://openreview.net/forum?id=kzGuiRXZrQ}
        }
    Args:
        nn (Module): PyTorch neural network module.
    """

    def __init__(
        self,
        hn_dim: Tuple[int, int] = (256, 128),
        cutoff: float = 5.0,
        num_layers: int = 5,
        num_rbfs: int = 20,
        edge_dim: Optional[int] = None,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        update_coords: bool = False,
        update_edges: bool = False,
    ):
        super(EQGATEdgeGNN, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.update_coords = update_coords
        self.update_edges = update_edges

        convs = []

        for i in range(num_layers):
            convs.append(
                EQGATGlobalEdgeConv(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    edge_dim=edge_dim,
                    cutoff=cutoff,
                    num_rbfs=num_rbfs,
                    has_v_in=i > 0,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                    use_cross_product=use_cross_product,
                    update_coords=update_coords,
                    update_edges=update_edges,
                )
            )

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([LayerNorm(dims=hn_dim) for _ in range(num_layers)])
        self.out_norm = LayerNorm(hn_dim)

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        pos: Tensor,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos = pos / torch.norm(pos, dim=1).unsqueeze(1)
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1).sqrt(), min=1e-6)
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        s: Tensor,
        v: Tensor,
        p: Tensor,
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Optional[Tensor]],
        edge_attr_initial_ohe: Tensor,
        edge_attr_global_embedding: Tensor,
        batch: Tensor = None,
        ligand_mask: Tensor = None,
        batch_ligand: Tensor = None,
    ) -> Tuple[Tensor]:

        assert edge_attr_initial_ohe.size(1) == 3
        for i in range(len(self.convs)):
            s, v = self.norms[i](x={"s": s, "v": v}, batch=batch)
            out = self.convs[i](
                x=(s, v, p),
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_attr_initial_ohe=edge_attr_initial_ohe,
                edge_attr_global_embedding=edge_attr_global_embedding,
                ligand_mask=ligand_mask,
            )
            s, v, p, e = out
            if self.update_coords:
                edge_attr = self.calculate_edge_attrs(edge_index, edge_attr[-1], p)
            if self.update_edges:
                edge_attr = (edge_attr[0], edge_attr[1], edge_attr[2], e)
        s, v = self.out_norm(x={"s": s, "v": v}, batch=batch)
        return s, v, p, edge_attr[-1]


class DiffusionScoreModelSphere(nn.Module):
    """Diffusion Score Model on using generalized score matching from the main paper
    The final (global) scores are non-equivariant since scalars are mixed at the end.
    The equivariance is already handled in the induced Lie algebra dynamics.
    """

    def __init__(
        self,
        atom_feat_dim: int,
        edge_feat_dim: int,
        hn_dim: Tuple[int, int] = (128, 32),
        edim: int = 16,
        cutoff: float = 5.0,
        num_layers: int = 5,
        num_rbfs: int = 20,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        global_translations: bool = False,
        update_coords: bool = False,
        update_edges: bool = False,
    ):
        super(DiffusionScoreModelSphere, self).__init__()

        self.update_coords = update_coords
        self.update_edges = update_edges
        self.hn_dim = hn_dim
        self.edim = edim
        self.atom_mapping = DenseLayer(atom_feat_dim, hn_dim[0], bias=True)
        self.edge_mapping = DenseLayer(edge_feat_dim, edim, bias=True)
        self.time_mapping = DenseLayer(1, hn_dim[0], bias=True)
        self.time_mapping_edge = DenseLayer(1, edim, bias=True)
        self.edge_pre = nn.Sequential(
            DenseLayer(3 * num_rbfs, 2 * num_rbfs, activation=nn.Softplus()),
            DenseLayer(2 * num_rbfs, 1, activation=nn.Sigmoid()),
        )
        self.global_translations = global_translations
        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff=cutoff,
            num_layers=num_layers,
            num_rbfs=num_rbfs,
            edge_dim=edim,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            update_coords=update_coords,
            update_edges=update_edges,
        )

        coords_dim = 1 if update_coords else 0
        self.gated = GatedEquivBlock(
            in_dims=(hn_dim[0] + (edim * int(update_edges)), hn_dim[1] + coords_dim),
            out_dims=(hn_dim[0], hn_dim[1]),
            use_mlp=False,
        )
        self.out_score = nn.Sequential(
            DenseLayer(hn_dim[0] + 3 * hn_dim[1], hn_dim[0], activation=nn.SiLU()),
            DenseLayer(hn_dim[0], hn_dim[0], activation=nn.SiLU()),
            DenseLayer(hn_dim[0], 3),
        )
        if global_translations:
            self.out_score_com = nn.Sequential(
                DenseLayer(hn_dim[0] + 3 * hn_dim[1], hn_dim[0], activation=nn.SiLU()),
                DenseLayer(hn_dim[0], hn_dim[0], activation=nn.SiLU()),
                DenseLayer(hn_dim[0], 3),
            )
        else:
            self.out_score_com = None

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        pos: Tensor,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos = pos / torch.norm(pos, dim=1).unsqueeze(1)
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1).sqrt(), min=1e-6)
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_attr_initial_ohe: Tensor,
        batch: Tensor,
        batch_edge: Tensor,
        batch_ligand: Tensor = None,
        mask_ligand: Tensor = None,
    ) -> Dict:

        if t.ndim == 1:
            t = t.unsqueeze(-1)
        s = self.atom_mapping(x) + self.time_mapping(t)[batch]
        e = self.edge_mapping(edge_attr) + self.time_mapping_edge(t)[batch_edge]
        edge_attr = self.calculate_edge_attrs(edge_index, e, pos)

        assert edge_attr_initial_ohe is not None
        assert edge_attr_initial_ohe.size(1) == 3
        d = edge_attr[0]
        rbf = self.gnn.convs[0].radial_basis_func(d)
        rbf_ohe = torch.einsum("nk, nd -> nkd", (rbf, edge_attr_initial_ohe))
        rbf_ohe = rbf_ohe.view(d.size(0), -1)
        edge_attr_global_embedding = self.edge_pre(rbf_ohe)

        v = torch.zeros(
            (x.size(0), 3, self.hn_dim[1]), device=x.device, dtype=pos.dtype
        )

        s, v, p, e = self.gnn(
            s,
            v,
            pos,
            edge_index,
            edge_attr,
            edge_attr_initial_ohe,
            edge_attr_global_embedding,
            batch,
            mask_ligand,
        )
        if self.update_coords:
            v = torch.concat([v, p.unsqueeze(-1)], dim=-1)
        if self.update_edges:
            s = torch.concat((s, scatter_mean(e, edge_index[0], dim=0)), dim=-1)

        s, v = self.gated((s, v))
        v = v.reshape(v.size(0), -1)
        input = torch.cat([s, v], dim=-1)

        scores = self.out_score(input)
        scores = scores * mask_ligand.unsqueeze(-1)
        scores = scatter_add(scores, batch, dim=0)
        out_dict = {
            "score": scores,
        }

        if self.global_translations:
            scores_com = self.out_score_com(input)
            scores_com = scores_com * mask_ligand.unsqueeze(-1)
            scores_com = scatter_add(scores_com, batch, dim=0)
            out_dict.update({"score_com": scores_com})

        return out_dict


class DiffusionScoreModelRiemannian(nn.Module):
    """Diffusion Score Model on Riemannian Manifolds.
    Predicts a global translation and rotation score that are equivariant.
    """

    def __init__(
        self,
        atom_feat_dim: int,
        edge_feat_dim: int,
        hn_dim: Tuple[int, int] = (128, 32),
        edim: int = 16,
        cutoff: float = 5.0,
        num_layers: int = 5,
        num_rbfs: int = 20,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        global_translations: bool = False,
        update_coords: bool = False,
        update_edges: bool = False,
    ):
        super(DiffusionScoreModelRiemannian, self).__init__()

        self.update_coords = update_coords
        self.update_edges = update_edges
        self.hn_dim = hn_dim
        self.edim = edim
        self.atom_mapping = DenseLayer(atom_feat_dim, hn_dim[0], bias=True)
        self.edge_mapping = DenseLayer(edge_feat_dim, edim, bias=True)
        self.time_mapping = DenseLayer(1, hn_dim[0], bias=True)
        self.time_mapping_edge = DenseLayer(1, edim, bias=True)
        self.edge_pre = nn.Sequential(
            DenseLayer(3 * num_rbfs, 2 * num_rbfs, activation=nn.Softplus()),
            DenseLayer(2 * num_rbfs, 1, activation=nn.Sigmoid()),
        )
        self.global_translations = global_translations
        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff=cutoff,
            num_layers=num_layers,
            num_rbfs=num_rbfs,
            edge_dim=edim,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            update_coords=update_coords,
            update_edges=update_edges,
        )

        coords_dim = 1 if update_coords else 0
        self.gated = GatedEquivBlock(
            in_dims=(hn_dim[0] + (edim * int(update_edges)), hn_dim[1] + coords_dim),
            out_dims=(hn_dim[0], hn_dim[1]),
            use_mlp=False,
        )
        self.out_score = DenseLayer(hn_dim[1], 2)

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        pos: Tensor,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos = pos / torch.norm(pos, dim=1).unsqueeze(1)
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1).sqrt(), min=1e-6)
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_attr_initial_ohe: Tensor,
        batch: Tensor,
        batch_edge: Tensor,
        batch_ligand: Tensor = None,
        mask_ligand: Tensor = None,
    ) -> Dict:

        if t.ndim == 1:
            t = t.unsqueeze(-1)
        s = self.atom_mapping(x) + self.time_mapping(t)[batch]
        e = self.edge_mapping(edge_attr) + self.time_mapping_edge(t)[batch_edge]
        edge_attr = self.calculate_edge_attrs(edge_index, e, pos)

        assert edge_attr_initial_ohe is not None
        assert edge_attr_initial_ohe.size(1) == 3
        d = edge_attr[0]
        rbf = self.gnn.convs[0].radial_basis_func(d)
        rbf_ohe = torch.einsum("nk, nd -> nkd", (rbf, edge_attr_initial_ohe))
        rbf_ohe = rbf_ohe.view(d.size(0), -1)
        edge_attr_global_embedding = self.edge_pre(rbf_ohe)

        v = torch.zeros(
            (x.size(0), 3, self.hn_dim[1]), device=x.device, dtype=pos.dtype
        )

        s, v, p, e = self.gnn(
            s,
            v,
            pos,
            edge_index,
            edge_attr,
            edge_attr_initial_ohe,
            edge_attr_global_embedding,
            batch,
            mask_ligand,
        )
        if self.update_coords:
            v = torch.concat([v, p.unsqueeze(-1)], dim=-1)
        if self.update_edges:
            s = torch.concat((s, scatter_mean(e, edge_index[0], dim=0)), dim=-1)

        s, v = self.gated((s, v))

        scores = self.out_score(v) + (s * 0.0).sum()
        scores = scores * mask_ligand.unsqueeze(-1).unsqueeze(-1)
        scores = scatter_add(scores, batch, dim=0)

        translation_score, rotation_score = torch.chunk(scores, 2, dim=-1)
        out_dict = {
            "translation_score": translation_score,
            "rotation_score": rotation_score,
        }
        return out_dict


class DiffusionScoreModelFisher(nn.Module):
    """Diffusion Score Model on Euclidean Space.
    Predicts (local) translation scores for each atom in the point cloud.
    Scores are equivariant to rotations and translations.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        atom_feat_dim: int,
        edge_feat_dim: int,
        hn_dim: Tuple[int, int] = (128, 32),
        edim: int = 16,
        cutoff: float = 5.0,
        num_layers: int = 5,
        num_rbfs: int = 20,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        global_translations: bool = False,
        update_coords: bool = False,
        update_edges: bool = False,
    ):
        super(DiffusionScoreModelFisher, self).__init__()

        self.update_coords = update_coords
        self.update_edges = update_edges
        self.hn_dim = hn_dim
        self.edim = edim
        self.atom_mapping = DenseLayer(atom_feat_dim, hn_dim[0], bias=True)
        self.edge_mapping = DenseLayer(edge_feat_dim, edim, bias=True)
        self.time_mapping = DenseLayer(1, hn_dim[0], bias=True)
        self.time_mapping_edge = DenseLayer(1, edim, bias=True)
        self.edge_pre = nn.Sequential(
            DenseLayer(3 * num_rbfs, 2 * num_rbfs, activation=nn.Softplus()),
            DenseLayer(2 * num_rbfs, 1, activation=nn.Sigmoid()),
        )
        self.global_translations = global_translations
        self.gnn = EQGATEdgeGNN(
            hn_dim=hn_dim,
            cutoff=cutoff,
            num_layers=num_layers,
            num_rbfs=num_rbfs,
            edge_dim=edim,
            use_cross_product=use_cross_product,
            vector_aggr=vector_aggr,
            update_coords=update_coords,
            update_edges=update_edges,
        )

        coords_dim = 1 if update_coords else 0
        self.gated = GatedEquivBlock(
            in_dims=(hn_dim[0] + (edim * int(update_edges)), hn_dim[1] + coords_dim),
            out_dims=(hn_dim[0], hn_dim[1]),
            use_mlp=False,
        )

        assert (
            not global_translations
        ), "Global translations are not supported in Fisher model."
        self.out_dim_score = 1 + int(global_translations)
        self.out_score = DenseLayer(hn_dim[1], self.out_dim_score)

    def calculate_edge_attrs(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        pos: Tensor,
    ):
        source, target = edge_index
        r = pos[target] - pos[source]
        pos = pos / torch.norm(pos, dim=1).unsqueeze(1)
        a = pos[target] * pos[source]
        a = a.sum(-1)
        d = torch.clamp(torch.pow(r, 2).sum(-1).sqrt(), min=1e-6)
        r_norm = torch.div(r, (1.0 + d.unsqueeze(-1)))
        edge_attr = (d, a, r_norm, edge_attr)
        return edge_attr

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        t: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        edge_attr_initial_ohe: Tensor,
        batch: Tensor,
        batch_edge: Tensor,
        batch_ligand: Tensor = None,
        mask_ligand: Tensor = None,
    ) -> Dict:

        if t.ndim == 1:
            t = t.unsqueeze(-1)
        s = self.atom_mapping(x) + self.time_mapping(t)[batch]
        e = self.edge_mapping(edge_attr) + self.time_mapping_edge(t)[batch_edge]
        edge_attr = self.calculate_edge_attrs(edge_index, e, pos)

        assert edge_attr_initial_ohe is not None
        assert edge_attr_initial_ohe.size(1) == 3
        d = edge_attr[0]
        rbf = self.gnn.convs[0].radial_basis_func(d)
        rbf_ohe = torch.einsum("nk, nd -> nkd", (rbf, edge_attr_initial_ohe))
        rbf_ohe = rbf_ohe.view(d.size(0), -1)
        edge_attr_global_embedding = self.edge_pre(rbf_ohe)

        v = torch.zeros(
            (x.size(0), 3, self.hn_dim[1]), device=x.device, dtype=pos.dtype
        )

        s, v, p, e = self.gnn(
            s,
            v,
            pos,
            edge_index,
            edge_attr,
            edge_attr_initial_ohe,
            edge_attr_global_embedding,
            batch,
            mask_ligand,
        )
        if self.update_coords:
            v = torch.concat([v, p.unsqueeze(-1)], dim=-1)
        if self.update_edges:
            s = torch.concat((s, scatter_mean(e, edge_index[0], dim=0)), dim=-1)

        s, v = self.gated((s, v))

        scores = self.out_score(v) + (s * 0.0).sum()
        scores = scores * mask_ligand.unsqueeze(-1).unsqueeze(-1)
        local_translation_score = scores[mask_ligand]
        if self.global_translations:
            global_translation_score = local_translation_score[:, :, 1]
            assert (
                batch_ligand is not None
            ), "Batch information is required for global translation scores."
            global_translation_score = scatter_add(
                global_translation_score, batch_ligand, dim=0
            )
            local_translation_score = local_translation_score[:, :, 0]
        else:
            global_translation_score = None
            local_translation_score = local_translation_score[:, :, 0]

        out_dict = {
            "local_translation_score": local_translation_score,
            "global_translation_score": global_translation_score,
        }
        return out_dict
