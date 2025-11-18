from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter, scatter_softmax

from sism.modules import DenseLayer, GatedEquivBlock, reset


def cross_product(a: Tensor, b: Tensor, dim: int) -> Tensor:
    if a.dtype != torch.float16 and b.dtype != torch.float16:
        return torch.linalg.cross(a, b, dim=dim)
    else:
        s1 = a[:, 1, :] * b[:, -1, :] - a[:, -1, :] * b[:, 1, :]
        s2 = a[:, -1, :] * b[:, 0, :] - a[:, 0, :] * b[:, -1, :]
        s3 = a[:, 0, :] * b[:, 1, :] - a[:, 1, :] * b[:, 0, :]
        cross = torch.stack([s1, s2, s3], dim=dim)
        return cross
    
    
class GaussianExpansion(torch.nn.Module):
    def __init__(self, max_value=5.0, K=20):
        super().__init__()
        offset = torch.linspace(0.0, max_value, K)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer("offset", offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class EQGATGlobalEdgeConv(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, Optional[int]],
        edge_dim: int,
        num_rbfs: int = 20,
        eps: float = 1e-6,
        has_v_in: bool = False,
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
        use_cross_product: bool = False,
        cutoff: float = 5.0,
        update_coords: bool = False,
        update_edges: bool = False,
    ):
        super().__init__(node_dim=0, aggr=None, flow="source_to_target")

        assert edge_dim is not None

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        self.use_cross_product = use_cross_product
        self.silu = nn.SiLU()
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3 if use_cross_product else 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()
            
        self.update_coords = update_coords
        if update_coords:
            coords_dim = 1
        else:
            coords_dim = 0
                
        self.update_edges = update_edges
        if update_edges:
            edge_dim_out = edge_dim
        else:
            edge_dim_out = 0
            
        input_edge_dim = 2 * self.si + edge_dim + 2 + 2
      
        self.cutoff = cutoff

        self.num_rbfs = num_rbfs
        self.radial_basis_func = GaussianExpansion(max_value=cutoff, K=num_rbfs)
        input_edge_dim += (3*num_rbfs)  # (ligand-ligand, ligand-pocket, pocket-pocket)
        input_edge_dim += 1  # global edge

        self.edge_net = nn.Sequential(
            DenseLayer(input_edge_dim, self.si, bias=True, activation=nn.SiLU()),
            DenseLayer(
                self.si, self.v_mul * self.vi + self.si + coords_dim + edge_dim_out, bias=True
            ),
        )

        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(
            in_dims=(self.si, self.vi),
            hs_dim=self.si,
            hv_dim=self.vi,
            out_dims=(self.so, self.vo),
            norm_eps=eps,
            use_mlp=use_mlp_update,
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        if self.has_v_in:
            reset(self.vector_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def aggregate(
        self,
        inputs: Tuple[Tensor, Tensor, Tensor, Tensor],
        index: Tensor,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(
            inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size
        )
        if self.update_coords:
            p = scatter(inputs[2], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        else:
            p = inputs[2]
        return s, v, p, inputs[-1]

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr_initial_ohe: Tensor,
        edge_attr_global_embedding: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        dim_size: Optional[int],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        d, a, r, e = edge_attr

        de0 = d.view(-1, 1)
        a0 = a.view(-1, 1)

        di, dj = torch.zeros_like(a0).to(a.device), torch.zeros_like(a0).to(a.device)
        aij = torch.cat([torch.cat([sa_i, sa_j], dim=-1), de0, a0, e, di, dj], dim=-1)
        aij = torch.cat([aij, edge_attr_global_embedding], dim=-1)

        rbf = self.radial_basis_func(d)
        assert edge_attr_initial_ohe is not None
        assert edge_attr_initial_ohe.size(1) == 3
        rbf_ohe = torch.einsum("nk, nd -> nkd", (rbf, edge_attr_initial_ohe))
        rbf_ohe = rbf_ohe.view(d.size(0), -1)
        aij = torch.cat([aij, rbf_ohe], dim=-1)
        aij = self.edge_net(aij)
        
        if self.update_coords:
            aij, p_ij = aij.split([aij.size(1) - 1, 1], dim=-1)
            p_j = p_ij * r
        else:
            p_j = torch.zeros_like(r)

        if self.update_edges:
            aij, edge = aij.split([aij.size(1) -  e.size(-1), e.size(-1)], dim=-1)
        else:
            edge = e
            
        if self.has_v_in:
            aij, vij0 = aij.split([self.si, self.v_mul * self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            if self.use_cross_product:
                vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
            else:
                vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            if self.use_cross_product:
                v_ij_cross = cross_product(va_i, va_j, dim=1)
                nv2_j = vij2 * v_ij_cross
                nv_j = nv0_j + nv1_j + nv2_j
            else:
                nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j, p_j, edge

    def forward(
        self,
        x: Tuple[Tensor, Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Tensor],
        edge_attr_initial_ohe: Tensor,
        edge_attr_global_embedding: Tensor,
        ligand_mask: Tensor,
    ):
        s, v, p = x
        d, a, r, e = edge_attr

        ms, mv, mp, me = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, a, r, e),
            edge_index=edge_index,
            dim_size=s.size(0),
            edge_attr_initial_ohe=edge_attr_initial_ohe,
            edge_attr_global_embedding=edge_attr_global_embedding,
        )

        s = ms + s
        v = mv + v
        
        if self.update_coords:
            p = (mp * ligand_mask.unsqueeze(-1)) + p
        if self.update_edges:
            e = me + e

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v, p, e
