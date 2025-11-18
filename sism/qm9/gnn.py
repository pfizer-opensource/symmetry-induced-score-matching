import torch
from torch import nn, Tensor
from typing import Optional, Tuple, Dict
from torch_scatter import scatter_mean

from sism.modules import GatedEquivBlock, DenseLayer, LayerNorm
from sism.qm9.convs import EQGATConv

class EQGATGNN(nn.Module):
    def __init__(
        self,
        hn_dim: Tuple[int, int] = (256, 128),
        cutoff: float = 5.0,
        num_layers: int = 5,
        num_rbfs: int = 20,
        edge_dim: Optional[int] = None,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
    ):
        super(EQGATGNN, self).__init__()
        self.sdim, self.vdim = hn_dim
        self.cutoff = cutoff
        self.num_layers = num_layers

        convs = []

        for i in range(num_layers):
            convs.append(
                EQGATConv(
                    in_dims=hn_dim,
                    out_dims=hn_dim,
                    num_rbfs=num_rbfs,
                    edge_dim=edge_dim,
                    cutoff=cutoff,
                    has_v_in=i > 0,
                    use_mlp_update=i < (num_layers - 1),
                    vector_aggr=vector_aggr,
                    use_cross_product=use_cross_product,
                )
            )

        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList([LayerNorm(dims=hn_dim) for _ in range(num_layers)])
        self.out_norm = LayerNorm(hn_dim)
        
    def forward(
        self,
        s: Tensor,
        v: Tensor,
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor, Tensor, Optional[Tensor]],
        batch: Tensor = None,
    ) -> Dict:
        for i in range(len(self.convs)):
            s, v = self.norms[i](x={"s": s, "v": v}, batch=batch)
            out = self.convs[i](
                x=(s, v),
                edge_index=edge_index,
                edge_attr=edge_attr,
            )
            s, v = out
        s, v = self.out_norm(x={"s": s, "v": v}, batch=batch)
        return s, v
    
class DiffusionScoreModelSphere(nn.Module):
    def __init__(self,
        atom_feat_dim: int,
        edge_feat_dim: int,
        hn_dim: Tuple[int, int] = (256, 128),
        cutoff: float = 5.0,
        num_layers: int = 5,
        num_rbfs: int = 20,
        use_cross_product: bool = False,
        vector_aggr: str = "mean",
        gsm: bool = True,
        ):
        
        super(DiffusionScoreModelSphere, self).__init__()
        
        self.hn_dim = hn_dim
        self.atom_mapping = DenseLayer(atom_feat_dim, hn_dim[0], bias=True) 
        self.edge_mapping = DenseLayer(edge_feat_dim, hn_dim[0] // 2, bias=True)   
        self.time_mapping = DenseLayer(1, hn_dim[0], bias=True)
        self.gnn = EQGATGNN(hn_dim=hn_dim,
                            cutoff=cutoff,
                            num_layers=num_layers,
                            num_rbfs=num_rbfs,
                            edge_dim=hn_dim[0] // 2,
                            use_cross_product=use_cross_product,
                            vector_aggr=vector_aggr,
                            )
        

        self.gsm = gsm
        
        if gsm:
            self.gated = GatedEquivBlock(in_dims=hn_dim,
                                         out_dims=(hn_dim[0], hn_dim[1]),
                                         use_mlp=False
                                        )
            self.act_fnc = nn.Softplus()
            self.out_score = nn.Sequential(DenseLayer(hn_dim[0] + 3 * hn_dim[1], hn_dim[0], activation=nn.SiLU()),
                                           DenseLayer(hn_dim[0], hn_dim[0], activation=nn.SiLU()),
                                           DenseLayer(hn_dim[0], 3)
                                           )
        else:
            self.gated = None
            self.act_fnc = None
            self.out_score = DenseLayer(hn_dim[1], 1, bias=False)
    
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

    def forward(self, 
                x: Tensor,
                pos: Tensor,
                t: Tensor,
                edge_index: Tensor,
                edge_attr: Tensor, 
                batch: Tensor) -> Dict:
        
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        s = self.atom_mapping(x) + self.time_mapping(t)[batch]
        e = self.edge_mapping(edge_attr)
        edge_attr = self.calculate_edge_attrs(edge_index, e, pos)
        
        v = torch.zeros((x.size(0), 3, self.hn_dim[1]), 
                        device=x.device,
                        dtype=pos.dtype
                        )
        
        s, v = self.gnn(s, v, edge_index, edge_attr, batch)
        
        if self.gsm:
            s, v = self.gated((s, v))
            v = v.reshape(v.size(0), -1)
            input = torch.cat([s, v], dim=-1)
            scores = self.out_score(input)
            out_dict = {
                "score": scores,
                }
        else:
            scores = self.out_score(v).squeeze(-1)
            scores = (0.0 * s).mean() + scores
            scores = scores - scatter_mean(scores, batch, dim=0)[batch]
            out_dict = {"score": scores}
            
        return out_dict