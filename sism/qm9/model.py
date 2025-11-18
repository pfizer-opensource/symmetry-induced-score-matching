import torch
import torch.nn.functional as F
from torch import Tensor
import lightning.pytorch as pl
from torch_geometric.data import Batch
from torch_sparse import coalesce
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from torch_scatter import scatter_mean
from tqdm import tqdm

from sism.qm9.gnn import DiffusionScoreModelSphere
from sism.diffusion import (get_diffusion_coefficients,  GeneralizedScoreMatching3Nsphere)

def coalesce_edges(edge_index, bond_edge_index, bond_edge_attr, n):
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

def zero_mean(x: Tensor, batch: Tensor, dim_size: int, dim=0):
    out = x - scatter_mean(x, index=batch, dim=dim, dim_size=dim_size)[batch]
    return out

def get_mu_nu_mask(batch):
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
    return mu_mask, nu_mask, ptr0, ptr1


class TrainerSphere(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)
       
        self.model = DiffusionScoreModelSphere(
            atom_feat_dim=hparams["atom_feat_dim"],
            edge_feat_dim=hparams["edge_feat_dim"],
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            cutoff=hparams["cutoff"],
            num_layers=hparams["num_layers"],
            num_rbfs=hparams["num_rbfs"],
            use_cross_product=hparams["use_cross_product"],
            vector_aggr=hparams["vector_aggr"],
            gsm=not hparams["fisher_dsm"],
        )
                
        assert not hparams["ode_training"]
        if not hparams["fisher_dsm"]:
            self.gsm = GeneralizedScoreMatching3Nsphere()
        else:
            self.gsm = None
            
        betas = get_diffusion_coefficients(T=self.hparams.timesteps,
                                           kind=hparams["noise_schedule"]
                                           )
        alphas = 1.0 - betas
        mean_coeff = torch.cumsum(alphas.log(), dim=0).exp()
        std_coeff = (1.0 - mean_coeff).sqrt()
        self.register_buffer("betas", betas)
        self.register_buffer("mean_coeff", mean_coeff)
        self.register_buffer("std_coeff", std_coeff)
        
    def reverse_sampling(self,
                         x: Tensor, 
                         bond_edge_index: Tensor, 
                         bond_edge_attr: Tensor,
                         batch: Tensor,
                         r_dynamic: bool = True,
                         theta_dynamic: bool = True,
                         phi_dynamic: bool = True,
                         stochastic_dynamic: bool = True,
                         save_traj: bool = False,
                         tqdm_verbose=False,
                         casimir_component: bool = True,
                         divergence_component: bool = False,
                         ):
        
        assert (sum((r_dynamic, theta_dynamic, phi_dynamic)) > 0)
          
        bs = int(batch.max() + 1)
        
        edge_index_global = (torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int().fill_diagonal_(0))
        edge_index_global, _ = dense_to_sparse(edge_index_global)
        edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        edge_index_global, edge_attr_global = coalesce_edges(
            edge_index=edge_index_global,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=x.size(0),
        )
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
        )
        
        x = F.one_hot(x, num_classes=16).float()
        edge_attr_global = F.one_hot(edge_attr_global, num_classes=5).float()
        
        if self.hparams.fisher_dsm:
            pos = torch.randn((x.size(0), 3), device=x.device, dtype=x.dtype)
            pos = zero_mean(pos, batch, dim_size=bs, dim=0)
        else:
            pos = self.gsm.prior.sample(x.size(0), device=x.device)
            pos = self.gsm.to_cartesian(pos)
       
        chain = reversed(range(self.hparams.timesteps))
        pos_traj = []
        if save_traj:
            pos_traj.append(pos.cpu())
        
        for t in tqdm(chain, total=self.hparams.timesteps) if tqdm_verbose else chain:
            t_ = torch.tensor([t] * bs, dtype=torch.long, device=x.device)
            t_emb = t_.float() / self.hparams.timesteps
            t_emb = t_emb.clamp(min=self.hparams.eps_min)
            t_emb = t_emb.unsqueeze(dim=1)
            
            betas = self.betas[t_]
            betas = (betas[batch]).unsqueeze(-1)
            score = self.model(x=x,
                               pos=pos,
                               t=t_emb, 
                               edge_index=edge_index_global,
                               edge_attr=edge_attr_global, 
                               batch=batch)["score"]
            
            noise = torch.randn_like(score)
            
            if not self.hparams.fisher_dsm:
                r, theta, phi = self.gsm.to_spherical(pos).chunk(3, dim=-1)
                
                # dilation
                d_r = (0.5 * r.log() + score[:, 0].unsqueeze(-1))
                d_r =  d_r * torch.einsum('bij, bj -> bi', (self.gsm.I, pos))
                
                # rotation (x,y)
                A_theta = torch.concat([
                    -torch.sin(phi),
                    torch.cos(phi),
                    torch.zeros_like(phi),
                ], dim=-1)
                A_theta = self.gsm.vector_to_skew_matrix(A_theta)
                d_theta = (0.5 * theta + score[:, 1].unsqueeze(-1))
                d_theta = d_theta * torch.einsum('bij, bj -> bi', (A_theta, pos))
                
                # (rotation around z)
                d_phi = (0.5 * phi + score[:, 2].unsqueeze(-1)) 
                d_phi = d_phi * torch.einsum('bij, bj -> bi', (self.gsm.Az, pos))
                                
                nr = noise[:, 0].unsqueeze(-1) * torch.einsum('bij, bj -> bi', (self.gsm.I, pos))
                n_theta = noise[:, 1].unsqueeze(-1) * torch.einsum('bij, bj -> bi', (A_theta, pos))
                n_phi = noise[:, 2].unsqueeze(-1) * torch.einsum('bij, bj -> bi', (self.gsm.Az, pos))
                
                if r_dynamic:
                    pos_update = pos + betas * d_r
                else:
                    pos_update = pos
                    
                if theta_dynamic:
                    pos_update = pos_update + betas * d_theta
                    
                if phi_dynamic:
                    pos_update = pos_update + betas * d_phi
                
                if stochastic_dynamic:
                    # noise term + casimir invariant terms + divergence term
                    if r_dynamic:
                        
                        pos_update = pos_update + betas.sqrt() * nr
                        
                        if casimir_component:
                            r_casimir =  0.5 * betas * pos
                        else:
                            r_casimir = 0.0
                            
                        if divergence_component:
                            r_div = 3.0 * betas * pos
                        else:
                            r_div = 0.0
                        
                        pos_update = pos_update + r_casimir + r_div
                        
                    if theta_dynamic:
                        
                        pos_update = pos_update + betas.sqrt() * n_theta
                        
                        if casimir_component:
                            theta_casimir = - 0.5 * betas * pos
                        else:
                            theta_casimir = 0.0
                        
                        if divergence_component:
                            tmp = pos.clone()
                            z = (tmp[:, -1]).clone()
                            tmp[:, -1] = 0.0
                            xysq = tmp.pow(2).sum(-1).sqrt()
                            coeff = z / xysq.clamp_min(1e-2)
                            theta_div = coeff.unsqueeze(-1) * torch.einsum('bij, bj -> bi', (A_theta, pos))
                        else:
                            theta_div = 0.0
                        
                        pos_update = pos_update + theta_casimir + theta_div
                        
                    if phi_dynamic:
                        
                        pos_update = pos_update + betas.sqrt() * n_phi
                                            
                        if casimir_component:
                            tmp = pos.clone()
                            tmp[:, -1] = 0.0
                            phi_casimir = - 0.5 * betas * tmp 
                        else:
                            phi_casimir = 0.0
                            
                        if divergence_component:
                            phi_div = 0.0
                        else:
                            phi_div = 0.0
                        
                        pos_update = pos_update + phi_casimir + phi_div
            else:
                pos_update = pos + 0.5 * betas * pos + betas * score
                if stochastic_dynamic:
                    pos_update = pos_update + betas.sqrt() * zero_mean(noise, batch, dim_size=bs, dim=0)
            
            pos = pos_update
            # no re-centering
            if save_traj:
                pos_traj.append(pos.detach().cpu())
            
        return pos, pos_traj
            
    def training_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step_fnc(batch=batch, batch_idx=batch_idx, stage="val")

    def step_fnc(self, batch, batch_idx, stage: str):
        batch_size = int(batch.batch.max()) + 1
        
        t = torch.randint(
            low=1,
            high=self.hparams.timesteps + 1,
            size=(batch_size,),
            dtype=torch.long,
            device=batch.x.device,
        )
       
        out_dict = self(batch=batch, t=t)
      
        sigma = out_dict["sigma_t"]
        sigma_b = sigma[batch.batch]
        target = -(1.0 / sigma_b) * out_dict["target"]
        pred = out_dict["score"]
        loss = scatter_mean((pred - target).pow(2).sum(-1),
                            dim=0, 
                            index=batch.batch, 
                            dim_size=batch_size
                            ).mean()
        
        self._log(loss=loss,
                  batch_size=int((batch.batch.max() + 1)),
                  stage=stage,
                  )
            
        return loss
        
    def forward(self, batch: Batch, t: Tensor):
        x: Tensor = batch.x
        pos: Tensor = batch.pos
        data_batch: Tensor = batch.batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bs = int(data_batch.max()) + 1

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        if not hasattr(batch, "fc_edge_index"):
            edge_index_global = (
                torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            edge_index_global, _ = dense_to_sparse(edge_index_global)
            edge_index_global = sort_edge_index(edge_index_global, sort_by_row=False)
        else:
            edge_index_global = batch.fc_edge_index

        edge_index_global, edge_attr_global = coalesce_edges(
            edge_index=edge_index_global,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=pos.size(0),
        )
        edge_index_global, edge_attr_global = sort_edge_index(
            edge_index=edge_index_global, edge_attr=edge_attr_global, sort_by_row=False
        )

        # as of now keep centered around origin, need to double check
        pos_centered = zero_mean(pos, data_batch, dim=0, dim_size=bs)
        m = self.mean_coeff[t].unsqueeze(-1)
        s = self.std_coeff[t].unsqueeze(-1)
        
        if self.hparams.fisher_dsm:
            noise_centered = torch.randn_like(pos_centered)
            noise_centered = zero_mean(noise_centered, data_batch, dim=0, dim_size=bs)
            m = m.sqrt()
            mean_t =  m[batch.batch] * pos_centered 
            eps_t = s[batch.batch] * noise_centered
            pos_t = mean_t + eps_t
            target = noise_centered
        else:
            r_theta_phi = self.gsm.to_spherical(pos_centered)
            noise = torch.randn_like(r_theta_phi)
            # try with and without m.sqrt()
            m = m.sqrt()
            r_theta_phi[:, 0] = r_theta_phi[:, 0].log()
            r_theta_phi_t =  m[batch.batch] * r_theta_phi + s[batch.batch] * noise
            r_theta_phi_t[:, 0] = r_theta_phi_t[:, 0].exp()
            pos_t = self.gsm.to_cartesian(r_theta_phi_t) 
            target = noise
            
        x = F.one_hot(x, num_classes=16).float()
        edge_attr_global = F.one_hot(edge_attr_global, num_classes=5).float()
        
        out = self.model(x=x,
                         pos=pos_t,
                         t=temb, 
                         edge_index=edge_index_global,
                         edge_attr=edge_attr_global, 
                         batch=data_batch
                         )

        out["sigma_t"] = s
        out["pos_t"] = pos_t
        out["target"] = target        
        out["x0"] = pos_centered
        out["xt"] = pos_t

        return out

    def _log(
        self,
        loss,
        batch_size,
        stage,
    ):
        self.log(
            f"{stage}/loss",
            loss,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=self.hparams["lr_patience"],
            cooldown=self.hparams["lr_cooldown"],
            factor=self.hparams["lr_factor"],
        )

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams["lr_frequency"],
            "monitor": "val/loss_epoch",
            "strict": False,
        }
        return [optimizer], [scheduler]


if __name__ == '__main__':
    hparams={
        "atom_feat_dim": 16,
        "edge_feat_dim": 16,
        "sdim": 128,
        "vdim": 64,
        "cutoff": 5.0,
        "num_layers": 5,
        "num_rbfs": 20,
        "use_cross_product": False,
        "vector_aggr": "mean",
        "timesteps": 100,
        "fisher_dsm": False,
        "noise_schedule": "linear-time",
        "ode_training": False,
    }
    model = TrainerSphere(hparams=hparams)
    print(model)
    print(sum(m.numel() for m in model.parameters() if m.requires_grad))
