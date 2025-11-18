import torch
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_mean
from typing import Dict, Tuple
import lightning.pytorch as pl
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, sort_edge_index
from tqdm import tqdm
import numpy as np
from functools import partial
from sism.diffusion import get_diffusion_coefficients
from sism.diffusion import GeneralizedScoreMatching3Nsphere
from sism.plcomplex.gnn import (
    DiffusionScoreModelSphere,
    DiffusionScoreModelRiemannian,
    DiffusionScoreModelFisher,
)
from sism.plcomplex.rsgm import so3
from sism.plcomplex.rsgm.geometry import axis_angle_to_matrix
from sism.plcomplex.rsgm.scheduler import get_sigma_scheduler
from sism.plcomplex.utils import (
    coalesce_edges,
    combine_protein_ligand_feats,
    get_mu_nu_mask,
    compute_PCs_batch,
)


def get_brownian_bridge_scheduler(T, kind="linear"):
    if kind == "linear":
        m_min, m_max = 0.001, 0.999
        m_t = np.linspace(m_min, m_max, T)
    elif kind == "sin":
        m_t = 1.0075 ** np.linspace(0, T, T)
        m_t = m_t / m_t[-1]
        m_t[-1] = 0.999
    return m_t


class TrainerSphere(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.num_atom_types = hparams["atom_feat_dim"]
        self.model = DiffusionScoreModelSphere(
            atom_feat_dim=hparams["atom_feat_dim"],
            edge_feat_dim=hparams["edge_feat_dim"],
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            edim=hparams["edim"],
            cutoff=hparams["cutoff"],
            num_layers=hparams["num_layers"],
            num_rbfs=hparams["num_rbfs"],
            use_cross_product=hparams["use_cross_product"],
            vector_aggr=hparams["vector_aggr"],
            global_translations=hparams["global_translations"],
            update_coords=hparams["update_coords"],
            update_edges=hparams["update_edges"],
        )
        self.gsm = GeneralizedScoreMatching3Nsphere()
        self.global_translations = hparams["global_translations"]

        betas = get_diffusion_coefficients(
            T=self.hparams.timesteps, kind=hparams["noise_schedule"]
        )
        alphas = 1.0 - betas
        mean_coeff = torch.cumsum(alphas.log(), dim=0).exp()
        std_coeff = (1.0 - mean_coeff).sqrt()
        self.register_buffer("betas", betas)
        self.register_buffer("mean_coeff", mean_coeff)
        self.register_buffer("std_coeff", std_coeff)

    def get_flow_coordinates(
        self,
        x1: Tensor,
        x2: Tensor,
    ) -> Dict[str, Tensor]:
        assert x1.ndim == 2 and x2.ndim == 2
        theta1 = torch.atan2(x1[:, [0, 1]].pow(2).sum(dim=-1).sqrt(), x1[:, 2])
        phi1 = torch.atan2(x1[:, 1], x1[:, 0])
        Ry_m = self.gsm.get_so3_cartesian_axes(-1.0 * theta1, axis="y")
        Rz_m = self.gsm.get_so3_cartesian_axes(-1.0 * phi1, axis="z")
        R_m = Ry_m @ Rz_m
        x2_tilde = torch.einsum("bij,bj->bi", R_m, x2)
        phi2 = torch.atan2(x2_tilde[:, 1], x2_tilde[:, 0])
        Rz2_m = self.gsm.get_so3_cartesian_axes(-1.0 * phi2, axis="z")
        tau = torch.stack([theta1, phi1, phi2], axis=1)
        return {"tau": tau, "Rz_m": Rz_m, "Ry_m": Ry_m, "R_m": R_m, "Rz2_m": Rz2_m}

    def forward_noising(
        self,
        pos_lig: Tensor,
        batch: Tensor,
        m: Tensor,
        s: Tensor,
        # center_ligand: bool = True
        use_PCs: bool = False,
    ) -> Tuple[Tensor, ...]:

        com = scatter_mean(pos_lig, batch, dim=0)
        pos_lig_0com = pos_lig - com[batch]

        if not use_PCs:
            mask_ptr = get_mu_nu_mask(batch)
            x1 = pos_lig_0com[mask_ptr["ptr0"]]  # (#Gr, 3)
            x2 = pos_lig_0com[mask_ptr["ptr1"]]
            fout = self.get_flow_coordinates(x1, x2)
            tau = fout["tau"]
        else:
            pc1, pc2 = compute_PCs_batch(pos_lig_0com, batch)
            fout = self.get_flow_coordinates(pc1, pc2)
            tau = fout["tau"]
            # raise NotImplementedError
        noise = torch.randn_like(tau)

        tau_t = m * tau + s * noise
        theta1_t, phi1_t, phi2_t = tau_t.chunk(3, dim=1)
        theta1_t = theta1_t.squeeze()
        phi1_t = phi1_t.squeeze()
        phi2_t = phi2_t.squeeze()

        R = fout["Rz2_m"] @ fout["R_m"]
        R = R.repeat_interleave(batch.bincount(), dim=0)

        pos_hat = torch.einsum("bij,bj->bi", R, pos_lig_0com)
        # now we apply the forward noising on the pos_hat
        Rz_phi1_t = self.gsm.get_so3_cartesian_axes(phi1_t, axis="z")
        Ry_theta1_t = self.gsm.get_so3_cartesian_axes(theta1_t, axis="y")
        Rz_phi2_t = self.gsm.get_so3_cartesian_axes(phi2_t, axis="z")

        # get rotation matrix
        R_t = Rz_phi1_t @ Ry_theta1_t @ Rz_phi2_t
        R_t = R_t.repeat_interleave(batch.bincount(), dim=0)
        pos_t = torch.einsum("bij,bj->bi", R_t, pos_hat)

        noise_com = torch.randn_like(com)
        if self.hparams.global_translations:
            com_t = m * com + s * noise_com
        else:
            com_t = com

        pos_t = pos_t + com_t[batch]

        return tau, tau_t, noise, pos_t, com_t, noise_com

    def apply_full_rotation_transform(
        self,
        betas,
        batch_ligand,
        theta1_t,
        score_theta_1,
        phi1_t,
        score_phi_1,
        phi2_t,
        score_phi_2,
        pos_t_ligand,
        theta1_dynamic: bool = True,
        phi1_dynamic: bool = True,
        phi2_dynamic: bool = True,
        stochastic_dynamic: bool = True,
    ):

        assert (
            scatter_mean(pos_t_ligand, batch_ligand, dim=0)[batch_ligand].pow(2).mean()
            < 1e-8
        )

        d_theta_1 = betas * (0.5 * theta1_t + score_theta_1)
        d_phi_1 = betas * (0.5 * phi1_t + score_phi_1)
        d_phi_2 = betas * (0.5 * phi2_t + score_phi_2)

        if stochastic_dynamic:
            d_theta_1 = d_theta_1 + betas.sqrt() * torch.randn_like(d_theta_1)
            d_phi_1 = d_phi_1 + betas.sqrt() * torch.randn_like(d_phi_1)
            d_phi_2 = d_phi_2 + betas.sqrt() * torch.randn_like(d_phi_2)

        Rz_dphi_1 = self.gsm.get_so3_cartesian_axes(d_phi_1, axis="z")
        Ry_dtheta_1 = self.gsm.get_so3_cartesian_axes(d_theta_1, axis="y")
        Rz_dphi_2 = self.gsm.get_so3_cartesian_axes(d_phi_2, axis="z")

        bs = len(batch_ligand.bincount())
        I = torch.eye(3, device=pos_t_ligand.device).unsqueeze(0).repeat(bs, 1, 1)
        if not theta1_dynamic:
            Ry_dtheta_1 = I
        if not phi1_dynamic:
            Rz_dphi_1 = I
        if not phi2_dynamic:
            Rz_dphi_2 = I
        # get finite rotation matrix
        dR = Rz_dphi_1 @ Ry_dtheta_1 @ Rz_dphi_2
        dR = dR.repeat_interleave(batch_ligand.bincount(), dim=0)
        pos_t_ligand = torch.einsum("bij,bj->bi", dR, pos_t_ligand)
        return pos_t_ligand

    def reverse_sampling(
        self,
        atom_types_ligand: Tensor,
        atom_types_pocket: Tensor,
        pos_ligand: Tensor,
        pos_pocket: Tensor,
        bond_edge_index: Tensor,
        bond_edge_attr: Tensor,
        batch_ligand: Tensor,
        batch_pocket: Tensor,
        theta1_dynamic: bool = True,
        phi1_dynamic: bool = True,
        phi2_dynamic: bool = True,
        stochastic_dynamic: bool = True,
        save_traj: bool = False,
        tqdm_verbose=False,
        casimir_component: bool = True,
        divergence_component: bool = False,
        full_rotation: bool = True,
        use_global_translations: bool = True,
        use_global_rotations: bool = True,
        use_PCs: bool = False,
    ):

        # We shift everything in pocket com
        pocket_com = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pos_pocket_0com = pos_pocket - pocket_com[batch_pocket]
        pos_ligand = pos_ligand - pocket_com[batch_ligand]

        # OHE
        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        atom_types_ligand = F.one_hot(
            atom_types_ligand.squeeze().long(), num_classes=self.num_atom_types
        ).float()

        edge_index_global_lig = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
        edge_index_global_lig = sort_edge_index(
            edge_index_global_lig, sort_by_row=False
        )

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )
        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )

        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        com = scatter_mean(pos_ligand, batch_ligand, dim=0)
        # We need these only for computing angles and applying rotations
        pos_lig_0com = pos_ligand - com[batch_ligand]

        assert sum((theta1_dynamic, phi1_dynamic, phi2_dynamic)) > 0
        bs = int(batch_ligand.max() + 1)

        tau = self.gsm.prior.sample(bs, device=atom_types_ligand.device)

        # theta1, phi1, phi2 = tau.chunk(3, dim=1)

        if use_global_translations:
            # we sample com_t from N(0,1)
            pos_com_t = torch.randn_like(com)
        else:
            pos_com_t = com

        if not use_PCs:
            mask_ptr = get_mu_nu_mask(batch_ligand)
            x1 = pos_lig_0com[mask_ptr["ptr0"]]
            x2 = pos_lig_0com[mask_ptr["ptr1"]]
            fout = self.get_flow_coordinates(x1, x2)
        else:
            pc1, pc2 = compute_PCs_batch(pos_lig_0com, batch_ligand)
            fout = self.get_flow_coordinates(pc1, pc2)
        tau_0 = fout["tau"]

        noise = torch.randn_like(tau)
        tau_t = 0.0 * tau_0 + 1.0 * noise
        theta1_t, phi1_t, phi2_t = tau_t.chunk(3, dim=1)
        theta1_t = theta1_t.squeeze(dim=1)
        phi1_t = phi1_t.squeeze(dim=1)
        phi2_t = phi2_t.squeeze(dim=1)

        R = fout["Rz2_m"] @ fout["R_m"]
        R = R.repeat_interleave(batch_ligand.bincount(), dim=0)
        pos_hat = torch.einsum("bij,bj->bi", R, pos_lig_0com)
        Rz_phi1_t = self.gsm.get_so3_cartesian_axes(phi1_t, axis="z")
        Ry_theta1_t = self.gsm.get_so3_cartesian_axes(theta1_t, axis="y")
        Rz_phi2_t = self.gsm.get_so3_cartesian_axes(phi2_t, axis="z")

        # get rotation matrix, (completely noise)
        R_t = Rz_phi1_t @ Ry_theta1_t @ Rz_phi2_t
        R_t = R_t.repeat_interleave(batch_ligand.bincount(), dim=0)
        pos_t_ligand_0cm = torch.einsum("bij,bj->bi", R_t, pos_hat)

        pos_t_ligand = pos_t_ligand_0cm + pos_com_t[batch_ligand]

        chain = reversed(range(self.hparams.timesteps))
        pos_traj = []
        scores_traj = []
        scores_traj_com = []
        tau_traj = []
        batch_num_nodes = batch_ligand.bincount()
        if save_traj:
            pos_traj.append(pos_t_ligand.cpu())
            tau_traj.append(tau_t.cpu())

        # pos_pocket_com = pos_pocket - scatter_mean(pos_ligand, batch_ligand, dim=0)[batch_pocket]

        for t in tqdm(chain, total=self.hparams.timesteps) if tqdm_verbose else chain:

            t_ = torch.tensor([t] * bs, dtype=torch.long, device=pos_ligand.device)
            t_emb = t_.float() / self.hparams.timesteps
            t_emb = t_emb.clamp(min=self.hparams.eps_min)
            t_emb = t_emb.unsqueeze(dim=1)

            betas = self.betas[t_]

            # combine protein and ligand in one representation for translations
            (
                pos_joint,
                atom_types_joint,
                batch_full,
                mask_ligand,
                edge_index_global,
                edge_attr_global,
                _,
                edge_initial_interaction,
                batch_edge_global,
            ) = combine_protein_ligand_feats(
                pos_ligand=pos_t_ligand,
                pos_pocket=pos_pocket_0com,
                atom_types_ligand=atom_types_ligand,
                atom_types_pocket=atom_types_pocket,
                batch_ligand=batch_ligand,
                batch_pocket=batch_pocket,
                edge_attr_ligand=edge_attr_global_lig,
                num_bond_classes=5,
                cutoff_p=self.hparams.cutoff,
                cutoff_lp=self.hparams.cutoff,
            )

            out = self.model(
                x=atom_types_joint,
                pos=pos_joint,
                t=t_emb,
                edge_index=edge_index_global,
                edge_attr=edge_attr_global,
                edge_attr_initial_ohe=edge_initial_interaction,
                batch=batch_full,
                batch_ligand=batch_ligand,
                mask_ligand=mask_ligand,
                batch_edge=batch_edge_global,
            )

            scores_rot = out["score"]
            scores_com = out["score_com"]

            score_theta_1, score_phi_1, score_phi_2 = scores_rot.chunk(3, dim=1)
            (score_theta_1, score_phi_1, score_phi_2) = (
                score_theta_1.squeeze(dim=1),
                score_phi_1.squeeze(dim=1),
                score_phi_2.squeeze(dim=1),
            )

            if not full_rotation:
                # Not gonna update this now
                if theta1_dynamic:
                    A_theta_1 = torch.cat(
                        [
                            -torch.sin(phi1_t).unsqueeze(-1),
                            torch.cos(phi1_t).unsqueeze(-1),
                            torch.zeros_like(phi1_t).unsqueeze(-1),
                        ],
                        dim=-1,
                    )

                    A_theta_1 = self.gsm.vector_to_skew_matrix(A_theta_1)

                    d_theta_1 = (0.5 * theta1_t + score_theta_1).unsqueeze(-1)
                    d_theta_1 = d_theta_1.repeat_interleave(batch_num_nodes, dim=0)
                    lie_alg_theta_1 = torch.einsum(
                        "bij,bj->bi",
                        A_theta_1.repeat_interleave(batch_num_nodes, dim=0),
                        pos_t_ligand,
                    )

                    d_theta_1 = d_theta_1 * lie_alg_theta_1

                    if casimir_component:
                        casimir_theta_1 = torch.einsum(
                            "bij,bjk->bik", A_theta_1, A_theta_1
                        ).repeat_interleave(batch_num_nodes, dim=0)
                        casimir_theta_1 = torch.einsum(
                            "bij,bj->bi", casimir_theta_1, pos_t_ligand
                        )
                    else:
                        casimir_theta_1 = 0.0

                    d_theta_1 = d_theta_1 + 0.5 * casimir_theta_1
                else:
                    d_theta_1 = 0.0

                if phi1_dynamic:
                    d_phi_1 = (
                        (0.5 * phi1_t + score_phi_1)
                        .unsqueeze(-1)
                        .repeat_interleave(batch_num_nodes, dim=0)
                    )
                    d_phi_1 = d_phi_1 * torch.einsum(
                        "bij,bj->bi", self.gsm.Az, pos_t_ligand
                    )

                    if casimir_component:
                        casimir_phi_1 = self.gsm.Az2
                        casimir_phi_1 = torch.einsum(
                            "bij,bj->bi", casimir_phi_1, pos_t_ligand
                        )
                    else:
                        casimir_phi_1 = 0.0

                    d_phi_1 = d_phi_1 + 0.5 * casimir_phi_1
                else:
                    d_phi_1 = 0.0

                if phi2_dynamic:
                    A_phi_2 = torch.cat(
                        [
                            (torch.cos(phi1_t) * torch.sin(theta1_t)).unsqueeze(-1),
                            (torch.sin(phi1_t) * torch.sin(theta1_t)).unsqueeze(-1),
                            torch.cos(theta1_t).unsqueeze(-1),
                        ],
                        dim=-1,
                    )
                    A_phi_2 = self.gsm.vector_to_skew_matrix(A_phi_2)
                    d_phi_2 = (
                        (0.5 * phi2_t + score_phi_2)
                        .unsqueeze(-1)
                        .repeat_interleave(batch_num_nodes, dim=0)
                    )
                    d_phi_2 = d_phi_2 * torch.einsum(
                        "bij,bj->bi",
                        A_phi_2.repeat_interleave(batch_num_nodes, dim=0),
                        pos_t_ligand,
                    )

                    if casimir_component:
                        casimir_phi_2 = torch.einsum(
                            "bij,bjk->bik", A_phi_2, A_phi_2
                        ).repeat_interleave(batch_num_nodes, dim=0)
                        casimir_phi_2 = torch.einsum(
                            "bij,bj->bi", casimir_phi_2, pos_t_ligand
                        )
                    else:
                        casimir_phi_2 = 0.0

                    d_phi_2 = d_phi_2 + 0.5 * casimir_phi_2

                else:
                    d_phi_2 = 0.0

                pos_t_ligand = pos_t_ligand + betas[batch_ligand].unsqueeze(-1) * (
                    d_phi_1 + d_phi_2 + d_theta_1
                )
            else:
                # The below assumes ligand in com
                pos_t_ligand_0com = pos_t_ligand - pos_com_t[batch_ligand]
                pos_t_ligand_0com = self.apply_full_rotation_transform(
                    betas,
                    batch_ligand,
                    theta1_t,
                    score_theta_1,
                    phi1_t,
                    score_phi_1,
                    phi2_t,
                    score_phi_2,
                    pos_t_ligand_0com,
                    theta1_dynamic=theta1_dynamic,
                    phi1_dynamic=phi1_dynamic,
                    phi2_dynamic=phi2_dynamic,
                    stochastic_dynamic=stochastic_dynamic,
                )

            # get flow coordinates of updates
            if not use_PCs:
                x1 = pos_t_ligand_0com[mask_ptr["ptr0"]]
                x2 = pos_t_ligand_0com[mask_ptr["ptr1"]]
                fout = self.get_flow_coordinates(x1, x2)
            else:
                pc1, pc2 = compute_PCs_batch(pos_t_ligand_0com, batch_ligand)
                fout = self.get_flow_coordinates(pc1, pc2)
            tau_t = fout["tau"]
            theta1_t, phi1_t, phi2_t = tau_t.chunk(3, dim=1)
            theta1_t = theta1_t.squeeze(dim=1)
            phi1_t = phi1_t.squeeze(dim=1)
            phi2_t = phi2_t.squeeze(dim=1)

            if use_global_translations:
                pos_com_t = pos_com_t + 0.5 * betas.unsqueeze(-1) * pos_com_t
                pos_com_t = pos_com_t + betas.unsqueeze(-1) * scores_com
                pos_com_t = pos_com_t + betas.sqrt().unsqueeze(-1) * torch.randn_like(
                    pos_com_t
                )
            else:
                pos_com_t = com

            pos_t_ligand = pos_t_ligand_0com + pos_com_t[batch_ligand]

            if save_traj:
                pos_traj.append(
                    pos_t_ligand.cpu()
                    # + pos_com_t[batch_ligand].cpu()
                    # + pocket_com[batch_ligand].cpu()
                )
                scores_traj.append(scores_rot.cpu().detach())
                scores_traj_com.append(scores_com.cpu().detach())
                tau_traj.append(tau_t.cpu())

        return pos_t_ligand, pos_traj, scores_traj, tau_traj, scores_traj_com

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
        target = -(1.0 / sigma) * out_dict["target"]
        pred = out_dict["score"]

        loss = (pred - target).pow(2)

        loss_theta1 = loss[:, 0].mean()
        loss_phi1 = loss[:, 1].mean()
        loss_phi2 = loss[:, 2].mean()
        loss = loss_theta1 + loss_phi1 + loss_phi2

        if self.hparams.global_translations:
            pred_com = out_dict["score_com"]
            target_com = -(1.0 / sigma) * out_dict["target_com"]
            loss_com = (pred_com - target_com).pow(2).sum(-1).mean(0)
        else:
            loss_com = 0.0

        loss = loss + loss_com

        self._log(
            loss_theta1=loss_theta1,
            loss_phi1=loss_phi1,
            loss_phi2=loss_phi2,
            loss_com=loss_com,
            loss=loss,
            batch_size=int((batch.batch.max() + 1)),
            stage=stage,
        )

        return loss

    def forward(self, batch: Batch, t: Tensor):

        atom_types_ligand: Tensor = batch.x
        atom_types_pocket: Tensor = batch.x_pocket
        pos_ligand: Tensor = batch.pos
        pos_pocket: Tensor = batch.pos_pocket
        batch_ligand: Tensor = batch.batch
        batch_pocket: Tensor = batch.pos_pocket_batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        # We shift everything in pocket com
        pocket_com = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pos_pocket_0com = pos_pocket - pocket_com[batch_pocket]
        pos_ligand = pos_ligand - pocket_com[batch_ligand]

        out = {}

        # OHE
        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        atom_types_ligand = F.one_hot(
            atom_types_ligand.squeeze().long(), num_classes=self.num_atom_types
        ).float()

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        if not hasattr(batch, "fc_edge_index"):
            edge_index_global_lig = (
                torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
            edge_index_global_lig = sort_edge_index(
                edge_index_global_lig, sort_by_row=False
            )
        else:
            edge_index_global_lig = batch.fc_edge_index

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )

        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        m = self.mean_coeff[t].unsqueeze(-1).sqrt()
        s = self.std_coeff[t].unsqueeze(-1)

        tau, tau_t, noise, pos_t_ligand, com_t, noise_com = self.forward_noising(
            pos_lig=pos_ligand,
            batch=batch_ligand,
            m=m,
            s=s,
            use_PCs=self.hparams.use_pcs,
        )

        # combine protein and ligand in one representation for translations
        (
            pos_joint,
            atom_types_joint,
            batch_full,
            mask_ligand,
            edge_index_global,
            edge_attr_global,
            _,
            edge_initial_interaction,
            batch_edge_global,
        ) = combine_protein_ligand_feats(
            pos_ligand=pos_t_ligand,
            pos_pocket=pos_pocket_0com,
            atom_types_ligand=atom_types_ligand,
            atom_types_pocket=atom_types_pocket,
            batch_ligand=batch_ligand,
            batch_pocket=batch_pocket,
            edge_attr_ligand=edge_attr_global_lig,
            num_bond_classes=5,
            cutoff_p=self.hparams.cutoff,
            cutoff_lp=self.hparams.cutoff,
        )

        out = self.model(
            x=atom_types_joint,
            pos=pos_joint,
            t=temb,
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            edge_attr_initial_ohe=edge_initial_interaction,
            batch=batch_full,
            batch_ligand=batch_ligand,
            mask_ligand=mask_ligand,
            batch_edge=batch_edge_global,
        )

        scores_rot = out["score"]
        scores_com = out["score_com"] if self.global_translations else 0.0

        out["score"] = scores_rot
        out["score_com"] = scores_com
        out["sigma_t"] = s
        out["pos_t"] = pos_t_ligand
        out["target"] = noise
        out["tau"] = tau
        out["tau_t"] = tau_t
        out["edge_index_global_translations"] = edge_index_global
        out["edge_index_global_translations_rot"] = edge_index_global
        out["target_com"] = noise_com

        return out

    def _log(
        self,
        loss_theta1,
        loss_phi1,
        loss_phi2,
        loss_com,
        loss,
        batch_size,
        stage,
    ):

        self.log(
            f"{stage}/loss_theta1",
            loss_theta1,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/loss_phi1",
            loss_phi1,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/loss_phi2",
            loss_phi2,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        if self.hparams.global_translations:
            self.log(
                f"{stage}/loss_com",
                loss_com,
                on_step=True,
                batch_size=batch_size,
                prog_bar=(stage == "train"),
                sync_dist=self.hparams.gpus > 1 and stage == "val",
            )
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


class TrainerRSGM(pl.LightningModule):
    def __init__(
        self,
        hparams: dict,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.num_atom_types = hparams["atom_feat_dim"]
        self.model = DiffusionScoreModelRiemannian(
            atom_feat_dim=hparams["atom_feat_dim"],
            edge_feat_dim=hparams["edge_feat_dim"],
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            edim=hparams["edim"],
            cutoff=hparams["cutoff"],
            num_layers=hparams["num_layers"],
            num_rbfs=hparams["num_rbfs"],
            use_cross_product=hparams["use_cross_product"],
            vector_aggr=hparams["vector_aggr"],
            global_translations=hparams["global_translations"],
            update_coords=hparams["update_coords"],
            update_edges=hparams["update_edges"],
        )

        self.global_translations = hparams["global_translations"]
        # for global translations use variance preserving scheduler SDE
        betas = get_diffusion_coefficients(
            T=self.hparams.timesteps, kind=hparams["noise_schedule"]
        )
        alphas = 1.0 - betas
        mean_coeff = torch.cumsum(alphas.log(), dim=0).exp()
        std_coeff = (1.0 - mean_coeff).sqrt()
        self.register_buffer("betas", betas)
        self.register_buffer("mean_coeff", mean_coeff)
        self.register_buffer("std_coeff", std_coeff)

        # for global rotations use variance exploding scheduler SDE
        self.sigma_min = 0.01
        self.sigma_max = 2.0
        sigmas = get_sigma_scheduler(
            min_sigma=self.sigma_min,
            max_sigma=self.sigma_max,
            n=self.hparams.timesteps + 1,
        )
        self.register_buffer("sigmas", torch.from_numpy(sigmas).float())

    def forward_noising(
        self,
        pos_lig: Tensor,
        batch: Tensor,
        m_translation: Tensor,
        s_translation: Tensor,
        s_rotation: Tensor,
    ) -> Tuple[Tensor, ...]:

        com = scatter_mean(pos_lig, batch, dim=0)
        pos_lig_0com = pos_lig - com[batch]

        # get random rotation
        rot_updates = [
            so3.sample_vec(eps=s.item()) for s in s_rotation.cpu().numpy().squeeze()
        ]
        rot_matrices = [
            axis_angle_to_matrix(torch.from_numpy(rot_update.squeeze())).float()
            for rot_update in rot_updates
        ]
        rot_matrices = torch.stack(rot_matrices).to(pos_lig.device)
        R_t = rot_matrices[batch]
        pos_t_rotated = torch.einsum("bij,bj->bi", R_t, pos_lig_0com)
        rot_scores = []
        for rot_update, rot_sigma in zip(
            rot_updates, s_rotation.cpu().numpy().squeeze()
        ):
            rot_score = torch.from_numpy(
                so3.score_vec(vec=rot_update, eps=rot_sigma)
            ).float()
            rot_scores.append(rot_score)
        rot_scores = (
            torch.stack(rot_scores).to(pos_lig.device).float()
        )  # there are the rotation targets

        rot_score_norms = []
        for rot_sigma in s_rotation:
            rot_score_norm = so3.score_norm(rot_sigma.unsqueeze(0).cpu())
            rot_score_norms.append(rot_score_norm)
        rot_score_norms = torch.stack(rot_score_norms, dim=0).to(pos_lig.device).float()

        # translate 0-com molecule back
        noise_com = torch.randn_like(com)  # these are the translation targets
        if self.hparams.global_translations:
            com_t = m_translation * com + s_translation * noise_com
        else:
            com_t = com

        pos_t = (
            pos_t_rotated + com_t[batch]
        )  # in prior state t->1 or t->T, the CoM should be drawn from N(0, I)

        return pos_t, com_t, noise_com, rot_scores, rot_score_norms

    def reverse_sampling(
        self,
        atom_types_ligand: Tensor,
        atom_types_pocket: Tensor,
        pos_ligand: Tensor,
        pos_pocket: Tensor,
        bond_edge_index: Tensor,
        bond_edge_attr: Tensor,
        batch_ligand: Tensor,
        batch_pocket: Tensor,
        theta1_dynamic: bool = True,
        phi1_dynamic: bool = True,
        phi2_dynamic: bool = True,
        stochastic_dynamic: bool = True,
        save_traj: bool = False,
        tqdm_verbose=False,
        casimir_component: bool = True,
        divergence_component: bool = False,
        full_rotation: bool = True,
        use_global_translations: bool = True,
    ):

        ###
        # General Idea:
        # As before in TrainerSphere, we first shift the ligand into the CoM of the pocket
        # Then we shift the ligand into the CoM of the ligand
        # We apply a random rotation from the prior of the rotation IGSO(3)
        # We apply a random translation from the prior of the translation N(0, I)
        # This goes into the model output because the model was trained like this predicting the translation and rotation scores
        # Updates:
        # We need to update the rotation matrices somehow. I am not sure how the rotations change? The SDE is assumed to be without drift,
        # i.e. a variance exploding SDE. The initial rotation vectors (axis-angle) need to be updated. Can be just perform a summation or do we require
        # the log/exp map to "translate" these vectors to change the rotation from R_t to R_{t-1}
        # If the above is resolved, we rotate the ligand (which is in its own 0-COM) and then we perform the translation update as before.
        # We repeat the procedure
        ###

        bs = int(batch_ligand.max() + 1)
        # We shift everything in pocket com
        pocket_com = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pos_pocket_0com = pos_pocket - pocket_com[batch_pocket]
        pos_ligand = pos_ligand - pocket_com[batch_ligand]

        # OHE
        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        atom_types_ligand = F.one_hot(
            atom_types_ligand.squeeze().long(), num_classes=self.num_atom_types
        ).float()

        edge_index_global_lig = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
        edge_index_global_lig = sort_edge_index(
            edge_index_global_lig, sort_by_row=False
        )

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )
        # not used. Is that supposed to be so?
        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )

        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        com = scatter_mean(pos_ligand, batch_ligand, dim=0)
        pos_lig_0com = pos_ligand - com[batch_ligand]

        # rotation from prior
        rotation_sigma = [self.sigmas[-1].squeeze()] * bs
        rot_updates = [so3.sample_vec(eps=s.item()) for s in rotation_sigma]
        rot_updates_torch = [
            torch.from_numpy(rot_update.squeeze()).float() for rot_update in rot_updates
        ]
        rot_updates_torch = torch.stack(rot_updates_torch).to(pos_ligand.device)
        rot_matrices = [
            axis_angle_to_matrix(torch.from_numpy(rot_update.squeeze())).float()
            for rot_update in rot_updates
        ]
        rot_matrices = torch.stack(rot_matrices).to(pos_ligand.device).float()
        rot_matrices_b = rot_matrices[batch_ligand]
        pos_lig_0com = torch.einsum("bij,bj->bi", rot_matrices_b, pos_lig_0com)

        # translation from prior
        translation = torch.randn_like(
            com
        )  # should be 0-com noise? currently model wasnt trained on 0-com noise...
        pos_t_ligand = pos_lig_0com + translation[batch_ligand]

        chain = reversed(range(self.hparams.timesteps))
        pos_traj = []
        trans_traj = []
        rot_traj = []
        batch_num_nodes = batch_ligand.bincount()
        dt = 1 / self.hparams.timesteps
        if save_traj:
            pos_traj.append(pos_t_ligand.cpu())
            rot_traj.append(rot_updates)
            trans_traj.append(translation.cpu())

        for t in tqdm(chain, total=self.hparams.timesteps) if tqdm_verbose else chain:

            t_ = torch.tensor([t] * bs, dtype=torch.long, device=pos_ligand.device)
            t_emb = t_.float() / self.hparams.timesteps
            t_emb = t_emb.clamp(min=self.hparams.eps_min)
            t_emb = t_emb.unsqueeze(dim=1)

            betas = self.betas[t_]

            # combine protein and ligand in one representation for translations
            (
                pos_joint,
                atom_types_joint,
                batch_full,
                mask_ligand,
                edge_index_global,
                edge_attr_global,
                _,
                edge_initial_interaction,
                batch_edge_global,
            ) = combine_protein_ligand_feats(
                pos_ligand=pos_t_ligand,
                pos_pocket=pos_pocket_0com,
                atom_types_ligand=atom_types_ligand,
                atom_types_pocket=atom_types_pocket,
                batch_ligand=batch_ligand,
                batch_pocket=batch_pocket,
                edge_attr_ligand=edge_attr_global_lig,
                num_bond_classes=5,
                cutoff_p=self.hparams.cutoff,
                cutoff_lp=self.hparams.cutoff,
            )

            out = self.model(
                x=atom_types_joint,
                pos=pos_joint,
                t=t_emb,
                edge_index=edge_index_global,
                edge_attr=edge_attr_global,
                edge_attr_initial_ohe=edge_initial_interaction,
                batch=batch_full,
                batch_ligand=batch_ligand,
                mask_ligand=mask_ligand,
                batch_edge=batch_edge_global,
            )

            scores_rot = out["rotation_score"].squeeze(-1)
            scores_translation = out["translation_score"].squeeze(-1)

            current_com = scatter_mean(pos_t_ligand, batch_ligand, dim=0)
            pos_t_ligand_0com = pos_t_ligand - current_com[batch_ligand]

            # rotation dynamics
            rot_sigmas_t = self.sigmas[t_]
            rot_sigmas_t_m1 = torch.where(
                t_ == 0, torch.zeros_like(t_), self.sigmas[t_ - 1]
            )
            dt_rot = rot_sigmas_t_m1 - rot_sigmas_t
            dt_rot = dt_rot.unsqueeze(-1)
            # dt_rot = dt_rot.fill_(1/100)
            # g_t = rot_sigmas_t * np.sqrt((2 * (np.log10(self.sigma_max) - np.log10(self.sigma_min))))
            # rotate_shift = (g_t ** 2) * scores_rot * dt_rot + g_t * torch.sqrt(dt_rot.abs()) * torch.randn_like(scores_rot)

            rot_g = rot_sigmas_t.unsqueeze(-1) * torch.sqrt(
                torch.tensor(2 * np.log10(self.sigma_max) - np.log10(self.sigma_min))
            )
            rot_z = torch.randn_like(scores_rot)
            rot_mean = scores_rot * dt_rot * rot_g**2
            rot_stoch = rot_g * torch.sqrt(dt_rot.abs()) * rot_z
            rotate_shift = rot_mean + rot_stoch

            rotate_shift = [
                axis_angle_to_matrix(torch.from_numpy(rot_update.squeeze())).float()
                for rot_update in rotate_shift.detach().cpu().numpy()
            ]
            rotate_shift = torch.stack(rotate_shift).to(pos_ligand.device).float()
            rot_matrices = torch.einsum("bij,bjk->bik", rot_matrices, rotate_shift)

            rot_matrices_b = rot_matrices[batch_ligand]
            pos_t_ligand_0com = torch.einsum(
                "bij,bj->bi", rot_matrices_b, pos_t_ligand_0com
            )

            # translation dynamics
            translation = translation + 0.5 * betas.unsqueeze(-1) * translation
            translation = translation + betas.unsqueeze(-1) * scores_translation
            translation = translation + betas.sqrt().unsqueeze(-1) * torch.randn_like(
                translation
            )

            pos_t_ligand = (
                pos_t_ligand_0com + translation[batch_ligand]
            )  # + current_com[batch_ligand]

            if save_traj:
                pos_traj.append(pos_t_ligand.cpu())
                rot_traj.append(rot_matrices.cpu())
                trans_traj.append(translation.cpu())

        return pos_t_ligand, pos_traj, rot_traj, trans_traj

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

        # translation
        target_translation = (
            -(1.0 / out_dict["translation_sigma"]) * out_dict["translation_target"]
        )
        pred_translation = out_dict["score_translation"]
        loss_translation = (
            (pred_translation.squeeze(-1) - target_translation).pow(2).sum(-1).mean(0)
        )

        # rotation
        target_rotation = out_dict["rotation_target"]
        pred_rotation = out_dict["score_rot"]
        rot_score_norm = out_dict["rotation_target_norm"]
        loss_rotation = (
            (
                (
                    (pred_rotation.squeeze(-1) - target_rotation)
                    / rot_score_norm.squeeze(-1)
                )
                ** 2
            )
            .sum(-1)
            .mean(0)
        )
        loss = loss_translation + loss_rotation

        self._log(
            loss_translation=loss_translation,
            loss_rotation=loss_rotation,
            loss=loss,
            batch_size=int((batch.batch.max() + 1)),
            stage=stage,
        )

        return loss

    def forward(self, batch: Batch, t: Tensor):

        atom_types_ligand: Tensor = batch.x
        atom_types_pocket: Tensor = batch.x_pocket
        pos_ligand: Tensor = batch.pos
        pos_pocket: Tensor = batch.pos_pocket
        batch_ligand: Tensor = batch.batch
        batch_pocket: Tensor = batch.pos_pocket_batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        # We shift everything in pocket com
        pocket_com = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pos_pocket_0com = pos_pocket - pocket_com[batch_pocket]
        pos_ligand = pos_ligand - pocket_com[batch_ligand]

        out = {}

        # OHE
        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        atom_types_ligand = F.one_hot(
            atom_types_ligand.squeeze().long(), num_classes=self.num_atom_types
        ).float()

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        if not hasattr(batch, "fc_edge_index"):
            edge_index_global_lig = (
                torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
            edge_index_global_lig = sort_edge_index(
                edge_index_global_lig, sort_by_row=False
            )
        else:
            edge_index_global_lig = batch.fc_edge_index

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )

        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        m_translation = self.mean_coeff[t].unsqueeze(-1).sqrt()
        s_translation = self.std_coeff[t].unsqueeze(-1)
        s_rotation = self.sigmas[t].unsqueeze(-1)

        pos_t, com_t, noise_com, rot_scores, rot_score_norms = self.forward_noising(
            pos_lig=pos_ligand,
            batch=batch_ligand,
            m_translation=m_translation,
            s_translation=s_translation,
            s_rotation=s_rotation,
        )

        # combine protein and ligand in one representation for translations
        (
            pos_joint,
            atom_types_joint,
            batch_full,
            mask_ligand,
            edge_index_global,
            edge_attr_global,
            _,
            edge_initial_interaction,
            batch_edge_global,
        ) = combine_protein_ligand_feats(
            pos_ligand=pos_t,
            pos_pocket=pos_pocket_0com,
            atom_types_ligand=atom_types_ligand,
            atom_types_pocket=atom_types_pocket,
            batch_ligand=batch_ligand,
            batch_pocket=batch_pocket,
            edge_attr_ligand=edge_attr_global_lig,
            num_bond_classes=5,
            cutoff_p=self.hparams.cutoff,
            cutoff_lp=self.hparams.cutoff,
        )

        out = self.model(
            x=atom_types_joint,
            pos=pos_joint,
            t=temb,
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            edge_attr_initial_ohe=edge_initial_interaction,
            batch=batch_full,
            batch_ligand=batch_ligand,
            mask_ligand=mask_ligand,
            batch_edge=batch_edge_global,
        )

        scores_rot = out["rotation_score"]
        scores_translation = out["translation_score"]

        out["score_translation"] = scores_translation
        out["translation_sigma"] = s_translation
        out["translation_target"] = noise_com

        out["score_rot"] = scores_rot
        out["rotation_sigma"] = s_rotation
        out["rotation_target"] = rot_scores
        out["rotation_target_norm"] = rot_score_norms

        out["edge_index_global_translations"] = edge_index_global
        out["edge_index_global_translations_rot"] = edge_index_global

        return out

    def _log(
        self,
        loss_translation,
        loss_rotation,
        loss,
        batch_size,
        stage,
    ):

        self.log(
            f"{stage}/loss_translation",
            loss_translation,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
        self.log(
            f"{stage}/loss_rotation",
            loss_rotation,
            on_step=True,
            batch_size=batch_size,
            prog_bar=(stage == "train"),
            sync_dist=self.hparams.gpus > 1 and stage == "val",
        )
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


def get_target_fisher_bridge(
    input: Dict[str, Tensor], m_t: Tensor, sigma_t: Tensor, kind: str
):
    """
    Get the target for the diffusion model.
    input is a dict with keys "pos_0", "pos_1", "pos_t" and "epsilon_local".

    kind can be "grad", "noise", or "x0"

    See reference: http://arxiv.org/pdf/2205.07680
    E.g. "grad" is  the loss in Algorithm 1
    """
    assert isinstance(input, dict), "Input must be a dictionary."

    assert kind in ["grad", "noise", "x0"]
    if kind == "grad":
        return m_t * (input["pos_1"] - input["pos_0"]) + sigma_t * input["epsilon"]
    elif kind == "noise":
        return input["epsilon"]
    elif kind == "x0":
        return input["pos_0"]
    else:
        raise ValueError(f"Unknown target kind: {kind}")


class TrainerFisherBridge(pl.LightningModule):
    """Implements the Fisher Score Model as Brownian Bridge between ground-truth ligand point cloud and a fully rotated ligand point cloud."""

    def __init__(
        self,
        hparams: dict,
    ):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.num_atom_types = hparams["atom_feat_dim"]
        self.model = DiffusionScoreModelFisher(
            atom_feat_dim=hparams["atom_feat_dim"],
            edge_feat_dim=hparams["edge_feat_dim"],
            hn_dim=(hparams["sdim"], hparams["vdim"]),
            edim=hparams["edim"],
            cutoff=hparams["cutoff"],
            num_layers=hparams["num_layers"],
            num_rbfs=hparams["num_rbfs"],
            use_cross_product=hparams["use_cross_product"],
            vector_aggr=hparams["vector_aggr"],
            global_translations=hparams["global_translations"],
            update_coords=hparams["update_coords"],
            update_edges=hparams["update_edges"],
        )

        self.global_translations = hparams["global_translations"]
        assert not self.global_translations
        self.max_var = 1.0
        to_torch = partial(torch.tensor, dtype=torch.float32)

        m_t = get_brownian_bridge_scheduler(T=self.hparams.timesteps + 1, kind="linear")
        m_tminus = np.append(0, m_t[:-1])
        variance_t = 2.0 * (m_t - m_t**2) * self.max_var
        sigma_t = np.sqrt(variance_t)
        variance_tminus = np.append(0.0, variance_t[:-1])
        variance_t_tminus = (
            variance_t - variance_tminus * ((1.0 - m_t) / (1.0 - m_tminus)) ** 2
        )
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t
        self.register_buffer("m_t", to_torch(m_t))
        self.register_buffer("m_tminus", to_torch(m_tminus))
        self.register_buffer("variance_t", to_torch(variance_t))
        self.register_buffer("sigma_t", to_torch(sigma_t))
        self.register_buffer("variance_tminus", to_torch(variance_tminus))
        self.register_buffer("variance_t_tminus", to_torch(variance_t_tminus))
        self.register_buffer("posterior_variance_t", to_torch(posterior_variance_t))

        # for global rotations use variance exploding scheduler SDE
        self.sigma_min = 0.01
        self.sigma_max = 2.0
        sigmas = get_sigma_scheduler(
            min_sigma=self.sigma_min,
            max_sigma=self.sigma_max,
            n=self.hparams.timesteps + 1,
        )
        self.register_buffer("sigmas", torch.from_numpy(sigmas).float())

        self.regression_target = hparams.get("regression_target", "x0")
        assert self.regression_target in ["grad", "noise", "x0"], (
            f"Invalid regression target: {self.regression_target}. "
            "Must be one of 'grad', 'noise', or 'x0'."
        )
        print("Performing regression on:", self.regression_target)

    def forward_noising(
        self,
        pos_lig: Tensor,
        batch: Tensor,
        s_rotation: Tensor,
        m_t: Tensor,
        sigma_t: Tensor,
    ) -> Tuple[Tensor, ...]:

        # get rotation at prior with t=T
        rot_updates = [
            so3.sample_vec(eps=s.item()) for s in s_rotation.cpu().numpy().squeeze()
        ]
        rot_matrices = [
            axis_angle_to_matrix(torch.from_numpy(rot_update.squeeze())).float()
            for rot_update in rot_updates
        ]
        rot_matrices = torch.stack(rot_matrices).to(pos_lig.device)
        R_t = rot_matrices[batch]
        pos_lig_rotated = torch.einsum("bij,bj->bi", R_t, pos_lig)

        pos_prior_local = torch.randn_like(pos_lig)
        pos_t_bridged = m_t * pos_lig_rotated + (1.0 - m_t) * pos_lig
        pos_t_bridged = pos_t_bridged + sigma_t * pos_prior_local

        outputs = {
            "pos_t": pos_t_bridged,
            "pos_0": pos_lig,
            "pos_1": pos_lig_rotated,
            "epsilon": pos_prior_local,
        }
        return outputs

    def reverse_sampling(
        self,
        atom_types_ligand: Tensor,
        atom_types_pocket: Tensor,
        pos_ligand: Tensor,
        pos_pocket: Tensor,
        bond_edge_index: Tensor,
        bond_edge_attr: Tensor,
        batch_ligand: Tensor,
        batch_pocket: Tensor,
        save_traj: bool = False,
        tqdm_verbose: bool = False,
    ):

        bs = int(batch_ligand.max() + 1)
        # We shift everything in pocket com
        pocket_com = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pos_pocket_0com = pos_pocket - pocket_com[batch_pocket]
        pos_ligand = pos_ligand - pocket_com[batch_ligand]

        # OHE
        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        atom_types_ligand = F.one_hot(
            atom_types_ligand.squeeze().long(), num_classes=self.num_atom_types
        ).float()

        edge_index_global_lig = (
            torch.eq(batch_ligand.unsqueeze(0), batch_ligand.unsqueeze(-1))
            .int()
            .fill_diagonal_(0)
        )
        edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
        edge_index_global_lig = sort_edge_index(
            edge_index_global_lig, sort_by_row=False
        )

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )
        # not used. Is that supposed to be so?
        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )

        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        # rotation from prior
        rotation_sigma = [self.sigmas[-1].squeeze()] * bs
        rot_updates = [so3.sample_vec(eps=s.item()) for s in rotation_sigma]
        rot_updates_torch = [
            torch.from_numpy(rot_update.squeeze()).float() for rot_update in rot_updates
        ]
        rot_updates_torch = torch.stack(rot_updates_torch).to(pos_ligand.device)
        rot_matrices = [
            axis_angle_to_matrix(torch.from_numpy(rot_update.squeeze())).float()
            for rot_update in rot_updates
        ]
        rot_matrices = torch.stack(rot_matrices).to(pos_ligand.device).float()
        rot_matrices_b = rot_matrices[batch_ligand]
        pos_lig_T = torch.einsum("bij,bj->bi", rot_matrices_b, pos_ligand)
        pos_t_ligand = pos_lig_T.clone()

        chain = reversed(range(1, self.hparams.timesteps + 1))
        pos_traj = []
        prediction_traj = []
        coeffs_traj = []
        if save_traj:
            pos_traj.append(pos_t_ligand.cpu())

        for t in tqdm(chain, total=self.hparams.timesteps) if tqdm_verbose else chain:

            t_ = torch.tensor([t] * bs, dtype=torch.long, device=pos_ligand.device)
            n_t_ = t_ - 1
            t_emb = t_.float() / self.hparams.timesteps
            t_emb = t_emb.clamp(min=self.hparams.eps_min)
            t_emb = t_emb.unsqueeze(dim=1)

            # combine protein and ligand in one representation for translations
            (
                pos_joint,
                atom_types_joint,
                batch_full,
                mask_ligand,
                edge_index_global,
                edge_attr_global,
                _,
                edge_initial_interaction,
                batch_edge_global,
            ) = combine_protein_ligand_feats(
                pos_ligand=pos_t_ligand,
                pos_pocket=pos_pocket_0com,
                atom_types_ligand=atom_types_ligand,
                atom_types_pocket=atom_types_pocket,
                batch_ligand=batch_ligand,
                batch_pocket=batch_pocket,
                edge_attr_ligand=edge_attr_global_lig,
                num_bond_classes=5,
                cutoff_p=self.hparams.cutoff,
                cutoff_lp=self.hparams.cutoff,
            )

            out = self.model(
                x=atom_types_joint,
                pos=pos_joint,
                t=t_emb,
                edge_index=edge_index_global,
                edge_attr=edge_attr_global,
                edge_attr_initial_ohe=edge_initial_interaction,
                batch=batch_full,
                batch_ligand=batch_ligand,
                mask_ligand=mask_ligand,
                batch_edge=batch_edge_global,
            )

            prediction = out["local_translation_score"]
            assert (
                self.regression_target == "x0"
            ), "Sampling procedure currently only implemented for `x0` parameterization"
            m_t = self.m_t[t_]
            m_nt = self.m_t[n_t_]
            var_t = self.variance_t[t_]
            var_nt = self.variance_t[n_t_]
            sigma2_t = (
                (var_t - var_nt * (1.0 - m_t) ** 2 / (1.0 - m_nt) ** 2) * var_nt / var_t
            )
            sigma_t = torch.sqrt(sigma2_t)
            noise = torch.randn_like(pos_t_ligand)
            x_tminus_mean = (
                (1.0 - m_nt) * prediction
                + m_nt * pos_lig_T
                + torch.sqrt((var_nt - sigma2_t) / var_t)
                * (pos_t_ligand - (1.0 - m_t) * prediction - m_t * pos_lig_T)
            )

            pos_t_ligand = x_tminus_mean + sigma_t * noise

            if save_traj:
                pos_traj.append(pos_t_ligand.cpu())
                prediction_traj.append(prediction.cpu())

        return pos_t_ligand, pos_traj, prediction_traj, coeffs_traj

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

        loss = (out_dict["target"] - out_dict["prediction"]).pow(2).sum(-1)
        loss = scatter_mean(
            loss, batch.batch, dim=0, dim_size=(batch.batch.max().item() + 1)
        )
        loss = torch.mean(loss, dim=0)

        self._log(
            loss=loss,
            batch_size=int((batch.batch.max() + 1)),
            stage=stage,
        )

        return loss

    def forward(self, batch: Batch, t: Tensor):

        atom_types_ligand: Tensor = batch.x
        atom_types_pocket: Tensor = batch.x_pocket
        pos_ligand: Tensor = batch.pos
        pos_pocket: Tensor = batch.pos_pocket
        batch_ligand: Tensor = batch.batch
        batch_pocket: Tensor = batch.pos_pocket_batch
        bond_edge_index = batch.edge_index
        bond_edge_attr = batch.edge_attr
        bond_edge_index, bond_edge_attr = sort_edge_index(
            edge_index=bond_edge_index, edge_attr=bond_edge_attr, sort_by_row=False
        )

        # We shift everything in pocket com
        pocket_com = scatter_mean(pos_pocket, batch_pocket, dim=0)
        pos_pocket_0com = pos_pocket - pocket_com[batch_pocket]
        pos_ligand = pos_ligand - pocket_com[batch_ligand]

        out = {}

        # OHE
        atom_types_pocket = F.one_hot(
            atom_types_pocket.squeeze().long(), num_classes=self.num_atom_types
        ).float()
        atom_types_ligand = F.one_hot(
            atom_types_ligand.squeeze().long(), num_classes=self.num_atom_types
        ).float()

        # TIME EMBEDDING
        temb = t.float() / self.hparams.timesteps
        temb = temb.clamp(min=self.hparams.eps_min)
        temb = temb.unsqueeze(dim=1)

        if not hasattr(batch, "fc_edge_index"):
            edge_index_global_lig = (
                torch.eq(batch.batch.unsqueeze(0), batch.batch.unsqueeze(-1))
                .int()
                .fill_diagonal_(0)
            )
            edge_index_global_lig, _ = dense_to_sparse(edge_index_global_lig)
            edge_index_global_lig = sort_edge_index(
                edge_index_global_lig, sort_by_row=False
            )
        else:
            edge_index_global_lig = batch.fc_edge_index

        edge_index_global_lig, edge_attr_global_lig = coalesce_edges(
            edge_index=edge_index_global_lig,
            bond_edge_index=bond_edge_index,
            bond_edge_attr=bond_edge_attr,
            n=batch_ligand.size(0),
        )
        edge_index_global_lig, edge_attr_global_lig = sort_edge_index(
            edge_index=edge_index_global_lig,
            edge_attr=edge_attr_global_lig,
            sort_by_row=False,
        )

        edge_attr_global_lig = F.one_hot(
            edge_attr_global_lig.long(), num_classes=5
        ).float()

        batch_size = len(t)
        m_t = self.m_t[t][batch_ligand].unsqueeze(-1)
        sigma_t = self.sigma_t[t][batch_ligand].unsqueeze(-1)
        t_prior = self.hparams.timesteps
        t_prior = torch.tensor(
            [t_prior] * batch_size, dtype=torch.long, device=pos_ligand.device
        )
        s_rotation = self.sigmas[t_prior].unsqueeze(-1)
        assert s_rotation.max().item() > self.sigma_max - 0.05

        data = self.forward_noising(
            pos_lig=pos_ligand,
            batch=batch_ligand,
            m_t=m_t,
            sigma_t=sigma_t,
            s_rotation=s_rotation,
        )

        # combine protein and ligand in one representation for translations
        (
            pos_joint,
            atom_types_joint,
            batch_full,
            mask_ligand,
            edge_index_global,
            edge_attr_global,
            _,
            edge_initial_interaction,
            batch_edge_global,
        ) = combine_protein_ligand_feats(
            pos_ligand=data["pos_t"],
            pos_pocket=pos_pocket_0com,
            atom_types_ligand=atom_types_ligand,
            atom_types_pocket=atom_types_pocket,
            batch_ligand=batch_ligand,
            batch_pocket=batch_pocket,
            edge_attr_ligand=edge_attr_global_lig,
            num_bond_classes=5,
            cutoff_p=self.hparams.cutoff,
            cutoff_lp=self.hparams.cutoff,
        )

        out = self.model(
            x=atom_types_joint,
            pos=pos_joint,
            t=temb,
            edge_index=edge_index_global,
            edge_attr=edge_attr_global,
            edge_attr_initial_ohe=edge_initial_interaction,
            batch=batch_full,
            batch_ligand=batch_ligand,
            mask_ligand=mask_ligand,
            batch_edge=batch_edge_global,
        )

        local_translation_score = out["local_translation_score"]

        out["prediction"] = local_translation_score
        out["sigma_t"] = sigma_t
        out["m_t"] = m_t

        out["target"] = get_target_fisher_bridge(
            input=data,
            m_t=m_t,
            sigma_t=sigma_t,
            kind=self.regression_target,
        )
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
