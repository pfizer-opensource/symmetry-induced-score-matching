import argparse
from argparse import ArgumentParser
import os

from sism.utils import LoadFromCheckpoint, LoadFromFile


def add_arguments(parser):
    """Helper function to fill the parser object.

    Args:
        parser: Parser object
    Returns:
        parser: Updated parser object
    """

    # Load yaml file
    parser.add_argument(
        "--conf", "-c", type=open, action=LoadFromFile, help="Configuration yaml file"
    )  # keep first

    # Load from checkpoint
    parser.add_argument("--load-ckpt", default="", type=str)

    # DATA and FILES
    parser.add_argument(
        "-s",
        "--save-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to the dataset root folder",
    )

    # LEARNING
    parser.add_argument("-b", "--batch-size", default=128, type=int)
    parser.add_argument("--grad-clip-val", default=10.0, type=float)
    parser.add_argument("--amsgrad", default=False, action="store_true")
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--lr-frequency", default=5, type=int)
    parser.add_argument("--lr-patience", default=20, type=int)
    parser.add_argument("--lr-cooldown", default=5, type=int)
    parser.add_argument("--lr-factor", default=0.75, type=float)

    # MODEL
    parser.add_argument("--atom-feat-dim", default=16, type=int)
    parser.add_argument("--edge-feat-dim", default=5, type=int)
    parser.add_argument("--sdim", default=128, type=int)
    parser.add_argument("--vdim", default=32, type=int)
    parser.add_argument("--edim", default=16, type=int)
    parser.add_argument("--vector-aggr", default="mean", type=str)
    parser.add_argument("--use-cross-product", default=False, action="store_true")
    parser.add_argument("--num-layers", default=5, type=int)
    parser.add_argument("--num-rbfs", default=20, type=int)
    parser.add_argument("--cutoff", default=5.0, type=float)
    parser.add_argument("--global-translations", default=False, action="store_true")
    parser.add_argument("--update-coords", default=False, action="store_true")
    parser.add_argument("--update-edges", default=False, action="store_true")
    parser.add_argument("--use-pcs", default=False, action="store_true")

    # DIFFUSION NET
    parser.add_argument("--timesteps", default=100, type=int)
    parser.add_argument(
        "--noise-schedule",
        default="cosine",
        type=str,
        choices=["linear-time", "cosine"],
    )
    parser.add_argument("--eps-min", default=1e-3, type=float)

    # GENERAL
    parser.add_argument("-i", "--id", type=int, default=0)
    parser.add_argument("-g", "--gpus", default=0, type=int)
    parser.add_argument("-e", "--num-epochs", default=200, type=int)
    parser.add_argument("--eval-freq", default=1.0, type=float)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--detect-anomaly", default=False, action="store_true")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--accum-batch", default=1, type=int)
    parser.add_argument("--weight-decay", default=1e-6, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--remove-hs", default=True, action="store_true")
    parser.add_argument(
        "--joint-property-prediction", default=True, action="store_true"
    )
    parser.add_argument("--regression-property", default="docking_score")
    parser.add_argument("--property-training", default=False, action="store_true")
    parser.add_argument("--dataset", default="crossdocked")
    parser.add_argument(
        "--model", default="gsm", type=str, choices=["gsm", "rsgm", "fisher_bridge"]
    )
    parser.add_argument(
        "--regression-target",
        default="grad",
        choices=["grad", "noise", "x0"],
        help="Regression target for the fisher_bridge model. ",
    )

    return parser
