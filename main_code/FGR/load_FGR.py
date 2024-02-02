import os
import sys
from typing import List

import numpy as np
import pandas as pd
import rootutils
import torch
from rdkit.Chem.rdmolfiles import MolFromSmarts
from tokenizers import Tokenizer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from main_code.FGR.src.data.components.utils import (
    smiles2vector_fg,
    smiles2vector_mfg,
    standardize_smiles,
)
from main_code.FGR.src.models.components.autoencoder import FGRPretrainModel
from main_code.FGR.src.models.fgr_module import FGRPretrainLitModule

fgroups = pd.read_parquet("/home/rajeeva/Project/yeast_growth_pred/main_code/FGR/fg.parquet")[
    "SMARTS"
].tolist()  # Get functional groups
fgroups_list = [MolFromSmarts(x) for x in fgroups]  # Convert to RDKit Mol
tokenizer = Tokenizer.from_file(
    os.path.join(
        "/home/rajeeva/Project/yeast_growth_pred/main_code/FGR/tokenizers",
        f"BPE_pubchem_{500}.json",
    )
)


def get_representation(
    smiles: List[str],
    method: str,
    fgroups_list: List[MolFromSmarts],
    tokenizer: Tokenizer,
) -> np.ndarray:
    smiles = [standardize_smiles(smi) for smi in smiles]  # Standardize smiles
    if method == "FG":
        x = np.stack([smiles2vector_fg(x, fgroups_list) for x in smiles])
    elif method == "MFG":
        x = np.stack([smiles2vector_mfg(x, tokenizer) for x in smiles])
    elif method == "FGR":
        f_g = np.stack([smiles2vector_fg(x, fgroups_list) for x in smiles])
        mfg = np.stack([smiles2vector_mfg(x, tokenizer) for x in smiles])
        x = np.concatenate((f_g, mfg), axis=1)  # Concatenate both vectors
    else:
        raise ValueError("Method not supported")  # Raise error if method not supported
    return x


def get_fgr_module(ckpt_path: str):
    """Get the encoder of the pretrained FGR model.

    Args:
        ckpt_path (str): _description_
    """

    model = FGRPretrainLitModule.load_from_checkpoint(ckpt_path)
    model.eval()
    return model


def get_fgr_model(ckpt_path: str):
    """Get the encoder of the pretrained FGR model.

    Args:
        ckpt_path (str): _description_
    """

    model = torch.load(ckpt_path)
    # model = model.train()
    return model


if __name__ == "__main__":
    import pickle

    model = get_fgr_module("/home/rajeeva/Project/outputs/epoch_000_val_0.8505.ckpt")
    print(model)

    x = get_representation(
        ["CC(C)(C)NCC(O)c1cc(Cl)c(N)c(c1)C(F)(F)F", "CCN", "CCF"], "FGR", fgroups_list, tokenizer
    )

    x = torch.tensor(x, dtype=torch.float32, device="cuda:0")
    z_d = model(x)
    print(z_d[0])

    print(model.hparams)

    # with open("/home/rajeeva/Project/outputs/fgr_model_pretrain.pt", "wb") as f:
    # torch.save(model.net, "/home/rajeeva/Project/outputs/fgr_model_pretrain.pt")
    # torch.save(model.state_dict(), "/home/rajeeva/Project/outputs/fgr_model.pt")
