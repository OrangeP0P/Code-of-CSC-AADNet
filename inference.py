from __future__ import division
from __future__ import print_function
import os
import argparse
import csv
import numpy as np
from typing import List
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from utils import _channel_normalize, _moving_average

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current device is", device)


def makePath(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def _mean_covariance(trials: np.ndarray) -> np.ndarray:
    """
    trials: (N, C, T)
    """
    N, C, T = trials.shape
    cov_sum = np.zeros((C, C), dtype=np.float64)
    for i in range(N):
        Xi = trials[i]  # (C, T)
        cov_sum += (Xi @ Xi.T) / float(T)
    return cov_sum / float(N)

def _inv_sqrtm_psd(C: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w, U = np.linalg.eigh(C)
    w = np.clip(w, a_min=eps, a_max=None)
    inv_sqrt = (U * (w ** -0.5)) @ U.T
    return inv_sqrt

def _sub_calibration(subject_array: np.ndarray):
    X = subject_array
    transposed = False
    if X.shape[1] > X.shape[2]:
        X = np.transpose(X, (0, 2, 1))
        transposed = True

    Cbar = _mean_covariance(X.astype(np.float64))
    P = _inv_sqrtm_psd(Cbar, eps=1e-10)

    X_aligned = np.einsum('ij,njk->nik', P, X, optimize=True)

    if transposed:
        X_aligned = np.transpose(X_aligned, (0, 2, 1))
    return X_aligned, P

def getData(args, sid: int):
    data_path = os.path.join(args.data_path, f"S{sid}.npy")
    onedata = np.load(data_path)
    onedata = onedata.transpose(0, 2, 1)

    onedata = _channel_normalize(onedata, axis=-1, method="zscore")

    ma_window = getattr(args, "ma_window", 3)
    onedata = _moving_average(onedata, window=ma_window, axis=-1, mode="reflect")

    onedata_ea, _ = _sub_calibration(onedata)

    return onedata_ea


class CustomDatasets(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        return data


# ====================== Ensemble ======================
def load_ensemble_models(model_dir: str) -> List[nn.Module]:
    model_dir_path = Path(model_dir)
    if not model_dir_path.is_dir():
        raise FileNotFoundError(f"model_dir '{model_dir}' is not a directory or does not exist.")

    model_paths = sorted(
        list(model_dir_path.glob("*.pt")) + list(model_dir_path.glob("*.pth"))
    )
    if len(model_paths) == 0:
        raise FileNotFoundError(f"No .pt/.pth files found in model_dir '{model_dir}'")

    print("Ensemble models:")
    for p in model_paths:
        print("  ", p)

    models: List[nn.Module] = []
    for ckpt_path in model_paths:
        model = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        models.append(model)

    print(f"Loaded {len(models)} models for ensemble.")
    return models


def run_inference_ensemble(args, testdata: np.ndarray):
    test_loader = DataLoader(
        dataset=CustomDatasets(testdata),
        batch_size=args.batch_size,
        drop_last=False,
    )

    models = load_ensemble_models(args.model_dir)

    all_preds = []

    with torch.no_grad():
        for batch_data in test_loader:
            inputs = batch_data.to(device).float()

            logits_sum = None
            for m in models:
                out = m(inputs)
                if logits_sum is None:
                    logits_sum = out
                else:
                    logits_sum = logits_sum + out

            logits_avg = logits_sum / len(models)

            pred = torch.argmax(logits_avg, dim=1)
            all_preds.append(pred.detach().cpu().numpy())

    pre_labels = np.concatenate(all_preds, axis=0)
    return pre_labels


def parse_args():
    p = argparse.ArgumentParser(description="EEG Inference script (ensemble + same preprocessing as training)")

    p.add_argument("--model_name",
                   type=str,
                   default="CSC_AADNet",
                   help="Model name (unused when loading full models via torch.load)")

    p.add_argument("--model_path",
                   type=str,
                   default="",
                   help="(Optional) Single model checkpoint path (.pt/.pth), not used in ensemble mode")

    p.add_argument("--model_dir",
                   type=str,
                   default="select_model_path/",
                   help="Directory containing fold checkpoints (.pt/.pth) for ensemble")

    p.add_argument("--data_path",
                   type=str,
                   default="your_path/Testset_audio_visual/preprocessed/data/",
                   help="EEG source dir of test .npy files")

    p.add_argument("--out_csv",
                   type=str,
                   default="./result/cross_session",
                   help="Output CSV path prefix (id,predictions)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    options = {'MM-AAD': [10, 32, 20, 128]}
    args.subject_number = options['MM-AAD'][0]
    args.eeg_channel = options['MM-AAD'][1]
    args.trail_number = options['MM-AAD'][2]
    args.fs = options['MM-AAD'][3]

    args.batch_size = 64
    args.num_workers = 4
    args.num_classes = 2
    args.device = "cuda"

    args.ma_window = 3

    sub_ids = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

    for sid in sub_ids:
        print(f"\n==== Subject {sid} ====")
        testdata = getData(args, sid)
        pre_labels = run_inference_ensemble(args, testdata)

        out_path = f'{args.out_csv}_{sid}.csv'
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label"])
            for i, y in enumerate(pre_labels):
                writer.writerow([i, int(y)])
        print(f"Write predictions to: {out_path}")
