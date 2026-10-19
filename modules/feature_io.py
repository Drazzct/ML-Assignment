"""
Module: Feature IO
Shared save/load utilities for dense feature arrays (.npy and .h5) and sparse
matrices (.npz).

Both `TraditionalFeatureExtractor` and `ModernFeatureExtractor` use these
functions so that their outputs are interchangeable — any saved feature
file can be loaded and fed directly into the shared `Classifier`.
"""

import os
import numpy as np
import h5py
from scipy import sparse

from modules.config import FEATURES_DIR


def save_features(embeddings, filename, features_dir=FEATURES_DIR, save_format="npy"):
    """
    Saves a feature array to disk.

    Implementation Guide:
    1. `os.makedirs(features_dir, exist_ok=True)`.
    2. Build the full path: `path = os.path.join(features_dir, filename)`.
    3. Save sparse matrices as `.npz` when `save_format == 'npz'`.
    4. If `save_format` is 'npy' or 'h5', convert sparse matrices to dense.
    5. Save the dense array in the selected format.
       ```
         with h5py.File(path + '.h5', 'w') as f:
             f.create_dataset('embeddings', data=embeddings)
         ```

    Args:
        embeddings (np.ndarray | scipy.sparse matrix): Feature array of shape (N, D).
        filename (str): Base filename (without extension).
        features_dir (str): Directory to save into.
        save_format (str): 'npy', 'h5', or 'npz'.
    """
    os.makedirs(features_dir, exist_ok=True)
    path = os.path.join(features_dir, filename)

    if save_format == "npz":
        if not sparse.issparse(embeddings):
            raise TypeError("The 'npz' format requires a scipy sparse matrix.")
        sparse.save_npz(path + ".npz", embeddings)
        saved_path = path + ".npz"
    elif save_format == "npy":
        array = embeddings.toarray() if sparse.issparse(embeddings) else embeddings
        np.save(path + ".npy", array, allow_pickle=False)
        saved_path = path + ".npy"
    elif save_format == "h5":
        array = embeddings.toarray() if sparse.issparse(embeddings) else embeddings
        with h5py.File(path + ".h5", "w") as feature_file:
            feature_file.create_dataset("embeddings", data=array)
        saved_path = path + ".h5"
    else:
        raise ValueError("save_format must be 'npy', 'h5', or 'npz'.")

    print(f"Saved features to {saved_path} with shape {embeddings.shape}.")


def load_features(filename, features_dir=FEATURES_DIR, save_format="npy"):
    """
    Loads a feature array from disk.

    Implementation Guide:
    1. Build the full path with the correct extension.
    2. If `save_format == 'npy'`: `return np.load(path, allow_pickle=False)`.
    3. If `save_format == 'npz'`: `return scipy.sparse.load_npz(path)`.
    4. If `save_format == 'h5'`:
       ```
       with h5py.File(path, 'r') as f:
           return f['embeddings'][:]
       ```

    Args:
        filename (str): Base filename (without extension).
        features_dir (str): Directory to load from.
        save_format (str): 'npy', 'h5', or 'npz'.
    Returns:
        np.ndarray | scipy.sparse matrix: Loaded features of shape (N, D).
    """
    path = os.path.join(features_dir, filename)
    if save_format == "npy":
        return np.load(path + ".npy", allow_pickle=False)
    if save_format == "npz":
        return sparse.load_npz(path + ".npz")
    if save_format == "h5":
        with h5py.File(path + ".h5", "r") as feature_file:
            return feature_file["embeddings"][:]
    raise ValueError("save_format must be 'npy', 'h5', or 'npz'.")
