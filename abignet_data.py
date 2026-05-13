import random
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

def set_seed(seed=101):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def patch_number(path):
    name = path.stem
    try:
        return int(name.split("_")[-1])
    except Exception:
        return 0

def load_image_features(img_dir):
    img_dir = Path(img_dir)
    files = sorted(
        img_dir.glob("combined_features_patch_*.npy"),
        key=patch_number,
    )
    if len(files) == 0:
        raise FileNotFoundError(f"No combined_features_patch_*.npy files found in {img_dir}")
    feats = []
    for file_path in files:
        x = np.load(file_path).astype(np.float32).reshape(-1)
        feats.append(x)
    feats = np.vstack(feats)
    return feats

def make_graph_from_folder(img_dir, label):
    # Load one image folder as one graph.
    x = load_image_features(img_dir)
    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor([int(label)], dtype=torch.long),
    )
    return data

def parse_list(text):
    return [item.strip() for item in text.split(",") if item.strip() != ""]

def parse_int_list(text):
    return [int(item.strip()) for item in text.split(",") if item.strip() != ""]

def make_dataset_from_folders(folder_text, label_text):
    folders = parse_list(folder_text)
    labels = parse_int_list(label_text)

    if len(folders) != len(labels):
        raise ValueError("Number of folders and labels must match.")
    data_list = []

    for folder, label in zip(folders, labels):
        data = make_graph_from_folder(folder, label)
        data_list.append(data)
        print(f"Loaded {folder}")
        print(f"  nodes: {data.x.shape[0]}, features: {data.x.shape[1]}, label: {label}")

    return data_list

def fit_feature_scaler(data_list):
    # Fit scaler on training nodes only.
    all_features = torch.cat([data.x for data in data_list], dim=0).cpu().numpy()
    scaler = StandardScaler()
    scaler.fit(all_features)
    return scaler

def apply_feature_scaler(data_list, scaler):
    for data in data_list:
        x_np = data.x.cpu().numpy()
        data.x = torch.tensor(scaler.transform(x_np), dtype=torch.float32)
    return data_list

def prepare_train_val_test_data(
    train_dirs, train_labels, 
    val_dirs, val_labels,
    test_dirs, test_labels,
):
    train_data = make_dataset_from_folders(train_dirs, train_labels)
    val_data = make_dataset_from_folders(val_dirs, val_labels)
    test_data = make_dataset_from_folders(test_dirs, test_labels)
    scaler = fit_feature_scaler(train_data)
    train_data = apply_feature_scaler(train_data, scaler)
    val_data = apply_feature_scaler(val_data, scaler)
    test_data = apply_feature_scaler(test_data, scaler)

    print("Final graph objects")
    print(f"  Train graphs: {len(train_data)}")
    print(f"  Val graphs  : {len(val_data)}")
    print(f"  Test graphs : {len(test_data)}")
    print(f"  Input dim   : {train_data[0].x.shape[1]}")

    return train_data, val_data, test_data, scaler