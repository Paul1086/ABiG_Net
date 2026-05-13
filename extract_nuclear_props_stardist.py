# Table 3 of our paper
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import tifffile as tiff
from PIL import Image, ImageFile
from skimage.measure import regionprops, regionprops_table
from stardist.models import StarDist2D
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops


ImageFile.LOAD_TRUNCATED_IMAGES = True

# Our patch images are mostly PNG or TIFF files.
IMAGE_EXTENSIONS = {".png", ".tif", ".tiff"}

# We did not use solidity and orientation in the final feature set.
PROPS_TO_MEASURE = [
    "label", "centroid",
    "area", "major_axis_length", "minor_axis_length", 
    "perimeter", "eccentricity", "equivalent_diameter", "mean_intensity",
    #solidity, orientation
]

PROPS_COLUMNS = [
    "label", "centroid-0", "centroid-1",
    "area", "major_axis_length", "minor_axis_length",
    "perimeter", "eccentricity", "equivalent_diameter",  "mean_intensity",
    "contrast", "energy", "correlation", "homogeneity", "ASM",
]

# Read image as RGB. If PIL fails, use OpenCV as a fallback reader
def read_rgb(path):
    try:
        if path.stat().st_size == 0:
            return None

        with Image.open(path) as im:
            im.load()
            img = np.asarray(im)

    except Exception:
        return read_rgb_cv2(path)

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)

    elif img.ndim == 3 and img.shape[2] == 4:
        rgb = img[:, :, :3].astype(np.float32)
        alpha = img[:, :, 3:4].astype(np.float32) / 255.0
        img = alpha * rgb + (1.0 - alpha) * 255.0

    if img.ndim != 3 or img.shape[2] != 3:
        return None

    return img.astype(np.uint8)


# We made a backup image reader for files that PIL cannot open

def read_rgb_cv2(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None

    if img is None:
        return None

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)

    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.uint8)

def get_tissue_mask(img, threshold):
    return np.any(img > threshold, axis=2)


def mask_bbox(mask):
    yy, xx = np.where(mask)

    if len(yy) == 0:
        return None

    return yy.min(), yy.max() + 1, xx.min(), xx.max() + 1


def normalize_patch(img, mask, low=1.0, high=99.8):
    img = img.astype(np.float32)
    out = np.zeros_like(img, dtype=np.float32)

    for ch in range(3):
        vals = img[:, :, ch][mask]

        if vals.size == 0:
            continue

        p_low, p_high = np.percentile(vals, (low, high))

        if p_high <= p_low:
            p_high = p_low + 1.0

        out[:, :, ch] = np.clip((img[:, :, ch] - p_low) / (p_high - p_low), 0, 1)

    out[~mask] = 0

    return out

def rgb_to_gray_uint8(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.uint8)
def glcm_features_for_region(gray_img, region_mask):
    vals = gray_img[region_mask]
    if vals.size < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    yy, xx = np.where(region_mask)
    if len(yy) < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1

    patch = gray_img[y0:y1, x0:x1].copy()
    mask = region_mask[y0:y1, x0:x1]

    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    patch[~mask] = 0

    # Quantize grayscale values before GLCM calculation
    patch = (patch.astype(np.float32) / 32.0).astype(np.uint8)
    patch = np.clip(patch, 0, 7)

    glcm = graycomatrix(
        patch,
        distances=[1], angles=[0], levels=8, symmetric=True, normed=True,
    )
    contrast = graycoprops(glcm, "contrast")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    asm_value = graycoprops(glcm, "ASM")[0, 0]

    values = [contrast, energy, correlation, homogeneity, asm_value]
    values = [0.0 if not np.isfinite(v) else float(v) for v in values]

    return values

def compute_texture_features(labels, gray_img, label_list):
    texture_features = np.zeros((len(label_list), 5), dtype=np.float32)

    for i, label_id in enumerate(label_list):
        region_mask = labels == label_id
        texture_features[i, :] = glcm_features_for_region(gray_img, region_mask)

    return texture_features


def get_img_folders(root):
    root = Path(root)

    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("img_")
    )


def save_column_file(out_dir):
    col_path = Path(out_dir) / "props_cols.json"

    if col_path.exists():
        return

    with open(col_path, "w", encoding="utf-8") as f:
        json.dump(PROPS_COLUMNS, f, indent=2)


def extract_one_patch(
    img_path, out_dir, model,
    prob_thresh, nms_thresh,
    min_tissue_frac, nonblack_thresh, crop_to_tissue,
    overwrite,
):
    img_path = Path(img_path)
    out_dir = Path(out_dir)

    label_path = out_dir / f"labels_{img_path.stem}.tiff"
    prop_path = out_dir / f"props_{img_path.stem}.npy"

    if label_path.exists() and prop_path.exists() and not overwrite:
        return "existing"

    img = read_rgb(img_path)

    if img is None:
        return "bad_file"

    tissue_mask = get_tissue_mask(img, nonblack_thresh)

    if tissue_mask.mean() < min_tissue_frac:
        return "low_tissue"

    # Run StarDist only on the tissue bounding box.
    if crop_to_tissue:
        box = mask_bbox(tissue_mask)

        if box is None:
            return "low_tissue"

        y0, y1, x0, x1 = box
        img_small = img[y0:y1, x0:x1]
        mask_small = tissue_mask[y0:y1, x0:x1]
        model_input = normalize_patch(img_small, mask_small)

    else:
        y0, y1 = 0, img.shape[0]
        x0, x1 = 0, img.shape[1]
        model_input = normalize_patch(img, tissue_mask)

    labels_small, _ = model.predict_instances(
        model_input,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )

    labels = np.zeros(img.shape[:2], dtype=np.uint16)
    labels[y0:y1, x0:x1] = labels_small.astype(np.uint16)
    labels[~tissue_mask] = 0

    tiff.imwrite(str(label_path), labels, compression="zlib")

    gray_img = rgb_to_gray_uint8(img)

    props = regionprops_table(
        labels,
        intensity_image=gray_img,
        properties=PROPS_TO_MEASURE,
    )

    if len(props["label"]) == 0:
        props_array = np.zeros((0, len(PROPS_COLUMNS)), dtype=np.float32)
        np.save(prop_path, props_array)
        return "processed"

    base_props = np.column_stack([
        props["label"],
        props["centroid-0"],
        props["centroid-1"],
        props["area"],
        props["major_axis_length"],
        props["minor_axis_length"],
        props["perimeter"],
        props["eccentricity"],
        props["equivalent_diameter"],
        props["mean_intensity"],
    ])

    # Add GLCM texture features for each detected nucleus.
    texture_props = compute_texture_features(
        labels=labels,
        gray_img=gray_img,
        label_list=props["label"],
    )

    props_array = np.column_stack([base_props, texture_props])
    np.save(prop_path, props_array)

    return "processed"


def process_img_folder(in_dir, out_dir, model,
    pattern, prob_thresh, nms_thresh,
    min_tissue_frac, nonblack_thresh,
    crop_to_tissue, overwrite,
):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_column_file(out_dir)

    patch_files = sorted(
        p for p in in_dir.glob(pattern)
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    counts = {
        "processed": 0,
        "existing": 0,
        "low_tissue": 0,
        "bad_file": 0,
    }

    for img_path in tqdm(patch_files, desc=in_dir.name):
        status = extract_one_patch(
            img_path=img_path,
            out_dir=out_dir,
            model=model,
            prob_thresh=prob_thresh,
            nms_thresh=nms_thresh,
            min_tissue_frac=min_tissue_frac,
            nonblack_thresh=nonblack_thresh,
            crop_to_tissue=crop_to_tissue,
            overwrite=overwrite,
        )

        counts[status] += 1

    print(
        f"{in_dir.name}: "
        f"{counts['processed']} processed, "
        f"{counts['existing']} existing, "
        f"{counts['low_tissue']} low tissue, "
        f"{counts['bad_file']} bad files"
    )

    return counts

def run_stardist_extraction( root_in, root_out, pattern="*.png",
    prob_thresh=0.2, nms_thresh=0.25, min_tissue_frac=0.15, nonblack_thresh=0,
    crop_to_tissue=True, overwrite=False, model_name="2D_versatile_he",
):
    root_in = Path(root_in)
    root_out = Path(root_out)
    root_out.mkdir(parents=True, exist_ok=True)

    img_folders = get_img_folders(root_in)

    if len(img_folders) == 0:
        raise FileNotFoundError(f"No img_* folders found in {root_in}")

    print(f"Found {len(img_folders)} image folders.")
    print(f"Input : {root_in}")
    print(f"Output: {root_out}")

    # we are using 2D_versatile_he as the model_name
    model = StarDist2D.from_pretrained(model_name)

    total_counts = {
        "processed": 0,
        "existing": 0,
        "low_tissue": 0,
        "bad_file": 0,
    }

    for in_dir in img_folders:
        out_dir = root_out / in_dir.name

        counts = process_img_folder(
            in_dir=in_dir, out_dir=out_dir, model=model, pattern=pattern,
            prob_thresh=prob_thresh, nms_thresh=nms_thresh, min_tissue_frac=min_tissue_frac,
            nonblack_thresh=nonblack_thresh, crop_to_tissue=crop_to_tissue, overwrite=overwrite,
        )

        for key in total_counts:
            total_counts[key] += counts[key]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root-in", required=True)
    parser.add_argument("--root-out", required=True)
    parser.add_argument("--pattern", default="*.png")

    parser.add_argument("--prob-thresh", type=float, default=0.2)
    parser.add_argument("--nms-thresh", type=float, default=0.25)
    parser.add_argument("--min-tissue-frac", type=float, default=0.15)
    parser.add_argument("--nonblack-thresh", type=int, default=0)

    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_stardist_extraction(
        root_in=args.root_in,
        root_out=args.root_out,
        pattern=args.pattern,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        min_tissue_frac=args.min_tissue_frac,
        nonblack_thresh=args.nonblack_thresh,
        crop_to_tissue=not args.no_crop,
        overwrite=args.overwrite,
    )