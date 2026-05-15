# Approximate Bilevel Graph Structure Learning for Histopathology Image Classification


The structural and spatial arrangements of cells within tissues represent their functional states, making
graph-based learning highly suitable for histopathology image analysis. Existing methods often rely on fixed
graphs that may not capture complex tissue interactions. In this work, we propose ABiG-Net (Approximate
Bilevel Optimization for Graph Structure Learning via Neural Networks), a novel framework that learns
patch interactions within whole slide images (WSI) or large regions of interest (ROI) while simultaneously
learning discriminative node embeddings for the downstream classification task. ABiG-Net hierarchically
models tissue architecture by constructing local patch-level graphs from cellular organization and learning
a global image-level graph that captures sparse, biologically meaningful patch connections through firstorder
approximate bilevel optimization. Here, ‘‘bilevel" denotes a nested optimization strategy where
classifier parameters are optimized at a lower level using training data, while graph adjacency parameters are
optimized at an upper level using validation loss. The first-order approximation removes costly second-order
hypergradient terms, reducing memory and runtime while keeping image-level graph learning tractable for
large WSIs. Clinically, the learned long-range patch interactions link spatially separated but morphologically
related regions, producing interpretable, diagnostically relevant region-to-region evidence. Experiments on
two histopathology datasets demonstrate its effectiveness: on the Extended CRC dataset, ABiG-Net achieves
97.33±1.15% accuracy for three-class colorectal cancer grading and 98.33±0.58% for binary classification;
on the melanoma dataset, it attains 96.27 ± 0.74% for tumor–lymphocyte ROI classification. These results
show that first-order approximate bilevel graph structure learning provides an adaptive, interpretable, and
tractable alternative for large-scale computational pathology.



The input image patches should be organized as follows: 

```text
sample_data/
├── img_001/
│   ├── patch_0.png
│   ├── patch_1.png
│   └── ...
├── img_002/
│   ├── patch_0.png
│   ├── patch_1.png
│   └── ...
└── img_003/
    ├── patch_0.png
    ├── patch_1.png
    └── ...
```

To demonstrate how to use the code, we provide three sample images, each with 10 randomly selected patches.
In this sample example: 
img_001 = training image
img_002 = validation image
img_003 = test image


Training instruction for ABiG-Net:

python run_training.py \
  --train-dirs "path/to/sample_data/img_001" \
  --train-labels "0" \
  --val-dirs "path/to/sample_data/img_002" \
  --val-labels "0" \
  --test-dirs "path/to/sample_data/img_003" \
  --test-labels "1" \
  --out-dim 2 \
  --batch-size 1 \
  --num-iterations 200 \
  --output-dir "outputs/sample_run"



For the full dataset, multiple image folders can be passed using comma-separated paths and labels.
With full dataset info, use following command: 

python run_training.py \
  --train-dirs "path/to/img_001,path/to/img_004,path/to/img_005,path/to/img_006" \
  --train-labels "0,1,0,1" \
  --val-dirs "path/to/img_002,path/to/img_007" \
  --val-labels "0,1" \
  --test-dirs "path/to/img_003,path/to/img_008" \
  --test-labels "1,0" \
  --out-dim 2 \
  --batch-size 1 \
  --num-iterations 200 \
  --output-dir "outputs/full_dataset_run"




Following instructions can be followed to extract features from direct patches:

* Extract nuclear properties
  
python extract_nuclear_props_stardist.py \
  --root-in "path/to/sample_data" \
  --root-out "path/to/sample_data" \
  --pattern "*.png" \
  --overwrite

* Extract 18 cell graph features

python extract_cell_graph_features.py \
  --root-dir "path/to/sample_data" \
  --radius 64 \
  --theta-sim 0.70 \
  --overwrite

* Extract Voronoi/Delaunay/MST/NN features

root_dir = "path/to/sample_data";
extract_vor_del_mst_nn_features(root_dir, 64);

* Combine patch features

python combine_patch_features.py \
  --root-dir "path/to/sample_data" \
  --overwrite
  
Each combined patch feature contains: 51 Voronoi/Delaunay/MST/NN features + 18 cell graph features = 69 features


AI-assisted editing note:
AI-assisted tools were used only to help polish code formatting and comments. The authors written, reviewed and tested all code in this repository.
