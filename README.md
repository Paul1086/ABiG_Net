# Approximate Bilevel Graph Structure Learning for Histopathology Image Classification


ABiG-Net (Approximate Bilevel Optimization for Graph Structure Learning via Neural Networks) is a graph-based histopathology framework that learns adaptive patch interactions within WSIs/ROIs while jointly learning discriminative node embeddings for classification. It constructs local patch-level graphs from cellular organization and learns a sparse global image-level graph through first-order approximate bilevel optimization, avoiding costly second-order hypergradient computation. In this bilevel setup, the lower level optimizes classifier parameters using training data, while the upper level optimizes graph adjacency parameters using validation loss.



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

```text
img_001 = training image
img_002 = validation image
img_003 = test image
```


Training instruction for ABiG-Net:
```
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
```


For the full dataset, multiple image folders can be passed using comma-separated paths and labels.
With full dataset info, use following command: 
(Please set the hyperparameters based on your requirements)
```
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
```



Following instructions can be followed to extract features from direct patches:

* Extract nuclear properties (Table 3)
``` 
python extract_nuclear_props_stardist.py \
  --root-in "path/to/sample_data" \
  --root-out "path/to/sample_data" \
  --pattern "*.png" \
  --overwrite
```
* Extract 18 cell graph features (Table 4: 18 patch-level graph features)
```
python extract_cell_graph_features.py \
  --root-dir "path/to/sample_data" \
  --radius 64 \
  --theta-sim 0.70 \
  --overwrite
```

* Extract Voronoi/Delaunay/MST/NN features (Table 5: 51 patch-level graph features)
```
root_dir = "path/to/sample_data";
extract_vor_del_mst_nn_features(root_dir, 64);
```

* Combine patch features
```
python combine_patch_features.py \
  --root-dir "path/to/sample_data" \
  --overwrite
```  
Each combined patch feature contains: 51 Voronoi/Delaunay/MST/NN features + 18 cell graph features = 69 features


AI-assisted editing note:
AI-assisted tools were used only to help polish code formatting and comments. The authors written, reviewed and tested all code in this repository.
```
