function extract_vor_del_mst_nn_features(root_dir, min_nuclei, start_idx)
if nargin < 2
    min_nuclei = 64;
end
if nargin < 3
    start_idx = 1;
end
root_dir = string(root_dir);

if ~isfolder(root_dir)
    error("Folder not found: %s", root_dir);
end
% Grab all img_ folders directly.
d = dir(fullfile(root_dir, "img_*"));
img_folders = string({d([d.isdir]).name});
img_folders = sort(img_folders);
if isempty(img_folders)
    disp("No img_* folders found.");
    return;
end
fprintf("Found %d img_* folders.\n", numel(img_folders));
for i = start_idx:numel(img_folders)
    img_dir = fullfile(root_dir, img_folders(i));
    fprintf("\nProcessing %s (%d/%d)...\n", ...
        img_folders(i), i, numel(img_folders));
    files = dir(fullfile(img_dir, "props_*.npy"));
    if isempty(files)
        fprintf("  No props_*.npy files found.\n");
        continue;
    end
    fprintf("  Found %d props files.\n", numel(files));
    saved_count = 0;
    skipped_count = 0;
    failed_count = 0;
    for j = 1:numel(files)
        file_path = fullfile(files(j).folder, files(j).name);
        try
            props = readNPY(file_path);
        catch ME
            warning("Failed to load %s: %s", files(j).name, ME.message);
            failed_count = failed_count + 1;
            continue;
        end
        % Need at least label + centroid-0 + centroid-1.
        if size(props, 1) <= min_nuclei || size(props, 2) < 3
            skipped_count = skipped_count + 1;
            continue;
        end
        % props columns:
        % column 1 = label
        % column 2 = centroid-0 / y
        % column 3 = centroid-1 / x
        y = double(props(:, 2));
        x = double(props(:, 3));
        % Remove invalid centroid values, if any.
        valid = isfinite(x) & isfinite(y);
        x = x(valid);
        y = y(valid);
        if numel(x) <= min_nuclei
            skipped_count = skipped_count + 1;
            continue;
        end
        try
            [graph_feats, feat_names] = vor_del_mst_nn_features(x, y);
        catch ME
            warning("Feature extraction failed for %s: %s", ...
                files(j).name, ME.message);
            failed_count = failed_count + 1;
            continue;
        end
        out_name = strrep(files(j).name, "props_", "vordelmstnn_graph_features_");
        out_name = strrep(out_name, ".npy", ".mat");
        out_path = fullfile(files(j).folder, out_name);
        save(out_path, "graph_feats", "feat_names", "-v7");
        saved_count = saved_count + 1;
    end
    fprintf("  Saved %d. Skipped %d. Failed %d.\n", ...
        saved_count, skipped_count, failed_count);
end
disp("Done.");
end