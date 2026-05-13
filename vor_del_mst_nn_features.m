function [vfeature, feat_desc] = vor_del_mst_nn_features(x, y)

if nargin < 2, error('Need x and y coordinates'); end

xy = unique(double([x(:), y(:)]), 'rows', 'stable');
n = size(xy, 1);
vfeature = zeros(1, 51);

vorArea = []; vorEdge = []; vorChord = [];
triSide = []; triArea = [];

% Voronoi & Delaunay Features
if n >= 3 && rank(xy - mean(xy, 1)) >= 2
    try
        % Voronoi calculations
        [V, C] = voronoin(xy);
        for i = 1:length(C)
            idx = C{i};
            if isempty(idx) || any(idx == 1), continue; end
            
            cell_xy = V(idx, :);
            if size(cell_xy, 1) < 3 || any(~isfinite(cell_xy(:))), continue; end
            
            vorArea(end+1, 1) = polyarea(cell_xy(:,1), cell_xy(:,2)); %#ok<AGROW>
            
            % Perimeter edges & chords
            closed_cell = [cell_xy; cell_xy(1,:)];
            steps = diff(closed_cell);
            vorEdge = [vorEdge; hypot(steps(:,1), steps(:,2))]; %#ok<AGROW>
            vorChord = [vorChord; pdist(cell_xy)'];             %#ok<AGROW>
        end
        
        % Delaunay calculations
        tri = delaunay(xy(:,1), xy(:,2));
        if ~isempty(tri)
            p1 = xy(tri(:,1),:); p2 = xy(tri(:,2),:); p3 = xy(tri(:,3),:);
            
            triSide = [hypot(p1(:,1)-p2(:,1), p1(:,2)-p2(:,2));
                       hypot(p1(:,1)-p3(:,1), p1(:,2)-p3(:,2));
                       hypot(p2(:,1)-p3(:,1), p2(:,2)-p3(:,2))];
                   
            % Vectorized triangle area
            triArea = abs((p2(:,1)-p1(:,1)).*(p3(:,2)-p1(:,2)) - ...
                          (p3(:,1)-p1(:,1)).*(p2(:,2)-p1(:,2))) / 2;
        end
    catch
        % Silently skip planar features if geometry is degenerate
    end
end

% Distance-based Features (MST, KNN, Radius)
if n > 1
    % Pairwise distance matrix
    D = pdist2(xy, xy);
    
    % MST via standard graph approach
    G = graph(D);
    T = minspantree(G);
    mstEdge = T.Edges.Weight;    
    % KNN distances (k=3, 5, 7)
    Dsort = sort(D, 2);
    k_vals = [3, 5, 7];
    knnSum = zeros(3, n);
    for i = 1:3
        k = min(k_vals(i), n-1);
        knnSum(i,:) = sum(Dsort(:, 2:k+1), 2)';
    end    
    % Neighbors in radius (r=10, 20, 30, 40, 50)
    r_vals = [10, 20, 30, 40, 50];
    radCount = zeros(5, n);
    for i = 1:5
        radCount(i,:) = sum(D <= r_vals(i), 2)' - 1; % -1 to exclude self
    end
else
    mstEdge = []; knnSum = zeros(3, n); radCount = zeros(5, n);
end
% Feature Assignments
vfeature(1) = sum(vorArea);
[vfeature(2), vfeature(3), vfeature(4), vfeature(5)] = get_stats(vorArea);
[vfeature(6), vfeature(7), vfeature(8), vfeature(9)] = get_stats(vorEdge);
[vfeature(10), vfeature(11), vfeature(12), vfeature(13)] = get_stats(vorChord);
% Note: Delaunay order slightly alters std/mean/minmax mapping
[vfeature(15), vfeature(16), vfeature(14), vfeature(17)] = get_stats(triSide);
[vfeature(19), vfeature(20), vfeature(18), vfeature(21)] = get_stats(triArea);
[vfeature(23), vfeature(22), vfeature(24), vfeature(25)] = get_stats(mstEdge);
vfeature(26) = n;
if sum(vorArea) > eps
    vfeature(27) = n / sum(vorArea);
end
% KNN stats
for i = 1:3
    [vfeature(30+i), vfeature(27+i), ~, vfeature(33+i)] = get_stats(knnSum(i,:));
end
% Radius stats
for i = 1:5
    [vfeature(41+i), vfeature(36+i), ~, vfeature(46+i)] = get_stats(radCount(i,:));
end
% Descriptions (Optional output)
if nargout > 1
    feat_desc = get_descriptions();
end
end
% Helper Functions
function [s, m, mm, d] = get_stats(x)
% Consolidates std, mean, min/max ratio, and disorder calculations
if isempty(x)
    s=0; m=0; mm=0; d=0;
    return;
end
s = std(x);
m = mean(x);
if max(x) <= eps
    mm = 0;
else
    mm = min(x) / max(x);
end
if m <= eps
    d = 0;
else
    d = 1 - 1 / (1 + s / m);
end
end
function desc = get_descriptions()
% Compacted list of descriptions
desc = {
    'Total Voronoi polygon area'; 'Voronoi area SD'; 'Voronoi area mean'; 'Voronoi area min-to-max ratio'; 'Voronoi area disorder';
    'Voronoi perimeter-edge length SD'; 'Voronoi perimeter-edge length mean'; 'Voronoi perimeter-edge length min-to-max ratio'; 'Voronoi perimeter-edge length disorder';
    'Voronoi chord length SD'; 'Voronoi chord length mean'; 'Voronoi chord length min-to-max ratio'; 'Voronoi chord length disorder';
    'Delaunay side length min-to-max ratio'; 'Delaunay side length SD'; 'Delaunay side length mean'; 'Delaunay side length disorder';
    'Delaunay triangle area min-to-max ratio'; 'Delaunay triangle area SD'; 'Delaunay triangle area mean'; 'Delaunay triangle area disorder';
    'Minimum spanning tree edge length mean'; 'Minimum spanning tree edge length SD'; 'Minimum spanning tree edge length min-to-max ratio'; 'Minimum spanning tree edge length disorder';
    'Number of nuclei'; 'Nuclear density';
    'Mean summed distance to 3 nearest neighbors'; 'Mean summed distance to 5 nearest neighbors'; 'Mean summed distance to 7 nearest neighbors';
    'SD of summed distances to 3 nearest neighbors'; 'SD of summed distances to 5 nearest neighbors'; 'SD of summed distances to 7 nearest neighbors';
    'Disorder of summed distances to 3 nearest neighbors'; 'Disorder of summed distances to 5 nearest neighbors'; 'Disorder of summed distances to 7 nearest neighbors';
    'Mean number of neighbors within 10px'; 'Mean number of neighbors within 20px'; 'Mean number of neighbors within 30px'; 'Mean number of neighbors within 40px'; 'Mean number of neighbors within 50px';
    'SD of neighbors within 10px'; 'SD of neighbors within 20px'; 'SD of neighbors within 30px'; 'SD of neighbors within 40px'; 'SD of neighbors within 50px';
    'Disorder of neighbors within 10px'; 'Disorder of neighbors within 20px'; 'Disorder of neighbors within 30px'; 'Disorder of neighbors within 40px'; 'Disorder of neighbors within 50px'
};
end