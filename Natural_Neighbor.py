import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from shapely.geometry import Polygon, Point
from tqdm import tqdm



def poly_area(coords):
    """
    Calculates the area of a polygon given its vertices.
    Uses the shapely library for robustness.
    
    Args:
        coords (np.ndarray): A NumPy array of shape (N, 2) with polygon vertices.
        
    Returns:
        float: The area of the polygon. Returns 0 if coords are invalid.
    """
    if coords is None or len(coords) < 3:
        return 0.0
    return Polygon(coords).area


def clipped_voronoi(nodes, clip_poly):
    """
    Computes the Voronoi diagram for a set of nodes and clips it to a given polygon.
    
    Args:
        nodes (np.ndarray): An array of shape (N, 2) of node coordinates.
        clip_poly (shapely.geometry.Polygon): The polygon to clip the Voronoi cells against.
        
    Returns:
        list: A list of shapely Polygon objects representing the clipped Voronoi cells.
    """
    if not isinstance(nodes, np.ndarray):
        nodes = np.array(nodes)

    # Add far-away points to bound the Voronoi diagram.
    # This helps ensure all Voronoi regions are finite.
    min_coord, max_coord = np.min(nodes, axis=0) - 1, np.max(nodes, axis=0) + 1
    bounding_box = np.array([
        [min_coord[0] - 1000, min_coord[1] - 1000],
        [max_coord[0] + 1000, min_coord[1] - 1000],
        [max_coord[0] + 1000, max_coord[1] + 1000],
        [min_coord[0] - 1000, max_coord[1] + 1000]
    ])
    extended_nodes = np.vstack([nodes, bounding_box])

    # Compute Voronoi diagram
    vor = Voronoi(extended_nodes)
    
    clipped_cells = []
    for i in range(len(nodes)):  
        # Get the region for the i-th node
        region_idx = vor.point_region[i]
        
        # Get the vertices of the Voronoi region
        if -1 in vor.regions[region_idx] or len(vor.regions[region_idx]) == 0:
            # This case handles unbounded regions, though adding far points should prevent it.
            clipped_cells.append(None) 
            continue
            
        vor_vertices = vor.vertices[vor.regions[region_idx]]
        
        # Create a shapely Polygon for the Voronoi cell
        vor_poly = Polygon(vor_vertices)
        if not vor_poly.is_valid:
            vor_poly = vor_poly.buffer(0)
        
        # Clip the Voronoi cell with the clipping area
        clipped_cell = clip_poly.intersection(vor_poly)
        
        clipped_cells.append(clipped_cell)
        
    return clipped_cells


def compute_natural_neighbor_weights(x_i, Q, clip_poly, orig_areas):
    """
    Computes the natural neighbor weights for a query point Q.
    
    Args:
        x_i (np.ndarray): The original data points (N, 2).
        Q (np.ndarray): The query point (2,) or (1, 2).
        clip_poly (shapely.geometry.Polygon): The clipping polygon (convex hull).
        orig_areas (list): A list of the areas of the original Voronoi cells.
        
    Returns:
        tuple: A tuple containing:
            - np.ndarray: Indices of the natural neighbors.
            - np.ndarray: Corresponding weights for each neighbor.
    """
    Q = np.asarray(Q).ravel()

    # Check if Q is one of the input points (within floating point tolerance)
    matches = np.all(np.isclose(x_i, Q, atol=1e-9), axis=1)
    if np.any(matches):
        neighbors = np.where(matches)[0][:1]
        weights = np.array([1.0])
        return neighbors, weights

    # Check if Q is inside the convex hull
    if not clip_poly.contains(Point(Q)):
        return np.array([], dtype=int), np.array([])
        
    # Add Q to the set of points and recompute the clipped Voronoi diagram
    extended_points = np.vstack([x_i, Q])
    v_cells_new = clipped_voronoi(extended_points, clip_poly)
    
    neighbors = []
    stolen_areas = []
    
    n = len(x_i)
    for i in range(n):
        # Calculate the area of the new cell for the original point i
        new_area = v_cells_new[i].area if v_cells_new[i] is not None else 0.0
        
        # If the original area was larger (beyond float precision noise), area was "stolen" by Q
        diff = orig_areas[i] - new_area
        if diff > 1e-9:
            neighbors.append(i)
            stolen_areas.append(diff)

    stolen_areas = np.array(stolen_areas)
    total_stolen_area = np.sum(stolen_areas)
    
    if total_stolen_area > 0:
        weights = stolen_areas / total_stolen_area
    else:
        # This can happen if Q is very close to an edge
        return np.array([], dtype=int), np.array([])

    return np.array(neighbors, dtype=int), weights


def natural_neighbor_map5(x_i, f_i, res, X_grid=None, Y_grid=None):
    """
    Performs natural neighbor interpolation on a 2D grid.
    This is the main function
    
    Args:
        x_i (np.ndarray): Input data points of shape (N, 2).
        f_i (np.ndarray): Function values at each data point, shape (N,) or (N, 1).
        res (int): The resolution of the output grid (res x res).
        X_grid (np.ndarray, optional): Custom X meshgrid of shape (res, res).
        Y_grid (np.ndarray, optional): Custom Y meshgrid of shape (res, res).
        
    Returns:
        tuple: A tuple containing:
            - np.ndarray: X coordinates of the grid.
            - np.ndarray: Y coordinates of the grid.
            - np.ndarray: Z (interpolated) values on the grid.
    """
    f_i = np.asarray(f_i).ravel()

    # Create the grid for interpolation if custom X_grid/Y_grid are not provided
    if X_grid is None or Y_grid is None:
        x_max, y_max = np.max(x_i[:, 0]), np.max(x_i[:, 1])
        x_grid = np.linspace(0, x_max, res)
        y_grid = np.linspace(0, y_max, res)
        X, Y = np.meshgrid(x_grid, y_grid)
    else:
        X, Y = X_grid, Y_grid
        res = X.shape[0]

    Z = np.full(X.shape, np.nan)
    
    # Compute the convex hull to define the interpolation area
    hull = ConvexHull(x_i)
    clip_poly = Polygon(x_i[hull.vertices])
    
    # Precompute the original Voronoi cells and their areas
    v_cells_orig = clipped_voronoi(x_i, clip_poly)
    orig_areas = [cell.area if cell is not None else 0.0 for cell in v_cells_orig]
    
    # Main loop for interpolation
    # tqdm provides a progress bar
    rows, cols = X.shape
    for i in tqdm(range(rows), desc="Natural Neighbor Interpolation"):
        for j in range(cols):
            Q = np.array([X[i, j], Y[i, j]])
            
            # Compute weights for the current query point
            neighbors, weights = compute_natural_neighbor_weights(x_i, Q, clip_poly, orig_areas)
            
            # If neighbors were found, compute the interpolated value
            if neighbors.size > 0 and weights.size > 0:
                Z[i, j] = np.dot(f_i[neighbors], weights)
                
    return X, Y, Z

