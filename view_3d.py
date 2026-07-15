"""
Visualize a simpeg/discretize mesh + model in PyVista.

USAGE
-----
1. After your inversion run, save the mesh, active cells, and recovered model:

       import numpy as np, pickle
       np.save("active_cells.npy", active_cells)
       np.save("model_recovered.npy", minv_tetm)   # log-conductivity, active cells only
       with open("mesh.pkl", "wb") as f:
           pickle.dump(mesh, f)

   Or for UBC format (TensorMesh only):
       mesh.write_UBC("mesh.txt", models={"resistivity.txt": np.exp(-minv_tetm)})

2. Edit the CONFIG block below, then run:
       python visualize_model.py
"""

import numpy as np
import pickle
import pyvista as pv
import shapely
from shapely.geometry import MultiPoint
from discretize import TreeMesh, TensorMesh
from simpeg import utils
import discretize

inversion_dir = '3dinversion_results_01/'

with np.load(f'out/{inversion_dir}final_model.npz', allow_pickle=True) as data:
    model = data['model']
    dpred = data['dpred']
    freqs = data['freqs']
    rx_locs = data['rx_locs']
    mesh = data['mesh'].item()
    active_cells = data['active_cells']

mesh = discretize.TensorMesh.deserialize(mesh)

# ============================================================
# CONFIG — edit these paths and options
# ============================================================

# MESH_FILE        = "/Users/johnkuttai/Dropbox/JohnLindsey/mtsphere/mesh_octree_fine.txt"          # path to pickled discretize mesh
# ACTIVE_FILE      = "/Users/johnkuttai/Dropbox/JohnLindsey/mtsphere/active_octree_fine.npy"  # boolean array, shape (nC,)
# MODEL_FILE       = "/Users/johnkuttai/Dropbox/JohnLindsey/mtsphere/final_model_hfreq.npy"  # 1D array over *active* cells (log-sigma)
# MODEL_FILE       = "/Users/johnkuttai/Dropbox/JohnLindsey/mtsphere/3D_model_raglan_smooth_labels.npy" 

# Model transform applied before plotting.
# "log10_rho" : convert log-sigma  → log10-resistivity  (standard MT display)
# "sigma"     : convert log-sigma  → conductivity
# "log_sigma" : plot as-is (natural-log conductivity)
MODEL_TRANSFORM  = "sigma"

# Scalar bar label shown in the plot
SCALAR_LABEL     = "log10(Resistivity [Ohm·m])"

# Value assigned to inactive (air/padding) cells — set to np.nan to hide them
INACTIVE_VALUE   = np.nan

# Color map
COLORMAP         = "turbo_r"

# Mesh cell edges on slices
SHOW_MESH_EDGES  = False       # draw cell edges on each slice
EDGE_COLOR       = "black"     # edge line colour
EDGE_OPACITY     = 0.4         # 0 = invisible, 1 = fully opaque
EDGE_LINE_WIDTH  = 0.5         # line width in screen pixels

# Scalar range for the colorbar — set to None for auto
SCALAR_RANGE     = (np.log10(100), np.log10(1e5))  # e.g. (1.0, 6.0)

# 3-D volume rendering — show the full model (air/inactive cells excluded)
SHOW_3D_MODEL    = True   # set to True to render the 3-D volume in addition to slices
MODEL_3D_OPACITY = 0.7    # opacity of the 3-D volume (0 = transparent, 1 = opaque)
MODEL_3D_CUTOFF  = np.log10(500)    # only render cells whose scalar value is <= this cutoff;
                          # set to None to render all active cells

# -------------------------------------------------------
# SLICES  — add or remove entries freely.
#
# Each entry is a dict with:
#   "normal" : "x" | "y" | "z"
#   "origin" : coordinate value along that axis where the slice sits
#              (in the same units as the mesh, e.g. UTM metres)
#              Set to None to place the slice at the mesh centre.
# -------------------------------------------------------
SLICES = [
    {"normal": "x", "origin": None},   # E–W vertical slice at mesh centre
    {"normal": "y", "origin": None},   # N–S vertical slice at mesh centre
    {"normal": "z", "origin": 300},    # horizontal slice starting elevation (m)
]

# When True, any z-normal slice in SLICES becomes a slider-controlled
# interactive depth slice.  The first z-entry's origin sets the initial value.
DEPTH_SLICE_INTERACTIVE    = True

# When True, x- and y-normal slices become slider-controlled vertical slices.
# Easting slider appears on the left; Northing slider on the right.
VERTICAL_SLICE_INTERACTIVE = True

# -------------------------------------------------------
# RECEIVER OVERLAY — set to None to skip
# -------------------------------------------------------
RECEIVER_FILE        =  "/Users/johnkuttai/Dropbox/JohnLindsey/mcr_admm/L2-3d/mcr_locations.npy" # "/Users/johnkuttai/Dropbox/JohnLindsey/mtsphere/raglan_locations.npy"
RECEIVER_Z_DEFAULT   = 0.0        # elevation used when the array has only 2 columns
RECEIVER_COLOR       = "white"    # point colour (any PyVista/matplotlib colour string)
RECEIVER_POINT_SIZE  = 10         # point size in screen pixels
RECEIVER_LABEL       = "Receivers"  # legend entry label (set to None to hide legend entry)

# -------------------------------------------------------
# CORE CLIPPING — clip the rendered volume to a concave-hull
# (alpha-shape) polygon enclosing the receiver locations, buffered
# outward by CORE_PADDING.  Cells whose centre falls outside the
# polygon, or outside the depth/height cutoffs, are not rendered.
# Requires RECEIVER_FILE to be set.
# -------------------------------------------------------
CLIP_TO_CORE       = True    # set to False to show the full mesh
CORE_HULL_RATIO    = 0.7     # 0.0 = tightest concave hull (alpha shape), 1.0 = convex hull
CORE_PADDING       = 100.0   # metres the polygon is buffered outward from the receiver hull
DEPTH_CUTOFF       = -800.0 # minimum elevation to render (bottom clip); set to None for full depth
HEIGHT_CUTOFF      = 480    # maximum elevation to render (top clip);    set to None for full height
SHOW_CLIP_POLYGON  = True     # draw the clipping polygon outline for reference
CLIP_POLYGON_COLOR = "yellow"

# -------------------------------------------------------
# GAUSSIAN SMOOTHING — isotropic 3-D Gaussian smoothing applied
# to the model before polygon clipping.
# -------------------------------------------------------
SMOOTH             = False   # set to True to enable smoothing
SMOOTH_SIGMA       = 100.0   # Gaussian sigma in metres (isotropic, all directions)

# ============================================================
# LOAD MESH + MODEL
# ============================================================

print("[INFO] Loading active cells and model ...")
model_active = model

# create a core mesh
# active_core_cells, mesh_core = utils.meshutils.extract_core_mesh(mesh)

# ============================================================
# EXPAND MODEL TO FULL MESH
# ============================================================

model_full = np.full(mesh.nC, np.nan)
model_full[active_cells] = 1/np.exp(model_active)  # convert log-sigma → resistivity for plotting
model_full[~active_cells] = INACTIVE_VALUE
model_plot = np.log10(model_full)

# ============================================================
# GAUSSIAN SMOOTHING (before polygon clip)
# ============================================================

if SMOOTH:
    from scipy.spatial import cKDTree as _cKDTree
    _valid = np.isfinite(model_plot)
    n_valid = int(_valid.sum())
    print(f"[INFO] Applying Gaussian smoothing (sigma={SMOOTH_SIGMA} m, {n_valid} active cells) ...")
    _cutoff = 3.0 * SMOOTH_SIGMA
    _tree   = _cKDTree(mesh.cell_centers[_valid])   # 3-D
    _W      = _tree.sparse_distance_matrix(_tree, _cutoff, output_type='coo_matrix')
    _W.data = np.exp(-_W.data ** 2 / (2.0 * SMOOTH_SIGMA ** 2))
    _W_csr  = _W.tocsr()
    _v      = model_plot[_valid]
    model_plot[_valid] = _W_csr.dot(_v) / _W_csr.dot(np.ones(n_valid))
    print("[INFO] Smoothing done.")
# np.save("3D_model_raglan_smooth.npy", model_plot)
# ============================================================
# BUILD PyVista GRID
# ============================================================

print(f"[INFO] Mesh type: {type(mesh).__name__}  nC={mesh.nC}")

if isinstance(mesh, discretize.TensorMesh):
    # RectilinearGrid — fastest and most memory-efficient for tensor meshes
    grid = pv.RectilinearGrid(mesh.nodes_x, mesh.nodes_y, mesh.nodes_z)
    # PyVista RectilinearGrid cell ordering matches discretize's Fortran ordering
    grid.cell_data[SCALAR_LABEL] = model_plot

elif isinstance(mesh, discretize.TreeMesh):
    # Export via VTK — discretize's to_vtk() returns a pyvista-compatible object
    try:
        vtk_obj = mesh.to_vtk()
        # to_vtk returns a pyvista UnstructuredGrid
        grid = pv.wrap(vtk_obj)
        # Cell ordering for TreeMesh matches the model ordering directly
        grid.cell_data[SCALAR_LABEL] = model_plot
    except AttributeError:
        raise RuntimeError(
            "mesh.to_vtk() not available — upgrade discretize to >= 0.8 "
            "or use TensorMesh instead."
        )
else:
    raise TypeError(f"Unsupported mesh type: {type(mesh)}")

# ============================================================
# CLIP GRID TO CORE (receiver XY footprint)
# ============================================================

clip_polygon_rings = None  # filled in below when CLIP_TO_CORE is active; used for the overlay

if CLIP_TO_CORE and RECEIVER_FILE is not None:
    print(f"[INFO] Clipping grid to receiver polygon (concave hull, ratio={CORE_HULL_RATIO}) ...")
    _rx = rx_locs
    if _rx.ndim == 1:
        _rx = _rx.reshape(1, -1)
    _rx_xy = _rx[:, :2].astype(float)

    _hull = shapely.concave_hull(MultiPoint(_rx_xy), ratio=CORE_HULL_RATIO)
    _clip_polygon = _hull.buffer(CORE_PADDING)

    _polys = list(_clip_polygon.geoms) if _clip_polygon.geom_type == "MultiPolygon" else [_clip_polygon]
    clip_polygon_rings = [np.asarray(p.exterior.coords) for p in _polys]

    _z_min = float(DEPTH_CUTOFF)  if DEPTH_CUTOFF  is not None else float(grid.bounds[4])
    _z_max = float(HEIGHT_CUTOFF) if HEIGHT_CUTOFF is not None else float(grid.bounds[5])

    _centers = grid.cell_centers().points
    _inside_xy = shapely.contains_xy(_clip_polygon, _centers[:, 0], _centers[:, 1])
    _inside_z  = (_centers[:, 2] >= _z_min) & (_centers[:, 2] <= _z_max)
    grid = grid.extract_cells(_inside_xy & _inside_z)

    print(f"[INFO] Clip polygon area: {_clip_polygon.area:.0f} m^2  "
          f"Z=[{_z_min:.0f}, {_z_max:.0f}]  kept {grid.n_cells} cells")

# ============================================================
# COMPUTE SLICE ORIGINS
# ============================================================

mesh_center = np.array(mesh.origin) + np.array([mesh.h[0].sum(), mesh.h[1].sum(), mesh.h[2].sum()]) / 2
axis_to_index = {"x": 0, "y": 1, "z": 2}
axis_to_normal = {
    "x": [1, 0, 0],   # x-z slice rotated 45° about the vertical (z) axis
    "y": [0, 1, 0],  # y-z slice rotated 45° about the vertical (z) axis
    "z": [0, 0, 1],
}
# axis_to_normal = {
#     "x": [1, 0, 0],   # x-z slice rotated 45° about the vertical (z) axis
#     "y": [0, 1, 0],  # y-z slice rotated 45° about the vertical (z) axis
#     "z": [0, 0, 1],
# }

# ============================================================
# PLOT
# ============================================================

print("[INFO] Rendering ...")

plotter = pv.Plotter()

scalar_range = SCALAR_RANGE
if scalar_range is None:
    finite_vals = model_plot[np.isfinite(model_plot)]
    if len(finite_vals):
        scalar_range = (np.percentile(finite_vals, 2), np.percentile(finite_vals, 98))
    else:
        scalar_range = (0, 1)

sargs = dict(
    title=SCALAR_LABEL,
    title_font_size=14,
    label_font_size=12,
    n_labels=5,
    fmt="%.1f",
    vertical=True,
)

# 3-D volume (active cells only — NaN air cells are thresholded out)
if SHOW_3D_MODEL:
    upper = MODEL_3D_CUTOFF if MODEL_3D_CUTOFF is not None else np.nanmax(model_plot[np.isfinite(model_plot)])
    grid_active = grid.threshold(value=upper, scalars=SCALAR_LABEL, method='lower')  # keeps cells <= cutoff, removes NaN
    plotter.add_mesh(
        grid_active,
        scalars=SCALAR_LABEL,
        cmap=COLORMAP,
        clim=scalar_range,
        scalar_bar_args=sargs,
        show_scalar_bar=True,
        show_edges=SHOW_MESH_EDGES,
        edge_color=EDGE_COLOR,
        edge_opacity=EDGE_OPACITY,
        line_width=EDGE_LINE_WIDTH,
        opacity=MODEL_3D_OPACITY,
    )

x_slices = [s for s in SLICES if s["normal"].lower() == "x"]
y_slices = [s for s in SLICES if s["normal"].lower() == "y"]
z_slices = [s for s in SLICES if s["normal"].lower() == "z"]

_slice_mesh_kwargs = dict(
    scalars=SCALAR_LABEL,
    cmap=COLORMAP,
    clim=scalar_range,
    scalar_bar_args=sargs,
    show_scalar_bar=True,
    show_edges=SHOW_MESH_EDGES,
    edge_color=EDGE_COLOR,
    edge_opacity=EDGE_OPACITY,
    line_width=EDGE_LINE_WIDTH,
)

# --- Easting (x-normal) slice ---
if VERTICAL_SLICE_INTERACTIVE and x_slices:
    x_init = float(x_slices[0]["origin"]) if x_slices[0]["origin"] is not None else float(mesh_center[0])

    def _x_slice_cb(value):
        origin_pt = [float(value), float(mesh_center[1]), float(mesh_center[2])]
        slc = grid.slice(normal=axis_to_normal["x"], origin=origin_pt)
        plotter.add_mesh(slc, name="_x_vert_slice", **_slice_mesh_kwargs)

    plotter.add_slider_widget(
        _x_slice_cb,
        rng=[float(grid.bounds[0]), float(grid.bounds[1])],
        value=x_init,
        title="Easting (m)",
        pointa=(0.025, 0.1),
        pointb=(0.025, 0.9),
        fmt="%.0f",
    )
    _x_slice_cb(x_init)
else:
    for s in x_slices:
        origin_val = s["origin"] if s["origin"] is not None else float(mesh_center[0])
        origin_pt  = [float(origin_val), float(mesh_center[1]), float(mesh_center[2])]
        slc = grid.slice(normal=axis_to_normal["x"], origin=origin_pt)
        plotter.add_mesh(slc, **_slice_mesh_kwargs)

# --- Northing (y-normal) slice ---
if VERTICAL_SLICE_INTERACTIVE and y_slices:
    y_init = float(y_slices[0]["origin"]) if y_slices[0]["origin"] is not None else float(mesh_center[1])

    def _y_slice_cb(value):
        origin_pt = [float(mesh_center[0]), float(value), float(mesh_center[2])]
        slc = grid.slice(normal=axis_to_normal["y"], origin=origin_pt)
        plotter.add_mesh(slc, name="_y_vert_slice", **_slice_mesh_kwargs)

    plotter.add_slider_widget(
        _y_slice_cb,
        rng=[float(grid.bounds[2]), float(grid.bounds[3])],
        value=y_init,
        title="Northing (m)",
        pointa=(0.975, 0.1),
        pointb=(0.975, 0.9),
        fmt="%.0f",
    )
    _y_slice_cb(y_init)
else:
    for s in y_slices:
        origin_val = s["origin"] if s["origin"] is not None else float(mesh_center[1])
        origin_pt  = [float(mesh_center[0]), float(origin_val), float(mesh_center[2])]
        slc = grid.slice(normal=axis_to_normal["y"], origin=origin_pt)
        plotter.add_mesh(slc, **_slice_mesh_kwargs)

# --- Depth (z-normal) slice ---
if DEPTH_SLICE_INTERACTIVE and z_slices:
    z_init = float(z_slices[0]["origin"]) if z_slices[0]["origin"] is not None else float(mesh_center[2])

    def _depth_slice_cb(value):
        origin_pt = [float(mesh_center[0]), float(mesh_center[1]), float(value)]
        slc = grid.slice(normal=[0, 0, 1], origin=origin_pt)
        plotter.add_mesh(slc, name="_z_depth_slice", **_slice_mesh_kwargs)

    plotter.add_slider_widget(
        _depth_slice_cb,
        rng=[float(grid.bounds[4]), float(grid.bounds[5])],
        value=z_init,
        title="Elevation (m)",
        pointa=(0.1, 0.05),
        pointb=(0.9, 0.05),
        fmt="%.0f",
    )
    _depth_slice_cb(z_init)
else:
    for s in z_slices:
        origin_val = s["origin"] if s["origin"] is not None else float(mesh_center[2])
        origin_pt  = [float(mesh_center[0]), float(mesh_center[1]), float(origin_val)]
        slc = grid.slice(normal=[0, 0, 1], origin=origin_pt)
        plotter.add_mesh(slc, **_slice_mesh_kwargs)

# Mesh bounding box outline for spatial reference
plotter.add_mesh(grid.outline(), color="black", line_width=1)

# Receiver locations
if RECEIVER_FILE is not None:
    print(f"Loading receivers")
    rx = rx_locs
    if rx.ndim == 1:
        rx = rx.reshape(1, -1)
    if rx.shape[1] == 2:
        # 2-column array — append a constant Z column
        rx = np.hstack([rx, np.full((len(rx), 1), RECEIVER_Z_DEFAULT)])
    rx_pts = pv.PolyData(rx[:, :3].astype(float))
    actor = plotter.add_mesh(
        rx_pts,
        color=RECEIVER_COLOR,
        point_size=RECEIVER_POINT_SIZE,
        render_points_as_spheres=True,
        label=RECEIVER_LABEL,
    )
    if RECEIVER_LABEL is not None:
        plotter.add_legend(bcolor=(0.1, 0.1, 0.1), border=True)
    print(f"[INFO] {len(rx)} receiver locations rendered.")

    if SHOW_CLIP_POLYGON and clip_polygon_rings is not None:
        for _ring_xy in clip_polygon_rings:
            _ring_z = np.full((len(_ring_xy), 1), RECEIVER_Z_DEFAULT)
            _ring_pts = np.hstack([_ring_xy, _ring_z])
            poly_line = pv.lines_from_points(_ring_pts, close=True)
            plotter.add_mesh(poly_line, color=CLIP_POLYGON_COLOR, line_width=3)

plotter.add_axes(line_width=3)
plotter.show_grid()
plotter.show(title="SimPEG Model Viewer")
