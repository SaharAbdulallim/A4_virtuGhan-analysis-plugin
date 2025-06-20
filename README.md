# A4: VirtuGhan Tile-Based Index Analysis

**Author:** Sahar Mohamed  
**Date:** June 2025

---

## Notebook Overview

This notebook demonstrates a modular, tile‑based workflow for computing and visualizing Sentinel‑2 spectral indices—specifically **NDVI** (Normalized Difference Vegetation Index) and **NDWI** (Normalized Difference Water Index)—using the **VirtuGhan** Python package. By leveraging VirtuGhan’s `vcube.tile.TileProcessor`, I fetch on‑the‑fly Cloud‑Optimized GeoTIFF (COG) tiles, apply custom formulas, style them with Matplotlib colormaps, and store results in an in‑memory “VirtualCube” container for easy retrieval and batch export.

---

## Goals

1. **High‑Level Tile Fetching**  
   Use `fetch_index_tile()` to retrieve styled index tiles for any two-band formula, without manual STAC or rasterio boilerplate.  
2. **Custom Index Computation**  
   Show how to compute NDWI by swapping band arguments and formula strings, demonstrating the flexibility of `compute_index()`.  
3. **Result Management**  
   Organize multiple tiles (NDVI, NDWI) in a `VirtualCube` container, list and display them in the notebook `A4_Notebook_VirtuGhanAnalysis.ipynb`, and save them in batch to disk.  
4. **Modular Design**  
   Encapsulate all functionality in `A4_utils.py` so these helpers can be reused in a QGIS plugin or other geospatial workflows.

---

## Key Components

- **`fetch_index_tile(lat, lon, …)`**  
  High‑level wrapper around VirtuGhan’s tile engine for any band‑difference index.  
- **`compute_index(b1_arr, b2_arr, formula, ...)`**  
  Generalized NumPy‑based index calculator for raw arrays.  
- **`VirtualCube`**  
  In‑memory container for storing, listing, displaying, and batch‑saving multiple tile images.  
- **`save_tile_image(img, x, y, z, folder)`**  
  Utility to save a PIL image with tile coordinates embedded in its filename.

---

## Dependencies & Environment
This notebook depends on the [VirtuGhan GitHub repository](https://github.com/kshitijrajsharma/virtughan) for its tile‑processing engine.

- **Python 3.10**  
- `virtughan` (vcube)  
- `mercantile`  
- `rasterio`  
- `numpy`  
- `Pillow`  
- `matplotlib`  
- `nest_asyncio`

A `environment.yml` is provided for Conda-based reproduction:

```yaml
name: a4-env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - numpy<2
  - pip
  - pip:
    - mercantile
    - rasterio
    - Pillow
    - matplotlib
    - nest_asyncio
    - -e ../virtughan
