'''
A4: Geospatial Utilities for VirtuGhan QGIS Plugin

This module provides helper functions for fetching and processing Sentinel-2 NDVI tiles
using VirtuGhan's `vcube.tile` interface. It supports tile-based access, NDVI computation,
and saving results, and can be reused in both notebooks and QGIS plugins.

Author: Sahar Mohamed
Date: 2025-06-18
'''

import mercantile
import numpy as np
from PIL import Image
from io import BytesIO
import asyncio
from vcube.tile import TileProcessor  # VirtuGhan tile engine


def fetch_ndvi_tile(lat, lon, zoom=12, start_date="2024-04-01", end_date="2024-04-30", cloud_cover=30):
    '''
    Fetches an NDVI tile image for a given location and date range using VirtuGhan TileProcessor.

    Parameters:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        zoom (int): Tile zoom level.
        start_date (str): Start of the date range (YYYY-MM-DD).
        end_date (str): End of the date range (YYYY-MM-DD).
        cloud_cover (int): Cloud cover threshold in percent.

    Returns:
        Tuple (image, metadata): PNG image and tile metadata.
    '''
    x, y, z = mercantile.tile(lon, lat, zoom)
    tile_processor = TileProcessor()
    loop = asyncio.get_event_loop()
    image_bytes, feature = loop.run_until_complete(
        tile_processor.cached_generate_tile(
            x=x,
            y=y,
            z=z,
            start_date=start_date,
            end_date=end_date,
            cloud_cover=cloud_cover,
            band1="red",
            band2="nir",
            formula="(band2-band1)/(band2+band1)",
            colormap_str="RdYlGn",
        )
    )
    image = Image.open(BytesIO(image_bytes))
    return image, feature


def compute_ndvi(nir_array, red_array):
    '''
    Computes NDVI from NIR and Red bands.

    Parameters:
        nir_array (np.ndarray): NIR band array.
        red_array (np.ndarray): Red band array.

    Returns:
        np.ndarray: NDVI array.
    '''
    nir = nir_array.astype('float32')
    red = red_array.astype('float32')
    ndvi = (nir - red) / (nir + red + 1e-6)  # avoid division by zero
    return ndvi


def save_tile_image(image, x, y, z, output_folder="."):
    '''
    Saves the NDVI tile image to disk.

    Parameters:
        image (PIL.Image): Tile image.
        x, y, z (int): Tile coordinates.
        output_folder (str): Output directory.
    '''
    filename = f"{output_folder}/tile_{x}_{y}_{z}.png"
    image.save(filename)
    print(f"Saved tile image as: {filename}")
