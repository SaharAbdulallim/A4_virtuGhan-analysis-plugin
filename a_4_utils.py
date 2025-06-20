'''
A4: Geospatial Utilities for VirtuGhan QGIS Plugin

This module provides helper functions and a container class for fetching,
processing, and organizing Sentinel-2 index tiles (e.g., NDVI, NDWI) using
VirtuGhan's `vcube.tile` interface. It supports tile-based access, index
computation, saving results, and storing them in a virtual cube container
for later retrieval.

Author: Sahar Mohamed
Date: 2025-06-20
'''

import mercantile
import numpy as np
from PIL import Image
from io import BytesIO
import asyncio
from vcube.tile import TileProcessor  # VirtuGhan tile engine


def fetch_index_tile(lat, lon, zoom=12,
                     start_date="2024-04-01",
                     end_date="2024-04-30",
                     cloud_cover=30,
                     band1="red",
                     band2="nir",
                     formula="(band2-band1)/(band2+band1)",
                     colormap_str="RdYlGn"):
    '''
    Fetches a styled index tile (e.g. NDVI, NDWI) for a given location and date range.

    Returns a PIL Image and STAC metadata feature.
    '''
    x, y, z = mercantile.tile(lon, lat, zoom)
    tp = TileProcessor()
    loop = asyncio.get_event_loop()
    image_bytes, feature = loop.run_until_complete(
        tp.cached_generate_tile(
            x=x, y=y, z=z,
            start_date=start_date, end_date=end_date,
            cloud_cover=cloud_cover,
            band1=band1, band2=band2,
            formula=formula,
            colormap_str=colormap_str
        )
    )
    img = Image.open(BytesIO(image_bytes))
    return img, feature, (x, y, z)


def compute_index(nir_array, red_array, numerator="band2-band1", denominator="band2+band1"):
    '''
    Computes a generalized index = (numerator)/(denominator) from two arrays.

    numerator: expression using 'band1' and 'band2'
    denominator: expression using 'band1' and 'band2'
    '''
    b1 = red_array.astype('float32')
    b2 = nir_array.astype('float32')
    num = eval(numerator.replace("band1", "b1").replace("band2", "b2"))
    den = eval(denominator.replace("band1", "b1").replace("band2", "b2")) + 1e-6
    return num / den


def save_tile_image(image, x, y, z, output_folder="."):
    '''
    Saves a PIL Image tile to disk using tile coordinates in the filename.
    '''
    filename = f"{output_folder}/tile_{z}_{x}_{y}.png"
    image.save(filename)
    print(f"Saved tile image as: {filename}")


class VirtualCube:
    '''
    A container for storing multiple index tiles with metadata.
    '''
    def __init__(self):
        self.tiles = {}  # key: (x,y,z,name), value: (Image, metadata)

    def add_tile(self, name, image, metadata, coords):
        '''Store a tile under a given name. coords = (x,y,z)'''
        key = (*coords, name)
        self.tiles[key] = (image, metadata)

    def list_tiles(self):
        '''List stored tile keys.'''
        return list(self.tiles.keys())

    def get_tile(self, name, coords):
        '''Retrieve stored tile by name and coords.'''
        key = (*coords, name)
        return self.tiles.get(key, None)

    def save_all(self, output_folder="."):
        '''Save all stored tiles to disk.'''
        for (x, y, z, name), (img, _) in self.tiles.items():
            filename = f"{output_folder}/{name}_{z}_{x}_{y}.png"
            img.save(filename)
        print(f"Saved {len(self.tiles)} tiles to {output_folder}")
