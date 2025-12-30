#!/usr/bin/env python3

import rasterio
from rasterio.enums import Resampling
import os
from affine import Affine

############################################
# Comprime i raster 630 x 630 in 320 x 320 #
############################################


# 1. Apri il dataset originale
input_dir="/leonardo_work/PHD_cozzani/seg_solarbackup/dataset/unsupervised_bologna630"
output_dir='/leonardo_work/PHD_cozzani/seg_solarbackup/dataset/unsupervised_bologna320'

#os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
files= os.listdir(input_dir)
for filename in files:
    if not filename.endswith('.tif'):
        continue  # Salta i file che non sono TIFF
    print(f"Processing {filename}...")
    
    with rasterio.open(os.path.join(input_dir, filename)) as src:
        data = src.read(
            # leggi tutti i bande
            out_shape=(
                src.count,
                320,      # nuova altezza
                320       # nuova larghezza
            ),
            resampling=Resampling.bilinear  # o .nearest, .cubic, ecc.
        )

        # 2. Calcola il nuovo transform: scala i pixel
        scale_x = src.width  / 320
        scale_y = src.height / 320
        transform = src.transform * Affine.scale(
            (scale_x),
            (scale_y)
        )

        # 3. Prepara il profilo per il nuovo file
        profile = src.profile
        profile.update({
            'height': 320,
            'width': 320,
            'transform': transform
        })

    # 4. Scrivi su disco il risultato
    with rasterio.open(os.path.join(output_dir, filename), 'w', **profile) as dst:
        dst.write(data)
