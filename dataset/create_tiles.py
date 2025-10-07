#!/usr/bin/env python3

##########################################################################################################################
# Ritaglia i raster originali in immagini con risoluzione 630 x 630 (successivamente da comprimere con compress_tiles.py)#
##########################################################################################################################


import os
import rasterio
from rasterio.windows import Window

# Percorso al file tile originale
input_dir = 'dataset/unsupervised_bologna20802'
# Cartella di output per le tessere
output_dir = 'dataset/unsupervised_bologna320/'
# Numero di suddivisioni (griglia NxN)
# Overlap in pixel su ogni lato
overlap = 5
# 1. Apri il dataset original
#os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
files= os.listdir(input_dir)
for filename in files:
    if not filename.endswith('.tif'):
        continue  # Salta i file che non sono TIFF
    print(f"Processing {filename}...")

    with rasterio.open(os.path.join(input_dir,filename)) as src:
        width = src.width
        height = src.height

        # Calcola dimensione base di ciascuna tile senza overlap
        base_width = 630 #width // n_cols
        base_height = 630 #height // n_rows
        # Calcola la dimensione totale della griglia
        n_cols = width // base_width 
        n_rows = height // base_height
        # Resto per l'ultima colonna/riga
        extra_width = width - base_width * n_cols
        extra_height = height - base_height * n_rows

        for row in range(n_rows):
            for col in range(n_cols):
                # Determina larghezza e altezza della tile corrente
                tw = base_width + (extra_width if col == n_cols - 1 else 0)
                th = base_height + (extra_height if row == n_rows - 1 else 0)

                # Calcola offset X, Y includendo overlap
                x_off = col * base_width - overlap
                y_off = row * base_height - overlap

                # Dimensione finestra con overlap su ogni lato
                win_width = tw + 2 * overlap
                win_height = th + 2 * overlap

                # Correggi i bordi (non uscire dal raster)
                if x_off < 0:
                    win_width += x_off
                    x_off = 0
                if y_off < 0:
                    win_height += y_off
                    y_off = 0
                if x_off + win_width > width:
                    win_width = width - x_off
                if y_off + win_height > height:
                    win_height = height - y_off

                window = Window(x_off, y_off, win_width, win_height)

                # Calcola il transform per la finestra
                transform = src.window_transform(window)

                # Leggi i dati della finestra
                data = src.read(window=window)

                # Prepara il profilo per il nuovo file
                profile = src.profile.copy()
                profile.update({
                    'height': int(win_height),
                    'width': int(win_width),
                    'transform': transform
                })
                name, _ = os.path.splitext(filename)
                 # Nome file di output
                out_path = os.path.join(
                    output_dir,
                    f'{name}_r{row}_c{col}.tif'
                )

                # Scrivi la tessera su disco
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data)

                print(f'Written {out_path} (offset={x_off},{y_off}, size={win_width}×{win_height})')
