# fa la georeferenzazione di .tif usando file .prj (e copia i .tif in nuova
# cartella

mkdir -p /leonardo_work/PHD_cozzani/seg_solar/dataset/unsupervised_bologna20802

for tif in /leonardo_work/DTbo_DTBO-HPC/Data/BOLOGNA_2022/TAVOLE_RGBN/*.tif; do
    base=$(basename "$tif" .tif)
    echo "Processing: $base"
    
    # I TFW sono già nella stessa cartella, usa il PRJ senza suffisso
    gdal_translate -of GTiff \
        -a_srs /leonardo_work/DTbo_DTBO-HPC/Data/BOLOGNA_2022/Bologna_2022_RGB_5cm.prj \
        "$tif" "/leonardo_work/PHD_cozzani/seg_solar/dataset/unsupervised_bologna20802/${base}.tif"
done
