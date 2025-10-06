from dataset import add_files_and_masks

output_dir = '/leonardo_work/PHD_cozzani/seg_solar/dataset/solardk_segformer/danish_google_noherlev/gentofte_trainval/train/'
input_dir = 'google/'
add_files_and_masks(6000, input_dir, output_dir)

