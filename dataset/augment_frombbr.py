from random import sample
import os
import shutil
def add_files(N):
    filenames=[filename for filename in os.listdir(input_dir)]   
    filenames_da_inserire = sample(filenames,N)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print("Directory already exists, files will be added to it.")
    print("Adding {} files to {}".format(len(filenames_da_inserire), output_dir))
    # Copy the selected files to the output directory
    for filename in filenames_da_inserire:
        shutil.copy(os.path.join(input_dir,filename),output_dir)
        
output_dir = '/leonardo_work/PHD_cozzani/danish_dataset2/solardk_dataset_neurips_v2/gentofte_trainval/train/positive/'
input_dir = '/leonardo_work/PHD_cozzani/danish_dataset2/solardk_dataset_neurips_v2/bbr/positive'
add_files(3000)
