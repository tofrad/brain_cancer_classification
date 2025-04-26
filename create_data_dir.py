#fusion of dataset directories
import os
import shutil

data_dir = "dataset"

for folder in os.listdir(data_dir):
    if ' ' in folder:  #get rid of the T1, T1C+, T2 etc. labels because they only classify the method the mri was taken with
        new_name = folder.split(' ')[0]
        src = os.path.join(data_dir, folder)
        dst = os.path.join(data_dir, new_name)

        if not os.path.exists(dst):
            os.makedirs(dst)

        for file in os.listdir(src):
            src_path = os.path.join(src, file)
            base, ext = os.path.splitext(file)
            dst_file = f"{base}_{folder.split(' ')[1]}{ext}"  # Z. B. "image1_T1.png"
            dst_path = os.path.join(dst, dst_file)

            shutil.move(src_path, dst_path)

        os.rmdir(src)
