# remove mp4, zip, gif... and all files cannot read
# transform all images to jpg
# rename the folders
# parse into train/test
import pathlib
import os
import random
import shutil
from PIL import Image
directory_name_mapping = {
    "gokou_ruri": "Ruri Gokou",
    "misaka_mikoto": "Mikoto Misaka",
    "remilia_scarlet": "Remilia Scarlet",
    "shana": "Shana",
    "tohsaka_rin": "Rin Tohsaka",
    "chartags 1": "Others",
}

def read_tags(tag_file_name="tags.txt"):
    batchdir = pathlib.Path(pathlib.Path(__file__).absolute().parent)

    rulefile = batchdir.joinpath(tag_file_name)
    with open(rulefile, "r", encoding="utf-8") as f:
        tags = f.read().splitlines()
    return batchdir, tags
    
def main():
    batchdir, tags = read_tags()
    tags = ["chartags 1"]
    
    for tag in tags:
        directory_path = batchdir.joinpath(tag)
        file_dir_path = directory_path.joinpath("danbooru.donmai.us")
        files = os.listdir(file_dir_path)
        count = 0
        for i, f in enumerate(files):
            f = file_dir_path.joinpath(f)
            print(f)
            if str(f).endswith(".zip") or str(f).endswith(".gif") or str(f).endswith(".mp4"):
                os.remove(f)
                continue
            try:
                image = Image.open(f)
                image = image.convert('RGB')
                image.save(os.path.join(directory_path, f"{tag}_{count}.jpg"))
                count += 1
            except Exception as e:
                print(e, "removed")
                os.remove(f)
        # rename the folders
        new_dir_path = batchdir.joinpath(directory_name_mapping[tag])
        os.rename(directory_path, new_dir_path)

def make_train_test(test_split=0.2):
    batchdir, tags = read_tags()
    train_path = batchdir.joinpath("train")
    test_path = batchdir.joinpath("test")
    try:
        os.makedirs(train_path)
        os.makedirs(test_path)
    except:
        raise

    tags = ["chartags 1"]

    for t in tags:
        dir_path = batchdir.joinpath(directory_name_mapping[t])
        train_path_tag = train_path.joinpath(directory_name_mapping[t])
        test_path_tag = test_path.joinpath(directory_name_mapping[t])
        files = os.listdir(dir_path)
        test_len = int(len(files) * test_split)
        train_len = len(files) - test_len
        random.shuffle(files)
        train_fs = files[:train_len].copy()
        test_fs = files[train_len:].copy()
        os.makedirs(train_path_tag)
        os.makedirs(test_path_tag)
        for f in train_fs:
            old_path = dir_path.joinpath(f)
            new_path = train_path_tag.joinpath(f)
            shutil.copy(old_path, new_path)
        for f in test_fs:
            old_path = dir_path.joinpath(f)
            new_path = test_path_tag.joinpath(f)
            shutil.copy(old_path, new_path)
        
#main()
make_train_test()