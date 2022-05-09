# Anime.ai
a deep learning models to work with anime characters (recognition and generation) 

# Classification Dataset

https://drive.google.com/drive/folders/12OlnNxpwvB1F2Ue4KbxiL2ZL0xCiCtWZ?usp=sharing (with train/test split, 11 characters including "others", each character has approximately 1000 full-body images in train and 250 images in test)

extract after downloading and put `train` and `test` under `dataset`

The images are retrieved from https://danbooru.donmai.us/

# Classification Models

ResNet18 (with ImageNet Pretrained Weights) can achieve 0.948 accuracy score on classifying 10 characters. Checkpoint: https://drive.google.com/drive/u/6/folders/1Re-oBKIB_Trp2GCDkDM-iZTOl5UHPaOb

# Dataset for Specific Characters

We made few more datasets with more images for some characters. https://drive.google.com/drive/folders/1Q6vBmHF-3ZenlP9nETkFdNyTP_GDcXbd?usp=sharing
- Miku Hatsune (5996 images)
- Reimu Hakurei (3957 images)
- Remilia Scarlet (7198 images)
- Characters other than Miku (19055 images)

The images are retrieved from https://danbooru.donmai.us/

# Generate New Art Works (Images) for Specific Characters

We experimented training a stylegan2 using https://github.com/lucidrains/stylegan2-pytorch on one of the above datasets to
generate new images (see `art_gen`).

# Future Improvements

- Fine-tune the classfication models, and apply them into videos
- Experiment with GANs on character datasets, try making improvements on generated results with classfication models
- Develope a pipeline that automatically search and filter new anime art works on the internet and improve the generator.
