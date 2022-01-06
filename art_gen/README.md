# Anime.ai - Art Work generation
The trained styleganv2 checkpoint can be found at https://drive.google.com/file/d/1Whxjt3-cVIESyASJz22ZGmcf7jchiRB0/view?usp=sharing

We used https://github.com/lucidrains/stylegan2-pytorch to train and generate the samples

# Future Improvements

The generator can succesfully capture the characteristics of the specific character Reimu Hakurei. However, the dataset is very noisy and sparse, as the images are from different artists and have different styles. This makes the generator hard to learn a specific style to generate a "good-looking" face and body while keeping the characteristics. This is the main difference from generating human-like face and animation characters. Some future works can be done by data cleaning, data grouping, background removing and possibly tuning the model architecture.