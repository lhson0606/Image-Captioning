import torch

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoints_dir = "checkpoints"
checkpoint = f"{checkpoints_dir}/best_model.pth.tar"
data_dir = "_data/flickr8k"
data_name = "flickr8k_5_cap_per_img_5_min_word_freq"
word_map_file = f'{data_dir}/WORDMAP_flickr8k_5_cap_per_img_5_min_word_freq.json'