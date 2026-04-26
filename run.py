import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, glob

TRAIN_DIR = 'dataset/DevanagariHandwrittenCharacterDataset/Train'
folders = sorted(os.listdir(TRAIN_DIR))[:3]

fig, axes = plt.subplots(3, 5, figsize=(15,9))
for i, folder in enumerate(folders):
    imgs = glob.glob(os.path.join(TRAIN_DIR, folder, '*.png'))[:5]
    for j, p in enumerate(imgs):
        img = cv2.imread(p, 0)
        axes[i][j].imshow(img, cmap='gray')
        axes[i][j].set_title(folder, fontsize=7)
        axes[i][j].axis('off')

plt.tight_layout()
plt.savefig('dataset_sample.png')
print('saved dataset_sample.png')