from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import random


max_rows = 5
max_cols = 10
image_dir = r"images\-all_molecules.pkl-150000-True-True-5-4" \
            r"-9-multi_gcn_feedback-0.001-1-0.5-False-True-[32, 32]-['mean', 'mean']-[True, " \
            r"True]-True-2-32-mean-0.0-100-10-100-False-False-False\final\upsampled"

filenames = []
for fname in os.listdir(image_dir):
    if fname.endswith('.png'):
        filenames.append((os.path.join(image_dir, fname), 'invalid'))
for fname in os.listdir(os.path.join(image_dir, 'valid')):
    if fname.endswith('.png'):
        filenames.append((os.path.join(image_dir, 'valid', fname), 'valid'))
for fname in os.listdir(os.path.join(image_dir, 'accurate')):
    if fname.endswith('.png'):
        filenames.append((os.path.join(image_dir, 'accurate', fname), 'accurate'))

print(len(filenames))
images = random.sample(filenames, max_rows * max_cols)

fig, axes = plt.subplots(nrows=max_rows, ncols=max_cols, squeeze=True, figsize=(max_cols, max_rows))
for idx, (image_path, validity) in enumerate(images):
    row = idx // max_cols
    col = idx % max_cols
    axes[row, col].axis("off")
    if validity == 'invalid':
        axes[row, col].imshow(mpimg.imread(image_path)[:, :, 0], cmap='Reds_r')
    elif validity == 'valid':
        axes[row, col].imshow(mpimg.imread(image_path)[:, :, 0], cmap='Blues_r')
    else:
        axes[row, col].imshow(mpimg.imread(image_path))
plt.subplots_adjust(wspace=.0, hspace=.0)
plt.savefig('output.png', dpi=1200)
