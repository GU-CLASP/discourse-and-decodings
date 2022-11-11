'''
Script to create attention heatmaps over attended (linked) objects,
heatmaps are created per sentence and combined across all images,
we use alpha blending that allows for better visualisations of such blending.
'''

import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm

# choose a sentence for which you want to create a heatmap, from s1 to s5
SENT_NUM = 's1'
# also specify location of the individual image heatmaps
SEARCH = 'nucleus'
SEARCH_CONF = '095'

ATTN_MAPS = f'../results/{SEARCH}/attn-imgs-{SEARCH_CONF}'
#ATTN_MAPS = f'../results/ref-attn-imgs'

def blend(list_images):

    """Place individual attention heatmaps on top of each other,
    create a single map for all images.
    Args:
        list_images: set of resized heatmaps for each image
    Returns:
        out: final attention heatmap for all images for the given sentence
    """

    out = np.zeros_like(list_images[0])
    alpha_1 = 0.9
    alpha_2 = 0.1
    for img in tqdm.tqdm(list_images):
        out = (alpha_1 * out) + (alpha_2 * img)
    #output = output / (alpha_1 + alpha_2)
    out = out.astype(np.uint8)
    return out

images = glob.glob(f'./{ATTN_MAPS}/{SENT_NUM}*')
images = [cv2.resize(cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB), (600, 600)) for i in images]
print(len(images))
output = blend(images)

im = Image.fromarray(output)
LEFT = 0
UPPER = 100
RIGHT = LEFT + 600
LOWER = UPPER + 400
im_cropped = im.crop((LEFT, UPPER, RIGHT, LOWER))
plt.imshow(np.asarray(im_cropped))
im_cropped.save(f'../results/{SEARCH}/attn-maps-{SEARCH_CONF}/hm-{SENT_NUM}.pdf', format='PDF', quality=600)
#im_cropped.save(f'../results/ref-attn-maps/hm-{SENT_NUM}.pdf', format='PDF', quality=600)
