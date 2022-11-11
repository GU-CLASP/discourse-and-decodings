'''
Create attention heatmaps per paragraph sentence for all images,
heatmaps are created with objects which were linked with noun phrase,
one can choose which linking method to take and visualise (check parameters in the main loop).
'''

import argparse
import base64
import json
import tqdm
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2

import spacy
spacy_nlp = spacy.load('en_core_web_sm')



def image_box_resize(image,
                     bboxes):

    """Resize bounding boxes from original coordinates to the scaled ones.
    Args:
        image: original image
        bboxes: (x, y, xmax, ymax) coordinates of bounding boxes
    Returns:
        img: original image, resized (scaled)
        boxes_scaled: newly scaled bounding boxes
    """

    boxes_scaled = []
    image_to_show = cv2.imread(image, 3)
    y_dim = image_to_show.shape[0]
    x_dim = image_to_show.shape[1]
    target_size = 2000
    x_scale = target_size / x_dim
    y_scale = target_size / y_dim
    img = cv2.resize(image_to_show, (target_size, target_size))
    img = np.array(img)
    for box in bboxes:
        origleft, origtop, origright, origbottom = box[0], box[1], box[2], box[3]
        x_scaled = int(np.round(origleft * x_scale))
        y_scaled = int(np.round(origtop * y_scale))
        xmax = int(np.round(origright * x_scale))
        ymax = int(np.round(origbottom * y_scale))
        boxes_scaled.append([x_scaled, y_scaled, xmax, ymax])
    return (
            img,
            boxes_scaled
    )




def image_vis(this_image,
              boxes_to_visualise,
              all_boxes,
              sent_id,
              image_path,
              save_path):

    """Visualisation of linked bounding boxes on top of the image.
    Args:
        this_image: image id
        boxes_to_visualise: ids of linked bounding boxes
        all_boxes: coordinates of original bounding boxes
        sent_id: current sentence id in the paragraph
        image_path: path to ADE20k images
    Returns:
        saves heatmaps per sentence per image
    """

    # open correct image; val image ids are > 100000
    this_image = int(this_image)
    if this_image > 100000:
        this_image = this_image - 100000
        val_this_image = "%08d" % (this_image,)
        val_this_image = f'ADE_val_{str(val_this_image)}.jpg'
        image = image_path + str(val_this_image)
    else:
        train_this_image = "%08d" % (this_image,)
        train_this_image = f'ADE_train_{str(train_this_image)}.jpg'
        image = image_path + str(train_this_image)
    boxes_filtered = [all_boxes[k] for k in boxes_to_visualise]
    # transform last two values in each box into xmax and ymax
    transformed_boxes = []
    for (x_coord, y_coord, width, height) in boxes_filtered:
        xmax = x_coord + width
        ymax = y_coord + height
        transformed_boxes.append([x_coord, y_coord, xmax, ymax])
    # adjust bounding boxes based on the resized image
    img_rescaled, boxes_rescaled = image_box_resize(image, transformed_boxes)
    # controlling how stretched the bounding box should be
    figure(figsize=(12, 18), dpi=80)
    plt.axis('off')
    plt.tight_layout()
    img = Image.fromarray(img_rescaled)
    white_img = 255 * np.ones((img.size[1], img.size[0] , 3), np.uint8)
    plt.imshow(white_img)
    for bbox in boxes_rescaled:
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          (bbox[2] - bbox[0]) - bbox[0],
                          (bbox[3] - bbox[1]) - bbox[1], fill=True,
                          linewidth=2, alpha=1, color='#00008B')
                )

    sid = str(int(sent_id) + 1)
    plt.savefig(f'{save_path}/s{sid}-' + str(this_image) + '.jpg')
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--feat_path',
                        help='Path to the image features',
                        default='/scratch/nikolai/tmm_dataset/frcnn_tellmemore/',
                        required=False)
    parser.add_argument('-i',
                        '--image_path',
                        help='Path to the tell me more images',
                        default='/scratch/nikolai/tmm_dataset/tell_me_more/',
                        required=False)
    parser.add_argument('-l',
                        '--linking_method_type',
                        help='Choose linking method to visualise;\
                            full set can be found in res_formatted.json',
                        default='L-(A)(N)-1-M',
                        required=False)
    parser.add_argument('-m',
                        '--filter_method',
                        help='Pick a filtering method that was used with linking,\
                            they should be identical',
                        default='(A)(N)',
                        required=False)
    parser.add_argument('-r',
                        '--results_file',
                        help='Path to the file with formatted results of linking',
                        default='./res_formatted_run-20220813-162422.json',
                        required=False)
    parser.add_argument('-o',
                        '--output_path',
                        help='Path to save heatmaps for all images for the specific sentence id',
                        default='./where',
                        required=False)
    args = vars(parser.parse_args())

    with open(args['results_file'], 'r', encoding='UTF-8') as a:
        links = json.load(a)

    for num, (image_id, v) in tqdm.tqdm(enumerate(links.items())):
        feat_file = args['feat_path'] + str(image_id) + '.npz'
        feat_loaded = np.load(feat_file)
        boxes = np.frombuffer(base64.b64decode(feat_loaded['boxes']),
                                               dtype=np.float32).reshape(36, 4).copy()
        nps = v['NPS-OBJ']
        objs = v[f'{args["linking_method_type"]}']
        # per sentence
        for sentid in range(5):
            nouns = [(iid, i[1]) for iid, i in enumerate(nps) if i[0] == sentid]
            noun_ids = [iid for iid, i in nouns]
            objids = []
            for p in noun_ids:
                if isinstance(objs[str(p)], list):
                    for pp in objs[str(p)]:
                        if pp != 'NONE':
                            objids.append(pp)
                else:
                    if objs[str(p)] != 'NONE':
                        objids.append(objs[str(p)])
            boxes_to_show = [k for (k, kk) in v[f'{args["filter_method"]}']]
            boxes_to_show_for_sent = [i for i in boxes_to_show if i in objids]
            image_vis(image_id,
                      boxes_to_show_for_sent,
                      boxes,
                      sentid,
                      args['image_path'],
                      args['output_path'])
