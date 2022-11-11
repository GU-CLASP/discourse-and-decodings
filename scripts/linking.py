'''
Linking noun phrases with objects in images,
requires descriptions of images and object detections.
'''

import json
from glob import glob
import time
import base64
import os
import tqdm

import spacy
import numpy as np
import pandas as pd
from scipy import spatial
from sentence_transformers import SentenceTransformer



def extract_visual_descr(adjs,
                         nouns,
                         adj_conf,
                         noun_conf,
                         vg_objs,
                         vg_attrs):

    """Generate object descriptions from visual detections; with or without control for confidence.
    Args:
        adjs: adjective ids
        nouns: noun ids
        adj_conf: predicted adj confidence scores
        noun_conf: predicted noun confidence scores
        vg_objs: list of visual genome objects
        vg_attrs: list of visual genome adjectives
    Returns:
        original_descriptions - full object descriptions, adj + noun
        a_descriptions - descriptions with examined attributes, (adj) + noun
        an_descriptions - descriptions with examined attributes and nouns, (adj) + (noun)
    p.s. len(an_descriptions) can be naturally < 36
    """

    num = 0
    obj_thresh = 0.4
    attr_thresh = 0.1
    # thresholds are taken from
    # https://github.com/peteanderson80/bottom-up-attention/blob/master/tools/demo.ipynb
    original_descriptions = []
    a_descriptions = []
    an_descriptions = []
    for attr, noun in zip(adjs, nouns):
        attr_conf = adj_conf[num]
        obj_conf = noun_conf[num]
        # check against confidence thresholds for attributes and nouns
        if attr_conf < attr_thresh:
            this_phrase = str(vg_objs[noun + 1])
            a_descriptions.append((num, this_phrase))
            if obj_conf >= obj_thresh:
                this_phrase = str(vg_objs[noun + 1])
                an_descriptions.append((num, this_phrase))
        else:
            this_phrase = str(vg_attrs[attr + 1]) + ' ' + str(vg_objs[noun + 1])
            a_descriptions.append((num, this_phrase))
            if obj_conf >= obj_thresh:
                this_phrase = str(vg_attrs[attr + 1]) + ' ' + str(vg_objs[noun + 1])
                an_descriptions.append((num, this_phrase))
        this_phrase = str(vg_attrs[attr + 1]) + ' ' + str(vg_objs[noun + 1])
        original_descriptions.append((num, this_phrase))
        num += 1
    return (
           original_descriptions,
           a_descriptions,
           an_descriptions
    )



def exclude_room_types(nps):

    """Remove object descriptions which correspond to room types (from text descriptions).
    Args:
        nps: original noun phrases
    Returns:
        rt_descriptions: descriptions which were mapped with room types
        nps_f: filteres noun phrases, no room type descriptions
    """

    # room types are taken from the original tell-me-more paper
    room_types = ['image', 'room', 'picture', 'photo', 'area', 'apartment', 'areas',
                  'middle', 'left', 'right', 'front', 'background', 'foreground', 'corner',
                  'bottom', 'top', 'center', 'indoor', 'outdoor', 'attic',
                  'basement', 'bathroom', 'bedroom', 'coop', 'dinette',
                  'home','garage', 'kennel', 'kitchen', 'kitchenette',
                  'laundromat', 'nursery', 'pantry',
                  'parlor', 'playroom', 'poolroom', 'shower', 'staircase', 'bar', 'cellar',
                  'storage', 'hostel', 'living']
    rt_descriptions = []
    nps_filtered = []
    for (npi, single_np) in nps:
        np_check = single_np.split()
        found = False
        for word in np_check:
            if word in room_types:
                found = True
        if found is False:
            nps_filtered.append((npi, single_np))
        else:
            rt_descriptions.append(single_np)
    return (
           rt_descriptions,
           nps_filtered
    )



def extract_text_descr(search_res,
                       #sequences_path,
                       imid):

    """Generate object descriptions from tell-me-more texts.
    Args:
        search_res: results from one of the decoding searches
        # sequences_path: path to .csv with tell-me-more texts
        imid: image_id from ADE20k
    Returns:
        rt_descriptions: phrases considered to be room types
        nps_out: noun phrases detected in text
    """

    #sequences = pd.read_csv(sequences_path, sep='\t', index_col=0)
    #paragraph = sequences[sequences.image_id == int(imid)].iloc[:, 5:11]
    #text = [paragraph['d1'].item(),
    #    paragraph['d2'].item(),
    #    paragraph['d3'].item(),
    #    paragraph['d4'].item(),
    #    paragraph['d5'].item()
    #        ]
    #full_text = ' '.join(text)
    full_text = search_res[str(imid)]['caption']
    # extract noun phrases; control for quality by checking the phrase head
    doc = spacy_nlp(full_text)
    noun_chunks = []
    # modification to know which noun phrase appears in which sentence
    sent_boundaries = []
    for sent in doc.sents:
        sent_boundaries.append((sent.start_char, sent.end_char))
    for noun_ch in doc.noun_chunks:
        nc_span = (noun_ch.start_char, noun_ch.end_char)
        append = False
        phrase = noun_ch.text
        subwords = spacy_nlp(phrase)
        for k in subwords:
            if k.head.pos_ == 'NOUN':
                append = True
        if append:
            for sbi, sbound in enumerate(sent_boundaries):
                if sbound[0] <= nc_span[0] <= sbound[1] and sbound[0] <= nc_span[1] <= sbound[1]:
                    noun_chunks.append((sbi, phrase))
    # remove text descriptions which are room types (e.g., bathroom)
    rt_descriptions, nps_out = exclude_room_types(noun_chunks)
    return (
           rt_descriptions,
           nps_out
    )



def encode_with_st(texts):

    """Encode object descriptions (visual) with Sentence Transformer.
    Args:
        texts: list of object descriptions
    Returns:
        embs: embeddings of object descriptions
    """

    embs = {}
    for (num, word) in texts:
        embs[num] = model.encode(word)
    return embs



def run_no_links_check(links):

    """Identify noun phrases which were linked to NONE, e.g. no object.
    Args:
        links: input set of links, e.g. {np: obj}
    Returns:
        no_links: list of noun phrase ids which were not linked to anything
    """

    no_links = []
    for k, val in links.items():
        if isinstance(val, list):
            if val[0] == 'NONE':
                no_links.append(k)
        elif isinstance(val, str):
            if val == 'NONE':
                no_links.append(k)
    return no_links



def run_plural_check(noun_phrase,
                     np_id,
                     linked_sorted,
                     descriptions):

    """Function for linking with plural forms.
    Args:
        noun_phrase: noun phrase to compare against object descriptions
        np_id: id of the noun phrase, required for correct mappings
        linked_sorted: dict of linked objects with noun phrase and their cosines
        descriptions: object descriptions
    Returns:
        linked_plural: updated list with links taking plurals into account
    """

    plural_check = {}
    np_doc = spacy_nlp(noun_phrase)
    for token in np_doc:
        if token.dep_ == 'ROOT':
            if token.tag_ == 'NNS':
                lemma_singular = token.lemma_
                prelim_plural = []
                for ind in list(linked_sorted.keys()):
                    if ind != 'NONE':
                        this_obj_label = [k[1] for k in descriptions if k[0] == ind][0]
                        object_doc = spacy_nlp(this_obj_label)
                        for token in object_doc:
                            if token.pos_ == 'NOUN':
                                # if singular forms match, then append multiple
                                if token.text == lemma_singular:
                                    prelim_plural.append(ind)
                        # if no match between singular forms,
                        # append the first object only (standard no plural case)
                        #if prelim_plural == []:
                        if not prelim_plural:
                            prelim_plural.append(list(linked_sorted.keys())[0])
                        plural_check[np_id] = prelim_plural
                    else:
                        plural_check[np_id] = 'NONE'
            # else, singular form
            else:
                plural_check[np_id] = [list(linked_sorted.keys())[0]]
    return plural_check



def link_main(np_id,
              target_np,
              np_emb,
              obj_embs,
              obj_descrs):

    """Function for linking with plural forms.
    Args:
        np_id: id of the noun phrase, required for correct mappings
        target_np: current noun phrase
        np_emb: embedding of the noun phrase
        obj_embs: embeddings of object descriptions
        obj_descrs: current object descriptions
    Returns:
        linked: result of linking without plurals
        linked_plurals: result of linking with plurals
    """

    linked_interm = {}
    linked_res = {}
    for obj_id, obj_emb in obj_embs.items():
        cos_sim = 1 - spatial.distance.cosine(np_emb, obj_emb)
        if cos_sim > 0.5:
            linked_interm[obj_id] = cos_sim
    #if linked_interm != {}:
    if linked_interm:
        linked_sorted = {k: v for k, v in sorted(linked_interm.items(),
                                                 key=lambda item: item[1], reverse=True)}
        linked_res[np_id] = list(linked_sorted.keys())[0]
    else:
        linked_sorted = {'NONE': 'NONE'}
        linked_res[np_id] = 'NONE'
    linked_plurals = run_plural_check(target_np,
                                      np_id,
                                      linked_sorted,
                                      obj_descrs)
    return (
           linked_res,
           linked_plurals
    )



def linking(descr_o,
            descr_a,
            descr_al,
            noun_phrase_filtered):

    """General function to run linking between noun phrases and object descriptions.
    Args:
        descr_o: attribute + noun object descriptions
        descr_a: (attribute) + noun object descriptions
        descr_al: (attribute) + (noun) object descriptions
        nps_f: noun phrases from texts, filtered for room types, indexed as {id: np}
    Returns:
        results of linking for all possible configurations
    """

    lnked = {}
    lnked_plural = {}
    lnked_attr = {}
    lnked_attr_plural = {}
    lnked_attr_noun = {}
    lnked_attr_noun_plural = {}
    emb_vis_o = encode_with_st(descr_o)
    emb_vis_a = encode_with_st(descr_a)
    emb_vis_al = encode_with_st(descr_al)
    # main linking loop
    for npi, (_, noun_phrase) in enumerate(noun_phrase_filtered):
        np_embedding = model.encode(noun_phrase)
        # attribute + noun
        linked_o, linked_o_pl = link_main(npi,
                                          noun_phrase,
                                          np_embedding,
                                          emb_vis_o,
                                          descr_o)
        lnked.update(linked_o)
        lnked_plural.update(linked_o_pl)
        # (attribute) + noun
        linked_a, linked_a_pl = link_main(npi,
                                          noun_phrase,
                                          np_embedding,
                                          emb_vis_a,
                                          descr_a)
        lnked_attr.update(linked_a)
        lnked_attr_plural.update(linked_a_pl)
        # (attribute) + (noun)
        linked_ao, linked_ao_pl = link_main(npi,
                                            noun_phrase,
                                            np_embedding,
                                            emb_vis_al,
                                            descr_al)
        lnked_attr_noun.update(linked_ao)
        lnked_attr_noun_plural.update(linked_ao_pl)
    return (
           lnked,
           lnked_plural,
           lnked_attr,
           lnked_attr_plural,
           lnked_attr_noun,
           lnked_attr_noun_plural
    )


if __name__ == '__main__':

    spacy_nlp = spacy.load('en_core_web_sm')
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    # image, feature, data directories
    DATA_PATH = '../../tell-me-more-cups-attention/data/tell-me-more/1600-400-20'
    FEAT_DIR = '/scratch/nikolai/tmm_dataset/frcnn_tellmemore/'
    #IMAGE_DIR = '/scratch/nikolai/tmm_dataset/tell_me_more/'
    #IDS_FILE = '../../tell-me-more-cups-attention/corpus/tell-me-more/image-description-sequences/data/sequences.csv'
    # paths for linking of different searches
    SPLITS_FILE = '../src/object_relation_transformer/data/tmmtalk.json'

    SEARCH_TYPE = 'sampling'
    SEARCH_CONF = '_t050'

    SEARCH_RES_BASE = f'/home/xilini/gem2022/results/{SEARCH_TYPE}/'
    # number of objects detected in images
    res = {}
    # load features and images
    #feats = glob(FEAT_DIR + '*.npz')
    #images = glob(IMAGE_DIR + '*.jpg')

    # Load classes
    genome_objects = ['__background__']
    with open(os.path.join(DATA_PATH, 'objects_vocab.txt'), encoding='UTF-8') as f:
        for obj in f.readlines():
            genome_objects.append(obj.split(',')[0].lower().strip())
    # Load attributes
    genome_attributes = ['__no_attribute__']
    with open(os.path.join(DATA_PATH, 'attributes_vocab.txt'), encoding='UTF-8') as f:
        for att in f.readlines():
            genome_attributes.append(att.split(',')[0].lower().strip())

    # gem2022
    # modification 1. get image id for the *test set* only, perform linking for them
    # modification 2. change function 'extract_text_descr' so that it takes captions from different searches, not the ground-truth
    with open(SPLITS_FILE, 'r') as f:
        splits = json.load(f)
    splits = splits['images']
    test_image_ids = []
    test_feats = []
    test_images = []
    for img in splits:
        if img['split'] == 'test':
            if 'train' in img['file_path']:
                imgid = str(img['file_path'].split('_')[-1].split('.jpg')[0].lstrip('0'))
            else:
                imgid = int(img['file_path'].split('_')[-1].split('.jpg')[0].lstrip('0'))
                imgid = str(100000 + imgid)
            test_image_ids.append(imgid)
            test_feats.append(f'{FEAT_DIR}{imgid}.npz')
            test_images.append(img['file_path'])
    with open(SEARCH_RES_BASE + SEARCH_TYPE + SEARCH_CONF + '.json', 'r') as f:
    #with open(SEARCH_RES_BASE + 'diversebeam2_group2_lambda050.json', 'r') as f:
        search_res = json.load(f)
    search_res = search_res['imgToEval']

    #for single_image in tqdm.tqdm(feats):
    for single_image in tqdm.tqdm(test_feats):
        feat_loaded = np.load(single_image)
        image_id = single_image.split('/')[-1].split('.npz')[0]
        objects_id = np.frombuffer(base64.b64decode(feat_loaded['objects_id']),
                                                    dtype=np.int64).copy()
        objects_conf = np.frombuffer(base64.b64decode(feat_loaded['objects_conf']),
                                                      dtype=np.float32).copy()
        attrs_id = np.frombuffer(base64.b64decode(feat_loaded['attrs_id']),
                                                  dtype=np.int64).copy()
        attrs_conf = np.frombuffer(base64.b64decode(feat_loaded['attrs_conf']),
                                                    dtype=np.float32).copy()
        # generate object descriptions from visual detections
        descr_original, descr_attr, descr_attr_label = extract_visual_descr(attrs_id,
                                                                            objects_id,
                                                                            attrs_conf,
                                                                            objects_conf,
                                                                            genome_objects,
                                                                            genome_attributes)
        # generate object descriptions from texts (paragraphs)
        #rt_words, nps_f = extract_text_descr(IDS_FILE, image_id)
        rt_words, nps_f = extract_text_descr(search_res, image_id)
        (
        linked, linked_plural,
        linked_attr, linked_attr_plural,
        linked_attr_noun, linked_attr_noun_plural
        ) = linking(descr_original,
                    descr_attr,
                    descr_attr_label,
                    nps_f)
        linked_none = run_no_links_check(linked)
        linked_none_plural = run_no_links_check(linked_plural)
        linked_attr_none = run_no_links_check(linked_attr)
        linked_attr_none_plural = run_no_links_check(linked_attr_plural)
        linked_attr_noun_none = run_no_links_check(linked_attr_noun)
        linked_attr_noun_none_plural = run_no_links_check(linked_attr_noun_plural)
        res[image_id] = [descr_original, descr_attr, descr_attr_label,
                         nps_f,
                         linked, linked_plural,
                         linked_attr, linked_attr_plural,
                         linked_attr_noun, linked_attr_noun_plural,
                         linked_none, linked_none_plural,
                         linked_attr_none, linked_attr_none_plural,
                         linked_attr_noun_none, linked_attr_noun_none_plural,
                         rt_words]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #with open(f'../results/linking_{SEARCH_TYPE}_run_{timestr}.json', 'w', encoding='UTF-8') as f5:
    with open(f'../results/{SEARCH_TYPE}/linking_{SEARCH_TYPE}{SEARCH_CONF}_run_{timestr}.json', 'w', encoding='UTF-8') as f5:
        json.dump(res, f5)
