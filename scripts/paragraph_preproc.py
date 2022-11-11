'''
module docstring placeholder
'''

import json
import argparse
from tqdm import tqdm
from spacy.lang.en import English
from autocorrect import Speller

from src import utils

spell = Speller(lang='en')
spacy_nlp = English()
spacy_nlp.add_pipe('sentencizer')


def process_paragraphs(pars, splits):
    '''
    _summary_

    Args:
        pars (_type_): _description_
        splits (_type_): _description_

    Returns:
        _type_: _description_
    '''

    curr_par_id = 0
    all_images = {}
    all_images['images'] = []

    for img_id, paragraph in tqdm(enumerate(pars, 1)):

        filepath = paragraph['url'].split('/')[-2]
        filename = paragraph['url'].split('/')[-1]
        split = splits[str(filename.split('.')[0])]
        par_spacy = spacy_nlp(paragraph['paragraph'])
        sentences = [sent for sent in par_spacy.sents]
        
        if len(sentences) > 5:
            sentences = sentences[:5]

        sent_dict = {}
        sent_dict['filepath'] = filepath
        sent_dict['filename'] = filename
        sent_dict['imgid'] = img_id
        sent_dict['split'] = split
        sent_dict['sentences'] = []
        par_ids = []

        all_tokens = []
        all_sentences = []
        for raw_sentence in sentences:
            tokens = [token.orth_.lower().rstrip() for token in raw_sentence]
            tokens = [t for t in tokens if t != '']
            # spelling
            tokens = [spell(t) for t in tokens]
            for this_token in tokens:
                all_tokens.append(this_token)
            all_sentences.append(raw_sentence.text)
        tokens_dict = {}
        if len(all_tokens) != 0:
            tokens_dict['tokens'] = all_tokens
            tokens_dict['raw'] = all_sentences
            tokens_dict['imageid'] = img_id
            tokens_dict['parid'] = curr_par_id
            par_ids.append(curr_par_id)
            curr_par_id += 1
            sent_dict['sentences'].append(tokens_dict)

        sent_dict['stanford_par_id'] = int(filename.split('.jpg')[0])
        sent_dict['parids'] = par_ids
        all_images['images'].append(sent_dict)

    return all_images


def main(pars,
         splits_folder,
         outfpath):
    '''
    _summary_

    Args:
        pars (_type_): _description_
        splits_folder (_type_): _description_
        outfpath (_type_): _description_
    '''

    # open data and split files
    paragraph_data = utils.open_file(pars)
    train_split = utils.open_file(splits_folder + 'train_split.json')
    val_split = utils.open_file(splits_folder + 'val_split.json')
    test_split = utils.open_file(splits_folder + 'test_split.json')

    # a bit ugly, but works for now
    full_splits = {}
    for single in train_split:
        full_splits[str(single)] = 'TRAIN'
    for single in val_split:
        full_splits[str(single)] = 'VAL'
    for single in test_split:
        full_splits[str(single)] = 'TEST'

    preprocessed_pars = process_paragraphs(paragraph_data, full_splits)

    with open(outfpath, 'w', encoding='UTF-8') as file_out:
        json.dump(preprocessed_pars, file_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_par_json',
                        default='../data/paragraphs_v1.json',
                        help='original file with stanford paragraphs')
    parser.add_argument('--splits',
                        default='../data/',
                        help='location of the merged paragraph splits')
    parser.add_argument('--out',
                        default='../results/stanford-pars-preproc.json',
                        help='output file with preprocessed paragraphs')
    args = parser.parse_args()

    main(
        args.input_par_json,
        args.splits,
        args.out
    )
