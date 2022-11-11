'''
Quick formatting of the linking result,
just naming and grouping into dictionary structure that makes sense
'''

import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r',
                        '--results_file',
                        help='Path to the file with formatted results of linking',
                        default='./res_linking_run-20220813-162422.json',
                        required=False)
    parser.add_argument('-o',
                        '--output_file',
                        help='Path to save formatted file with linking',
                        default='./res_formatted_run-20220813-162422.json',
                        required=False)
    args = vars(parser.parse_args())

    with open(args['results_file'], 'r', encoding='UTF-8') as f:
        links = json.load(f)
    cleaned = {}
    for k, v in links.items():
        cleaned[k] = {}
        cleaned[k]['NPS-OBJ'] = v[3]
        cleaned[k]['NPS-RT'] = v[16]
        cleaned[k]['AN'] = v[0]
        cleaned[k]['(A)N'] = v[1]
        cleaned[k]['(A)(N)'] = v[2]
        cleaned[k]['L-AN-1-1'] = v[4]
        cleaned[k]['L-AN-1-M'] = v[5]
        cleaned[k]['L-(A)N-1-1'] = v[6]
        cleaned[k]['L-(A)N-1-M'] = v[7]
        cleaned[k]['L-(A)(N)-1-1'] = v[8]
        cleaned[k]['L-(A)(N)-1-M'] = v[9]
    with open(args['output_file'], 'w', encoding='UTF-8') as f1:
        json.dump(cleaned, f1)
