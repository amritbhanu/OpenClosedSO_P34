from __future__ import print_function, division
import sys, os
sys.dont_write_bytecode = True
sys.path.append(os.path.abspath("."))
import csv
from Preprocess import *

import pickle

__author__ = 'amrit'


def pre_process(src, dst):
    Y = []
    features = []
    with open(src) as csvfile:
        reader = csv.DictReader(csvfile)
        headers = reader.fieldnames
        with open(dst, 'w') as csvfile1:
            writer = csv.DictWriter(csvfile1, fieldnames=headers)
            writer.writeheader()
            for i, row in enumerate(reader):
                print(i)
                text = row['Title']+' ' + row['BodyMarkdown']+' ' +row['Tag1']+' '+row['Tag2']+' '+row['Tag3']+' '+ row['Tag4']+' '+row['Tag5']
                # text = row['Title'] + ' ' + row['BodyMarkdown']
                line = process(text, string_lower, email_urls, unicode_normalisation, punctuate_preproc,
                               numeric_isolation, stopwords, stemming, word_len_less)
                dic = {}
                for x in headers:
                    if x == 'BodyMarkdown':
                        dic[x] = line
                    else:
                        dic[x] = row[x]
                writer.writerow(dic)


def dump_pkl(file_name, dump_name):
    with open(file_name) as csv_file:
        reader = csv.DictReader(csv_file)
        headers = reader.fieldnames
        pkl_data = {"text": [], "label": []}
        for i, row in enumerate(reader):
            pkl_data["text"].append(row['BodyMarkdown'])
            pkl_data["label"].append(row['OpenStatus'])
    with open(dump_name, 'wb') as dump_f:
        pickle.dump(pkl_data, dump_f)


def load_pkl(file_name):
    with open(file_name) as pkl_file:
        return pickle.load(pkl_file)


if __name__ == "__main__":
    # pre_process('data/train-sample.csv', 'data/train_mini.csv')
    dump_pkl('data/train.csv', 'data/train.pkl')
    # print(load_pkl('data/train_mini.pkl')['text'])

