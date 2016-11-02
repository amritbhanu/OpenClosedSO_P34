from __future__ import print_function, division
__author__ = 'amrit'

import csv
from Preprocess import *
import sys

sys.dont_write_bytecode = True

if __name__ == "__main__":
    Y = []
    features = []
    with open('/Users/amrit/Downloads/train.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        headers=reader.fieldnames[:6]+['Text']+reader.fieldnames[13:]
        with open('train.csv', 'w') as csvfile1:
            fieldnames = ['first_name', 'last_name']
            writer = csv.DictWriter(csvfile1, fieldnames=headers)
            writer.writeheader()
            for row in reader:
                str=row['Title']+' ' + row['BodyMarkdown']+' ' +row['Tag1']+' '+row['Tag2']+' '+row['Tag3']+' '+ row['Tag4']+' '+row['Tag5']
                line = process(str, string_lower, email_urls, unicode_normalisation, punctuate_preproc,
                                   numeric_isolation, stopwords, stemming, word_len_less)
                dic={}
                for x in headers:
                    if x=='Text':
                        dic[x]=line
                    else:
                        dic[x]=row[x]
                writer.writerow(dic)
