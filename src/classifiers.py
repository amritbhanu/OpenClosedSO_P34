from __future__ import print_function, division

import sys
import os
sys.path.append(os.path.abspath("."))
sys.dont_write_bytecode = True
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from SE_preprocess import load_pkl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pickle
from features import *
from demos import *

def naive_bayes(train_inp, train_out, test_inp):
    model = MultinomialNB().fit(train_inp, train_out)
    predicted = model.predict(test_inp)
    return model, predicted


def lin_svm(train_inp, train_out, test_inp):
    model = SVC(kernel='linear').fit(train_inp, train_out)
    predicted = model.predict(test_inp)
    return model, predicted


def rbf_svm(train_inp, train_out, test_inp):
    model = SVC(kernel='rbf').fit(train_inp, train_out)
    predicted = model.predict(test_inp)
    return model, predicted


def log_reg(train_inp, train_out, test_inp):
    model = LogisticRegression().fit(train_inp, train_out)
    predicted = model.predict(test_inp)
    return model, predicted


def dec_tree(train_inp, train_out, test_inp):
    model = DecisionTreeClassifier().fit(train_inp, train_out)
    predicted = model.predict(test_inp)
    return model, predicted

def split(inp, out, n_folds):
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=True)
    inp, out = np.array(inp), np.array(out)
    for train_index, test_index in skf.split(inp, out):
        yield inp[train_index], out[train_index], inp[test_index], out[test_index]


def measures(actual, predicted, labels):
    return precision_recall_fscore_support(actual, predicted, labels=labels)


def make_report(p, r, f1, s, labels, digits=2):
    labels = np.asarray(labels)
    width = max([len(l) for l in labels])
    last_line_heading = 'avg / total'
    width = max(len(last_line_heading), digits, width)
    headers = ["precision", "recall", "f1-score", "support"]
    fmt = '%% %ds' % width  # first column: class name
    fmt += '  '
    fmt += ' '.join(['% 9s' for _ in headers])
    fmt += '\n'
    headers = [""] + headers
    report = fmt % tuple(headers)
    report += '\n'
    for i, label in enumerate(labels):
        values = [label]
        for v in (p[i], r[i], f1[i]):
            values += ["{0:0.{1}f}".format(v, digits)]
        values += ["{0}".format(s[i])]
        report += fmt % tuple(values)
    report += '\n'
    # compute averages
    values = [last_line_heading]
    for v in (np.average(p, weights=s),
              np.average(r, weights=s),
              np.average(f1, weights=s)):
        values += ["{0:0.{1}f}".format(v, digits)]
    values += ['{0}'.format(np.sum(s))]
    report += fmt % tuple(values)
    return report

def run(corpus,label, classifier):
    print("***** %s *****" % classifier.__name__)
    labels=np.unique(label)
    precision, recall, f_score, support = np.array([0.0]*len(labels)), np.array([0.0]*len(labels)), \
        np.array([0.0]*len(labels)), np.array([0.0]*len(labels))

    i = 1
    splits = 2
    for train_inp, train_out, test_inp, test_out in split(corpus, label, splits):
        print("Split %d of %d" % (i, splits))
        model, predicted = classifier(train_inp, train_out, test_inp)
        p, r, f, s = measures(test_out, predicted, labels)
        precision += p
        recall += r
        f_score += f
        support += s
        i += 1
    precision /= len(labels)
    recall /= len(labels)
    f_score /= len(labels)
    support /= len(labels)
    return make_report(precision, recall, f_score, support, labels)

'''def __main__():
    nb_clf = Pipeline([('vect', CountVectorizer()),
                       # ('tfidf', TfidfTransformer()),
                       ('clf', MultinomialNB())])
    data = load_pkl('data/train_mini.pkl')
    nb_clf.fit(data['text'], data['label'])
    predicted = nb_clf.predict(data['text'])
    print(np.mean(predicted == data['label']))'''

def _test(res=""):
    data = load_pkl('/share/aagrawa8/Data/train.pkl')
    label = data['label']
    corpus=make_feature(data['text'], sel=res, norm="l2row", n_features=2000)

    classifiers=[lin_svm,log_reg,naive_bayes,dec_tree,rbf_svm]
    temp={}
    result={}
    for i in classifiers:
        temp[i.__name__]=run(corpus,label, i)
    result[res]=temp
    print(temp)
    with open('dump/'+res+'.pickle', 'wb') as handle:
        pickle.dump(result, handle)

if __name__ == '__main__':
    eval(cmd())
