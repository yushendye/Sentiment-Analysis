from tkinter import ALL
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
import random
import pickle


# Get models

from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

ALL_WORDS = []
DOCUMENTS = []

def get_review_data(text_file):
    return open(text_file, encoding="unicode_escape").read()



def get_docs_and_words(pos_data_file, neg_data_file):
    PART_OF_SPEACH_OF_INTEREST = ["J", "R"]


    positive_review = get_review_data(pos_data_file)
    negative_review = get_review_data(neg_data_file)


    for review_line in positive_review.split("\n"):
        DOCUMENTS.append((review_line, "pos"))
        words_in_line = word_tokenize(review_line)
        part_of_speach = nltk.pos_tag(words_in_line)

        for part in part_of_speach:
            if part[1][0] in PART_OF_SPEACH_OF_INTEREST:
                ALL_WORDS.append(part[0].lower())


    for review_line in negative_review.split("\n"):
        DOCUMENTS.append((review_line, "neg"))
        words_in_line = word_tokenize(review_line)

        part_of_speach = nltk.pos_tag(words_in_line)
        for part in part_of_speach:
            if part[1][0] in PART_OF_SPEACH_OF_INTEREST:
                ALL_WORDS.append(part[0])



def pickle_docs(save_path):
    save_as = open(save_path, "wb")
    pickle.dump(DOCUMENTS, save_as)
    save_as.close()
    print('Data has been saved!')


def find_features(document, word_features):
    words = word_tokenize(document)
    features = {}

    for word in word_features:
        features[word] = (word in words)

    return features
()

def main():
    POS_DATA_FILE = 'positive.txt'
    NEG_DATA_FILE = 'negative.txt'

    get_docs_and_words(POS_DATA_FILE, NEG_DATA_FILE)

    ALL_WORDS_FREQ_DIST = nltk.FreqDist(ALL_WORDS)
    feature_set = list(ALL_WORDS_FREQ_DIST.keys())[:5000]

    feature_set_file = open("feature_set.pickle", 'wb')
    pickle.dump(feature_set, feature_set_file)
    feature_set_file.close()
    print("Feature Set Saved!!")    


    features = [(find_features(review, feature_set), classification) for (review, classification) in DOCUMENTS]
    random.shuffle(features)

    train_data = features[:10000]
    valid_data = features[10000:]

    pickle_docs('data.pickle')
    
    #training with NaiveBayes classifier
    naive_bayes_classifier = nltk.NaiveBayesClassifier.train(train_data)    
    print('Naive Bayes Classifier Accuracy : ', nltk.classify.accuracy(naive_bayes_classifier, valid_data) * 100)
    naive_bayes_pickle = open('naive_bayes.pickle', 'wb')
    pickle.dump(naive_bayes_classifier, naive_bayes_pickle)
    naive_bayes_pickle.close()


    #training with LogisticRegression
    logistic_reg_classifier = SklearnClassifier(LogisticRegression())
    logistic_reg_classifier.train(train_data)
    print('Logistic Regression Classifier Accuracy : ', nltk.classify.accuracy(logistic_reg_classifier, valid_data) * 100)
    logistic_pickle = open('logistic_reg.pickle', 'wb')
    pickle.dump(logistic_reg_classifier, logistic_pickle)
    logistic_pickle.close()


    #training with SGD
    sgd_classifier = SklearnClassifier(SGDClassifier())
    sgd_classifier.train(train_data)
    print('SGD Classifier Accuracy : ', nltk.classify.accuracy(sgd_classifier, valid_data) * 100)
    sgd_pickle = open('sgd.pickle', 'wb')
    pickle.dump(sgd_classifier, sgd_pickle)
    sgd_pickle.close()

    #training with SVC
    svc_classifier = SklearnClassifier(SVC())
    svc_classifier.train(train_data)
    print('SVC Accuracy : ', nltk.classify.accuracy(svc_classifier, valid_data) * 100)
    svc_pickle = open('svc.pickle', 'wb')
    pickle.dump(svc_classifier, svc_pickle)
    svc_pickle.close()

    #training with NuSVC
    nusvc_classifier = SklearnClassifier(NuSVC())
    nusvc_classifier.train(train_data)
    print('NuSVC Accuracy : ', nltk.classify.accuracy(nusvc_classifier, valid_data) * 100)
    nusvc_pickle = open('nusvc.pickle', 'wb')
    pickle.dump(nusvc_classifier, nusvc_pickle)
    nusvc_pickle.close()

    #training with Linear SVC
    linear_svc_classifier = SklearnClassifier(LinearSVC())
    linear_svc_classifier.train(train_data)
    print('Linear SVC Accuracy : ', nltk.classify.accuracy(linear_svc_classifier, valid_data) * 100)
    linear_svc_pickle = open('linear_svc.pickle', 'wb')
    pickle.dump(linear_svc_classifier, linear_svc_pickle)
    linear_svc_pickle.close()


    #training with Bernoulli Classifier
    bernoulli_classifier = SklearnClassifier(BernoulliNB())
    bernoulli_classifier.train(train_data)
    print('Bernoulli Accuracy : ', nltk.classify.accuracy(bernoulli_classifier, valid_data) * 100)
    bernoulli_pickle = open('bernoulli.pickle', 'wb')
    pickle.dump(bernoulli_classifier, bernoulli_pickle)
    bernoulli_pickle.close()

    #training with Multinomial Classifier
    multinomial_classifier = SklearnClassifier(MultinomialNB())
    multinomial_classifier.train(train_data)
    print('Multinomial Accuracy : ', nltk.classify.accuracy(bernoulli_classifier, valid_data) * 100)
    multinomial_pickle = open('multinomial.pickle', 'wb')
    pickle.dump(multinomial_classifier, multinomial_pickle)
    multinomial_pickle.close()


if __name__ == '__main__':
    main()