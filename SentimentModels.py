import nltk
import pickle

from nltk.classify import ClassifierI
from statistics import mode
from SentimentModelTrainer import find_features


DOCUMENTS = []
FEATURES = []

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers        

    def classify(self, featureset):
        votes = []

        for c in self._classifiers:
            #print(type(c[0]))
            #print(type(c[1]))
            vote = c[0].classify(featureset)
            votes.append(vote)

        return mode(votes)


    def confidence(self, features):
        votes = []
        for classifier in self._classifiers:
            vote = classifier[0].classify(features)
            votes.append(vote)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)

        return conf



def get_models():
    naive_bayes_pickle = open('pickles/logistic_reg.pickle', 'rb')
    naive_bayes_model = pickle.load(naive_bayes_pickle)
    naive_bayes_pickle.close()

    
    logistic_pickle = open('pickles/logistic_reg.pickle', 'rb')
    logistic_reg_model = pickle.load(logistic_pickle)
    logistic_pickle.close()



    sgd_pickle = open('pickles/sgd.pickle', 'rb')
    sgd_model = pickle.load(sgd_pickle)
    sgd_pickle.close()


    svc_pickle = open('pickles/svc.pickle', 'rb')
    svc_model = pickle.load(svc_pickle)
    svc_pickle.close()


    nusvc_pickle = open('pickles/nusvc.pickle', 'rb')
    nusvc_model = pickle.load(nusvc_pickle)
    nusvc_pickle.close()

    linear_svc_pickle = open('pickles/linear_svc.pickle', 'rb')
    linear_svc_model = pickle.load(linear_svc_pickle)
    linear_svc_pickle.close()


    bernoulli_pickle = open('pickles/bernoulli.pickle', 'rb')
    bermoulli_model = pickle.load(bernoulli_pickle)
    bernoulli_pickle.close()


    multinomial_pickle = open('pickles/multinomial.pickle', 'rb')
    multinomial_model = pickle.load(multinomial_pickle)
    multinomial_pickle.close()

    return [naive_bayes_model, logistic_reg_model, sgd_model, svc_model, nusvc_model, linear_svc_model, bermoulli_model, multinomial_model]


def load_data():
    doc_f = open('pickles/data.pickle', 'rb')
    DOCUMENTS = pickle.load(doc_f)
    doc_f.close()


    features_f = open('pickles/feature_set.pickle', 'rb')
    FEATURES = pickle.load(features_f)
    features_f.close()

    return DOCUMENTS, FEATURES


def predict_sentiment(text):
    DOCUMENTS, FEATURES =  load_data()
    features = find_features(text, FEATURES)
    vote_classifier = VoteClassifier(get_models())
    return vote_classifier.classify(features), vote_classifier.confidence(features)