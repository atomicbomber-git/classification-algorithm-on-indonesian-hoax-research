from collections import OrderedDict

import numpy as np
import pandas as pandas
from nltk.corpus import stopwords
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler


true_data = np.array([1, 1, 1, 1, 0])
pred_data = np.array([0, 0, 0, 0, 0])

precision, recall, fscore, support = precision_recall_fscore_support(true_data, pred_data, average="binary")
accuracy = accuracy_score(true_data, pred_data)

print("PRECISION: {}".format(precision))
print("RECALL: {}".format(recall))
print("FSCORE: {}".format(fscore))
print("ACCURACY: {}".format(accuracy))

exit(0)

N_ESTIMATORS = 10
TEST_SIZE = 0.2
N_ITERATIONS = 1000
LEARNING_RATE = 0.1

INPUT_FILE_NAME = "cleaned_tweets.csv"
ROW_CLEANED_TWEET_TEXT = 3
ROW_LABEL = 2

cleaned_tweets = pandas.read_csv(INPUT_FILE_NAME, header=None, skiprows=[0])[
    [ROW_CLEANED_TWEET_TEXT]]

labels = pandas.read_csv(INPUT_FILE_NAME, header=None, skiprows=[0])[
    [ROW_LABEL]]


def process_target_text(target_text: str) -> bool:
    return str(target_text).strip().lower()[0] == 'h'


# Process data
tf_idf_converter = TfidfVectorizer(
    max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('indonesian'))
data = np.array([row[0] for row in cleaned_tweets.values.tolist()])[0:-1]
data = tf_idf_converter.fit_transform(data).toarray()

# Process target
target = np.array([process_target_text(row[0]) for row in labels.values.tolist()])[0:-1]


def perceptron(data_train, data_test, target_train, target_test):
    scaler = StandardScaler()
    scaler.fit(data_train)

    data_train_standard = scaler.transform(data_train)
    data_test_standard = scaler.transform(data_test)

    perceptron = Perceptron(max_iter=N_ITERATIONS, eta0=LEARNING_RATE, random_state=0)
    perceptron.fit(data_train_standard, target_train)
    predict = perceptron.predict(data_test_standard)

    return (*precision_recall_fscore_support(target_test, predict, average='binary'),
            accuracy_score(target_test, predict))


def naive_bayes(data_train, data_test, target_train, target_test):
    multinomialNB = MultinomialNB()
    multinomialNB.fit(data_train, target_train)
    predict = multinomialNB.predict(data_test)
    return (*precision_recall_fscore_support(target_test, predict, average='binary'), accuracy_score(target_test, predict))


def support_vector_machine(data_train, data_test, target_train, target_test):
    support_vector_machine = svm.SVC()
    support_vector_machine.fit(data_train, target_train)
    predict = support_vector_machine.predict(data_test)

    print(precision_recall_fscore_support(target_test, predict, average='binary'))


    return (
        *precision_recall_fscore_support(target_test, predict, average='binary'),
        accuracy_score(target_test, predict))


def decision_tree(data_train, data_test, target_train, target_test):
    decision_tree_classifier = tree.DecisionTreeClassifier()
    decision_tree_classifier.fit(data_train, target_train)
    predict = decision_tree_classifier.predict(data_test)
    return (
        *precision_recall_fscore_support(target_test, predict, average='binary'),
        accuracy_score(target_test, predict))


def random_forest(data_train, data_test, target_train, target_test):
    classifier = RandomForestClassifier(n_estimators=N_ESTIMATORS)
    classifier.fit(data_train, target_train)
    predict = classifier.predict(data_test)
    return (*precision_recall_fscore_support(target_test, predict, average='binary'),
            accuracy_score(target_test, predict))


algorithms = OrderedDict()
algorithms['perceptron'] = {'method': perceptron, 'precision': [], 'recall': [], 'fscore': [], 'accuracy': [], 'support': []}
algorithms['naive_bayes'] = {'method': naive_bayes, 'precision': [], 'recall': [], 'fscore': [], 'accuracy': [], 'support': []}
algorithms['support_vector_machine'] = {'method': support_vector_machine, 'precision': [], 'recall': [], 'fscore': [], 'accuracy': [], 'support': []}
algorithms['decision_tree'] = {'method': decision_tree, 'precision': [], 'recall': [], 'fscore': [], 'accuracy': [], 'support': []}
algorithms['random_forest'] = {'method': random_forest, 'precision': [], 'recall': [], 'fscore': [], 'accuracy': [], 'support': []}

kFolder = KFold(n_splits=5)

for train_index, test_index in kFolder.split(data):
    print("Tweet ke {}-{}".format(test_index[0] + 1, test_index[-1] + 1))

    data_train, target_train = data[train_index], target[train_index]
    data_test, target_test = data[test_index], target[test_index]

    print("Algoritma, Precision, Recall, F1-Score, Accuracy")

    for algorithm_name, content in algorithms.items():
        precision, recall, fscore, support, accuracy = content['method'](data_train, data_test, target_train, target_test)

        content['precision'].append(precision)
        content['recall'].append(recall)
        content['fscore'].append(fscore)
        content['accuracy'].append(accuracy)

        print("{}: {}".format(algorithm_name, support))


        # print("{}, {}, {}, {}, {}".format(
        #     algorithm_name.title().replace('_', ' '),
        #     precision,
        #     recall,
        #     fscore,
        #     accuracy
        # ))
    print("")

# print("Rata-Rata")
# print("Algoritma, Precision, Recall, F1-Score, Accuracy")
# for algorithm_name, content in algorithms.items():
#     print("{}, {}, {}, {}, {}".format(
#         algorithm_name.title().replace('_', ' '),
#         mean(content['precision']),
#         mean(content['recall']),
#         mean(content['fscore']),
#         mean(content['accuracy'])
#     ))
# print("")