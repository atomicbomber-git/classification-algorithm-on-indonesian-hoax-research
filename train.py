import pandas as pandas
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

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
data = [row[0] for row in cleaned_tweets.values.tolist()]
data = tf_idf_converter.fit_transform(data).toarray()

# Process target
target = [process_target_text(row[0]) for row in labels.values.tolist()]

# Split
data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=TEST_SIZE)


def perceptron(data_train, data_test, target_train, target_test):
    scaler = StandardScaler()
    scaler.fit(data_train)

    data_train_standard = scaler.transform(data_train)
    data_test_standard = scaler.transform(data_test)

    perceptron = Perceptron(max_iter=N_ITERATIONS, eta0=LEARNING_RATE, random_state=0)
    perceptron.fit(data_train_standard, target_train)

    prediction = perceptron.predict(data_test_standard)

    return accuracy_score(target_test, prediction)


def naive_bayes(data_train, data_test, target_train, target_test):
    multinomialNB = MultinomialNB()
    multinomialNB.fit(data_train, target_train)
    return accuracy_score(target_test, multinomialNB.predict(data_test))


def support_vector_machine(data_train, data_test, target_train, target_test):
    support_vector_machine = svm.SVC()
    support_vector_machine.fit(data_train, target_train)
    return accuracy_score(target_test, support_vector_machine.predict(data_test))

def decision_tree(data_train, data_test, target_train, target_test):
    decision_tree_classifier = tree.DecisionTreeClassifier()
    decision_tree_classifier.fit(data_train, target_train)
    return accuracy_score(target_test, decision_tree_classifier.predict(data_test))

def random_forest(data_train, data_test, target_train, target_test):
    classifier = RandomForestClassifier(n_estimators=N_ESTIMATORS)
    classifier.fit(data_train, target_train)
    return accuracy_score(target_test, classifier.predict(data_test))


print("Perceptron Accuracy: {}".format(perceptron(data_train, data_test, target_train, target_test)))
print("Naive Bayes Accuracy: {}".format(naive_bayes(data_train, data_test, target_train, target_test)))
print("Support Vector Machine Accuracy: {}".format(support_vector_machine(data_train, data_test, target_train, target_test)))
print("Decision Tree Classifier: {}".format(decision_tree(data_train, data_test, target_train, target_test)))
print("Random Forest Classifier: {}".format(random_forest(data_train, data_test, target_train, target_test)))
