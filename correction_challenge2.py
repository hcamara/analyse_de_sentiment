import warnings
warnings.simplefilter('ignore')
import os
import pandas, unidecode, json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
# Initialize Cassandra
from cassandra.cluster import Cluster
from xgboost import XGBClassifier
from flask import Flask, request


app = Flask(__name__)

class Main:

   def __init__(self):

        # Connect to the cluster
        cluster = Cluster(['13.38.152.128'])
        self.session = cluster.connect()

        # load Keyspace == databse
        self.session.set_keyspace('sentiment_analysis')

        self.vectorizer = None
        self.score = None
        self.model = None
        self.train()

   def dataset(self, champ):
      result = []
      rows = self.session.execute("""
      SELECT * FROM dataset
      """)
      for row in rows:
        if champ == 'avis':
           result.append(row.avis)

        if champ == 'note':
           result.append(row.note)

      return result
 
   def train(self):

    sentences = self.dataset('avis')
    y = self.dataset('note')

    # Split datasets // Overfitting
    sentences_train, sentences_test, y_train, y_test =   train_test_split(sentences, y, test_size=0.25, random_state=1000)

    # Verctorization of training and testing data
    self.vectorizer = CountVectorizer()
    self.vectorizer.fit(sentences_train)
    X_train = self.vectorizer.transform(sentences_train)
    X_test  = self.vectorizer.transform(sentences_test)

    # Init model and fit it
    self.model = XGBClassifier(max_depth=2, n_estimators=30)
    self.model.fit(X_train, y_train)

   def predict(self, json_text):
    # predictions
    result = self.vectorizer.transform([unidecode.unidecode(json_text)])
    result = self.model.predict(result)

    if str(result[0]) == "0":
        sentiment = "NEGATIVE"

    elif str(result[0]) == "1":
        sentiment = "POSITIVE"

    return sentiment
main = Main()

@app.route("/")
def index():
   return "Sentiment Analysis API"


@app.route("/predict", methods=["GET"])
def predict():
   text = request.args.get("query")
   result = main.predict(text)
   return result


if __name__ == "__main__":
   app.run("0.0.0.0", port=8080, debug=True)