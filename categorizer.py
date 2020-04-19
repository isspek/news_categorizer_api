import os
import pandas as pd
from ktrain import load_predictor, text, get_learner, get_predictor
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.backend import set_session
from loguru import logger

categorizer_model = './data/model/model'
data_path = './data/BBC News Train.csv'
random_state = 42
num_batch = 5
lr = 2e-5

GRAPH = None
SESS = None
PREDICTOR = None

expected_domains = {
    'politics': ['politics'],
    'sports': ['sports', 'sport', 'basketball', 'tennis', 'football'],
    'crime': ['crime', 'law'],
    'economy': ['economy', 'finance', 'business', 'markets', 'market', 'forbes'],
    'entertainment': ['entertainment', 'music', 'film', 'movie', 'hollywood', 'celebrity', 'celebrity-news'],
    'life': ['lifestyle', 'life-style', 'life', 'health'],
    'science and technology': ['technology', 'product', 'science', 'tech'],
    'uncategorized': ['uncategorized'],
}


def train():
    data = pd.read_csv(data_path)
    X = data['Text'].values.tolist()
    y = data['Category'].values.tolist()
    X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=random_state, shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=random_state,
                                                        shuffle=True)
    (X_train, y_train), (X_dev, y_dev), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                        x_test=X_dev, y_test=y_dev,
                                                                        class_names=set(y_train),
                                                                        preprocess_mode='bert',
                                                                        ngram_range=3,
                                                                        maxlen=512,
                                                                        max_features=35000)

    model = text.text_classifier('bert', train_data=(X_train, y_train), preproc=preproc)
    learner = get_learner(model, train_data=(X_train, y_train), val_data=(X_dev, y_dev), batch_size=2)
    learner.view_top_losses(n=num_batch, preproc=preproc)
    learner.autofit(lr, num_batch)
    print(learner.validate(val_data=(X_dev, y_dev)))
    predictor = get_predictor(learner.model, preproc)
    y_pred = predictor.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    predictor.save(categorizer_model)
    logger.info("Model has been saved")

    # Saving Model
    model_json = learner.model.to_json()
    with open(os.path.join(categorizer_model,"model.json"), "w") as json_file:
        json_file.write(model_json)

    learner.model.save_weights(os.path.join(categorizer_model,'model.h5'))
    logger.info("Model weights have been saved")
    return predictor

def load_predictor():
    global SESS
    SESS = tf.Session()
    set_session(SESS)

    # load the model, and pass in the custom metric function
    global GRAPH
    GRAPH = tf.get_default_graph()

    global PREDICTOR
    PREDICTOR = load_predictor(categorizer_model)
    if hasattr(PREDICTOR.model, '_make_predict_function'):
        PREDICTOR.model._make_predict_function()

if os.path.exists(categorizer_model):
    logger.debug('{} exists. Loading...'.format(categorizer_model))
    model = load_predictor()
else:
    logger.debug('{} does not exist. Training...'.format(categorizer_model))
    model = train()


def get_category_from_url(url):
    splitted_url = url.split('/')
    categories = splitted_url[3:-1]

    for domain, subdomains in expected_domains.items():
        if not categories:
            for category in categories:
                if any(category == subdomain for subdomain in subdomains):
                    return domain
        else:
            if any(subdomain in url for subdomain in subdomains):
                return domain

    return None


def get_category_from_content(content):
    '''
    Only supports ['business', 'entertainment', 'politics', 'sport', 'tech']
    :param content:
    :type content:
    :return:
    :rtype:
    '''
    return {'content': content, 'category': model.predict(content)}
