from gcforest.gcforest import GCForest
import keras

train_data, test_data = keras.datasets.mnist.load_data()
# train_data, test_data = keras.datasets.imdb.load_data()

train_X, train_Y = train_data
# train_X = np.array([X[:6] for X in train_X])
# train_X = np.reshape(train_X, newshape=(-1, 1, 6, 1))

test_X, test_Y = test_data
# test_X =np.array([X[:6] for X in test_X])
# test_X = np.reshape(test_X, newshape=(-1, 1, 6, 1))

config = {
    "cascade": {
        "random_state": 0,
        "max_layers": 100,
        "early_stopping_rounds": 3,
        "n_classes": 10,
        "estimators": [
            # {"n_folds":5,"type":"XGBClassifier","n_estimators":10,"max_depth":5,"objective":"multi:softprob", "silent":True, "nthread":-1, "learning_rate":0.1},
            {"n_folds":5,"type":"RandomForestClassifier","n_estimators":10,"max_depth":None,"n_jobs":-1},
            {"n_folds":5,"type":"ExtraTreesClassifier","n_estimators":10,"max_depth":None,"n_jobs":-1},
            {"n_folds":5,"type":"LogisticRegression"}
        ]
    }
}

gc = GCForest(config=config)
X_train_enc = gc.fit_transform(train_X, train_Y, X_test=test_X, y_test=test_Y)
# y_pred = gc.predict(test_X)

# print(keras.metrics.accuracy(y_true=test_Y, y_pred=y_pred))