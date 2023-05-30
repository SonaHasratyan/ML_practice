"""
    Try to beat your result on survival dataset from week 7 with a neural network.
    You can use only the layers, activations and losses that you are familiar from
    our course (you will pass some during this week, and you can use them afterward).
    Send the tf model creation, training and testing codes. Do not use test data for
    model selection, do it only once on the best network you choose and compare with
    the score you have achieved during week 7 team practice.
"""


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.metrics import AUC
from keras import regularizers

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from matplotlib import pyplot
from sklearn.utils import shuffle

tf.random.set_seed(78)
np.random.seed(78)


def threshold_selection(model, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        random_state=78,
        stratify=y_train,
    )

    model.fit(X_train, y_train)
    predict_probas = model.predict(X_val)

    auc = roc_auc_score(y_val, predict_probas)
    print("ROC AUC=%.3f" % auc)

    fpr, tpr, thresholds = roc_curve(y_val, predict_probas)

    J = tpr - fpr
    ix = np.argmax(J)
    print("Best Threshold=%f" % thresholds[ix])
    threshold = thresholds[ix]
    pyplot.plot(fpr, tpr, marker=".")
    pyplot.xlabel("False Positive Rate")
    pyplot.ylabel("True Positive Rate")

    return threshold


class Preprocessor:
    """
    All the preprocessing stages are done here - filling nans, scaling, feature extraction etc.
    """

    NANS_THRESHOLD = 60

    def __init__(self, random_state=78, do_validation=False):
        self.do_validation = do_validation
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.selected_features = None
        self.scaler = None

        self.random_state = random_state
        self.imputer = KNNImputer(n_neighbors=2)

    def fit(self, X_train, y_train):

        # Just in case checking whether there are any data points with nan labels, if so, remove them
        if y_train.isna().sum() != 0:
            y_train.drop(y_train.isna(), inplace=True)
            X_train.drop(y_train.isna(), inplace=True)

        if self.do_validation:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=0.15,
                random_state=self.random_state,
                stratify=y_train,
            )

        self.X_train, self.y_train = X_train, y_train

        self.__anomaly_detection()

        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_train)
        self.X_train[self.X_train.columns] = self.scaler.transform(
            self.X_train[self.X_train.columns]
        )

        self.__select_features()

    def transform(self, X_test):
        self.X_test = X_test

        self.X_test[self.X_test.columns] = self.scaler.transform(
            self.X_test[self.X_test.columns]
        )
        self.X_test = self.__regularize_data(self.X_test)

        # self.X_test = self.X_test.values

        return self.X_test

    def __regularize_data(self, X):

        print(f"Number of columns BEFORE dropping: {len(X.columns)}")
        X = X[self.selected_features].copy()

        print(f"Number of columns AFTER dropping: {len(X.columns)}")

        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

        print(
            f"Are there left any columns with nan values? - {any((X.isna().sum() * 100) / len(X) > 0)}"
        )

        return X

    def __select_features(self):
        """
        removing those columns which include over NANS_THRESHOLD % nans
        filling nans
        setting self.select_features
        """

        nans_percentage = (
            (self.X_train.isna().sum() * 100) / len(self.X_train)
        ).sort_values(ascending=False)
        cols_with_nans_stats = pd.DataFrame(
            {"columns": self.X_train.columns, "nans_percentage": nans_percentage}
        )
        cols_with_nans_stats.sort_values("nans_percentage", inplace=True)

        print(
            f"Number of columns including nans: "
            f'{len(cols_with_nans_stats[cols_with_nans_stats["nans_percentage"] > 0])}'
        )

        features_to_drop = cols_with_nans_stats[
            cols_with_nans_stats["nans_percentage"] > self.NANS_THRESHOLD
        ]["columns"].to_numpy()

        print(
            f"Number of columns with > {self.NANS_THRESHOLD}% nan values: {len(features_to_drop)}"
        )

        X_tmp = self.X_train[self.X_train.columns.difference(features_to_drop)].copy()

        X_tmp = pd.DataFrame(self.imputer.fit_transform(X_tmp), columns=X_tmp.columns)

        alpha = self.__get_lasso_alpha(X_tmp)

        feature_sel_model = SelectFromModel(
            Lasso(alpha=alpha, random_state=self.random_state)
        )

        feature_sel_model.fit(X_tmp, self.y_train)

        # list of the selected features
        self.selected_features = X_tmp.columns[(feature_sel_model.get_support())]

        # let's print some stats
        print(f"#Total features: {self.X_train.shape[1]}")
        print(f"#Selected features: {len(self.selected_features)}")

    def __anomaly_detection(self):
        # TODO: validate quantiles
        # Identify potential outliers for each column
        outliers = {}
        for col in self.X_train.columns:
            # print(self.X_train[col].describe())
            q1 = self.X_train[col].quantile(0.25)
            q3 = self.X_train[col].quantile(0.75)
            iqr = q3 - q1
            upper_lim = q3 + 1.5 * iqr
            lower_lim = q1 - 1.5 * iqr
            outliers[col] = self.X_train.loc[
                (self.X_train[col] < lower_lim) | (self.X_train[col] > upper_lim), col
            ]
            self.X_train.loc[
                (self.X_train[col] < lower_lim) | (self.X_train[col] > upper_lim), col
            ] = np.nan
            # print(self.X_train[col].describe())
        # print(outliers)

    def __get_lasso_alpha(self, X_tmp, grid_search=False):
        if grid_search:
            from sklearn.model_selection import GridSearchCV

            lasso = Lasso(random_state=self.random_state)
            params = {
                "alpha": [
                    1e-5,
                    1e-4,
                    1e-3,
                    1e-2,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    1,
                    2,
                    3,
                    4,
                    5,
                    10,
                    20,
                    30,
                    40,
                    50,
                    100,
                    200,
                    300,
                    400,
                    500,
                ]
            }
            Regressor = GridSearchCV(
                lasso, params, scoring="neg_mean_squared_error", cv=10
            )
            Regressor.fit(X_tmp, self.y_train)
            print("best parameter: ", Regressor.best_params_)
            print("best score: ", -Regressor.best_score_)
            return Regressor.best_params_["alpha"]
        else:
            return 0.001


df = pd.read_csv("hospital_deaths_train.csv")
y = df["In-hospital_death"]
X = df.drop("In-hospital_death", axis=1)
X, y = shuffle(X, y, random_state=78)

# y.to_numpy()
# X.to_numpy()

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=78,
    stratify=y,
)

preprocessor = Preprocessor()
preprocessor.fit(X_train, y_train)
X_train = preprocessor.transform(X_train)
X_val = preprocessor.transform(X_val)


model = Sequential(
    [
        Dense(5, activation="relu", name="layer1"),
        # Dense(3, activation="relu", name="layer2"),
        Dense(1, activation="sigmoid", name="layer3", use_bias=False),
    ]
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC()])


model.fit(X_train, y_train, epochs=1000, batch_size=64)

threshold = threshold_selection(model, X_train, y_train)
y_val_pred = model.predict(X_val)
predictions = (y_val_pred > threshold).astype(int)


print("roc_auc_score", roc_auc_score(y_val, predictions))
print("confusion_matrix", confusion_matrix(y_val, predictions))
