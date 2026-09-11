import numpy as np
import matplotlib.pyplot as plt

class Vectorizer:
    """
        Transform raw data into feature vectors. Support ordinal, numerical and categorical data.
        Also implements feature normalization and scaling.

        TODO: Support numerical, ordinal, categorical, histogram features.
    """
    def __init__(self, feature_config, num_bins=5):
        self.feature_config = feature_config
        self.feature_transforms = {}
        self.is_fit = False

    def get_numerical_vectorizer(self, name, verbose=False):
        """
        :return: function to map numerical x to a zero mean, unit std dev normalized score.
        """
        def vectorizer(x):
            """
            :param x: numerical value
            Return transformed score

            Hint: this fn knows mean and std from the outer scope
            """
            try:
                return (float(x) - self.means[name]) / self.stds[name]
            except:
                return 0

        return vectorizer

    def get_histogram_vectorizer(self, values):
        plt.hist(values, bins=self.num_bins)
        plt.show()
        
    def get_categorical_vectorizer(self, name):
        """
        :return: function to map categorical x to one-hot feature vector
        """
        def vectorizer(x):
            possible_value = self.possible[name]
            ind = np.where(possible_value == x)[0]
            out = np.zeros(len(possible_value))
            out[ind] = 1
            return out

        return vectorizer

    def get_ordinal_vectorizer(self, name):

        def vectorizer(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return 0

        return vectorizer



    def is_row_valid(self, row):
        """
        A row is valid if every configured feature is present and non-empty.
        """
        for num_feat in self.feature_config["numerical"]:
            value = row.get(num_feat, "")
            if value in ("", None):
                return False
            try:
                float(value)
            except (TypeError, ValueError):
                return False

        for cat_feature in self.feature_config["categorical"]:
            value = row.get(cat_feature, "")
            if value in ("", None):
                return False

        for ord_feat in self.feature_config["ordinal"]:
            value = row.get(ord_feat, "")
            if value in ("", None):
                return False
            try:
                float(value)
            except (TypeError, ValueError):
                return False


        return True

    def fit(self, X):
        """
            Leverage X to initialize all the feature vectorizers (e.g. compute means, std, etc)
            and store them in self.

            This implementation will depend on how you design your feature config.
        """
        # Drop invalid rows in place, so train/val/test (the callers' lists) end up filtered too.
        X[:] = [row for row in X if self.is_row_valid(row)]

        self.dict_vectorizer = {}
        self.means = {}
        self.stds = {}
        for num_feat in self.feature_config["numerical"]:
            values = [float(row[num_feat]) for row in X]

            mean = float(np.mean(values))
            std = float(np.std(values))
            if std == 0.0:
                std = 1.0

            self.means[num_feat] = mean
            self.stds[num_feat] = std
            self.dict_vectorizer[num_feat] = self.get_numerical_vectorizer(num_feat)

        self.possible = {}
        for cat_feature in self.feature_config["categorical"]:
            values = [row[cat_feature] for row in X]
            self.possible[cat_feature] = np.unique(values)
            self.dict_vectorizer[cat_feature] = self.get_categorical_vectorizer(cat_feature)


        for ord_feat in self.feature_config["ordinal"]:
            self.dict_vectorizer[ord_feat] = self.get_ordinal_vectorizer(ord_feat)



        self.is_fit = True


    def transform(self, X):
        """
        For each data point, apply the feature transforms and concatenate the results into a single feature vector.

        :param X: list of dicts, each dict is a datapoint
        """

        if not self.is_fit:
            self.fit(X)
        else:
            # val/test never go through fit(), so apply the same row filtering here.
            X[:] = [row for row in X if self.is_row_valid(row)]

        transform = []
        for i in range(len(X)):
            ind = 0
            L=[]

            for k in range(len(self.feature_config["numerical"])):
                name = self.feature_config["numerical"][k]
                x = X[i][name]
                funct = self.dict_vectorizer[name]
                L.append(funct(x))

            for k in range(len(self.feature_config["categorical"])):
                name = self.feature_config["categorical"][k]
                x = X[i][name]
                funct = self.dict_vectorizer[name]
                list = funct(x)
                for m in range(len(list)):
                    L.append(list[m])

            for k in range(len(self.feature_config["ordinal"])):
                name = self.feature_config["ordinal"][k]
                x = X[i][name]
                funct = self.dict_vectorizer[name]
                L.append(funct(x))

            
            transform.append(L)

        return np.asarray(transform, dtype=float)