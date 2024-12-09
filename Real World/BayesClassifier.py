import numpy as np #type:ignore

"""x1 is the vector for multiple features"""

def covar_matrix(df):
    mean = np.mean(df, axis=0)  
    centered = df - mean  
    cov_matrix = np.cov(centered, rowvar=False)  
    return cov_matrix

def mahalanobis_dist(x, mean, cov_matrix):
    diff = x - mean
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    return np.sqrt(np.dot(np.dot(diff, inv_cov_matrix), diff.T))

class bc:
    def __init__(self, df, class_column, features):
        self.df = df
        self.features = features
        self.class_column = class_column

    def pdf(self, x, class_data):
        class_features = class_data[self.features].values
        mean_vector = np.mean(class_features, axis=0)
        cov_matrix = covar_matrix(class_features)
        
        md = mahalanobis_dist(x, mean_vector, cov_matrix)
        exponential = (-0.5) * (md ** 2)
        denominator = ((2 * np.pi) ** (len(self.features) / 2)) * (np.linalg.det(cov_matrix) ** 0.5)
        return np.exp(exponential) / denominator

    def prior(self):
        class_counts = self.df[self.class_column].value_counts()
        total = len(self.df)
        return class_counts / total

    def post(self, x):
        priors = self.prior()
        posterior = {}
        
        for cls in priors.index:
            class_data = self.df[self.df[self.class_column] == cls]
            likelihood = self.pdf(x, class_data)
            posterior[cls] = likelihood * priors[cls]
        
        return posterior

    def predict(self, X):
        predictions = []
        
        for _, row in X.iterrows():
            x = row[self.features].values
            posterior = self.post(x)
            predicted_class = max(posterior, key=posterior.get)
            predictions.append(predicted_class)
        
        return predictions