import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from matplotlib.colors import ListedColormap

class decisionBoundary():
    def __init__(self,df, class_column, features):
        self.df = df
        self.class_column = class_column
        self.features = features

    def prior(self, class_data):
        return len(class_data) / len(self.df) 

    def decisionFunc(self, cov_matrix, x, class_data):
        mu_i = np.mean(class_data, axis=0).reshape(-1, 1) 
        sigma_i = cov_matrix
        diff = x - mu_i.T  
        inv = np.linalg.inv(sigma_i)  
        d1 = np.dot(inv, diff.T)
        term1 = -0.5 * np.sum(diff * d1.T, axis=1)
        term2 = -0.5 * len(mu_i) * np.log(2 * np.pi)
        term3 = -0.5 * np.log(np.linalg.det(sigma_i))
        term4 = np.log(self.prior(class_data))

        g = term1 + term2 + term3 + term4
        return g
    
    
    
    def normalData(self,X):
        scaler=MinMaxScaler()
        return scaler.fit_transform(X)
    
    
    
    def plot_decision_boundaries_with_cov(self, cov_matrices,normalise:bool):
        classes = self.df[self.class_column].unique()
        if normalise:
            self.df[self.features]=self.normalData(self.df[self.features])
        x1_min, x1_max = self.df[self.features[0]].min() - 1, self.df[self.features[0]].max() + 1
        x2_min, x2_max = self.df[self.features[1]].min() - 1, self.df[self.features[1]].max() + 1
        X1, X2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
        grid_points = np.c_[X1.ravel(), X2.ravel()]
        
        decision_values = {}
        for cls, cov_matrix in zip(classes, cov_matrices):
            class_data = self.df[self.df[self.class_column] == cls][self.features].values
            decision_values[cls] = self.decisionFunc(cov_matrix, grid_points, class_data)

        plt.figure(figsize=(10, 6))

        colors=['r','g','b']
        for i, cls1 in enumerate(classes):
            for j, cls2 in enumerate(classes):
                if i != j:
                    diff = decision_values[cls1] - decision_values[cls2]
                    plt.contour(
                        X1, X2, diff.reshape(X1.shape), levels=[1],color=[colors[i]]
                    )


        for cls, color in zip(classes, colors):
            class_data = self.df[self.df[self.class_column] == cls]
            plt.scatter(
                class_data[self.features[0]], class_data[self.features[1]], label=f"Class {cls}", color=color
            )

        plt.xlabel(self.features[0])
        plt.ylabel(self.features[1])
        plt.legend()
        plt.show()

  