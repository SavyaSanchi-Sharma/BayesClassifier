import numpy as np
import BayesClassifier as bc
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

class decisionBoundry(bc.bc):
    def _init_(self,df, class_column, features):
        super()._init_(df, class_column, features)

    def decisionFunc(self,class_data):
        x=[]
        for _, row in class_data.iterrows():
            x = row[self.features].values
        class_features = class_data[self.features].values
        mean_vector = np.mean(class_features, axis=0)
        cov_matrix = bc.covar_matrix(class_features)
        md = bc.mahalanobis_dist(x, mean_vector, cov_matrix)
        func=(-0.5)*md-np.log(2*np.pi)-(0.5)*np.log(np.linalg.det(cov_matrix))-self.pdf(x,class_data)
        return func 
    

    
    def plot_decision(self,X):
        x1_min, x1_max = X['x1'].min() - 1, X['x1'].max() + 1
        x2_min, x2_max = X['x2'].min() - 1, X['x2'].max() + 1
        X1,X2=np.meshgrid(
            np.arange(x1_min,x1_max),
            np.arange(x2_min,x2_max)
        )
        grid_points=np.c_[X1.ravel(),X2.ravel()]
        predPoints=[]
        for point in grid_points:
            posterior=self.post(point)
            predPoints.append(max(posterior,key=posterior.get))
        Z=np.array(predPoints).reshape(X1.shape)
        plt.figure(figsize=(10, 8))
        cmap = ListedColormap(['red', 'blue', 'green']) 
        contourf=plt.contourf(X1, X2, Z, alpha=0.5, cmap=cmap)
        plt.contour(X1,X2, Z, colors='black', linewidths=2, levels=np.unique(Z))
        #scatter_colors = ['darkred', 'darkblue', 'darkgreen']
        #for idx, class_label in enumerate(np.unique(y)):
        #    plt.scatter(X[y == class_label, 0], X[y == class_label, 1],
        #        c=scatter_colors[idx], label=f"Class {class_label}", edgecolor="k")
        
        plt.colorbar(contourf, label='Class Regions') 
        plt.title("Decision Boundary for Three Classes")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()