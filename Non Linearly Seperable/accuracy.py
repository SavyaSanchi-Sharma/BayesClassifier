import numpy as np
class accuracy:
    def __init__(self,y,y_hat):
        self.y=y
        self.y_hat=y_hat
    
    def confusion_matrix(self):
        cm=np.zeros((4,4),int)
        for true, pred in zip(self.y,self.y_hat):
            cm[true][pred] += 1
        return cm
    
    def accu_percent(self):
        cm = self.confusion_matrix()
        correct_predictions = np.trace(cm)  
        total_predictions = np.sum(cm)  
        return (correct_predictions / total_predictions) * 100
    def precision(self):
        n=len(np.unique(self.y))
        cm=self.confusion_matrix()
        p=[]
        for i in range(n):
            
            positive_class=np.sum(cm[:, i+1])
            print(f"Precision for class {i+1} is")
            prec=(cm[i+1][i+1]/positive_class)
            print(prec)
            p.append(prec)
        return p
    def recall(self):
        n=len(np.unique(self.y))
        cm=self.confusion_matrix() 
        r=[]
        for i in range(n):
            
            total=np.sum(cm[i+1, :])
            print(f"Recall for class{i+1} is ")
            rec=(cm[i+1][i+1]/total)
            print(rec)
            r.append(rec)
        return r
    def f_score(self):
        precisions = self.precision()
        recalls = self.recall()
        f_scores = []
        for p, r in zip(precisions, recalls):
                f_scores.append(2 * p * r / (p + r))
        for i, f in enumerate(f_scores):
            print(f"F-score for class {i+1} is {f}")
        return f_scores