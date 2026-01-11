#Program 11
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn import datasets 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.utils import Bunch
from typing import cast
iris_bunch = cast(Bunch, datasets.load_iris()) 
X = iris_bunch.data # Features 
y = iris_bunch.target # Target classes 
print(y) 
lda = LDA(n_components=2) 
X_lda = lda.fit_transform(X, y) 
lda_df = pd.DataFrame(data=X_lda, columns=['LD1', 'LD2']) 
lda_df['target'] = y 
lda_df['target'] = lda_df['target'].map({0: 'Setosa', 1: 'Versicolor', 
2: 'Virginica'}) 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(data=lda_df, x='LD1', y='LD2', hue='target', 
palette='viridis', s=100) 
plt.title('LDA of Iris Dataset') 
plt.xlabel('Linear Discriminant 1') 
plt.ylabel('Linear Discriminant 2') 
plt.legend(title='Species') 
plt.grid() 
plt.savefig("outputs/11_lda.png", dpi=150, bbox_inches="tight")
plt.close()