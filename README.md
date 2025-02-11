## Introduction  
---------------

The **Dry Bean Dataset** from the UCI Machine Learning Repository is a well-structured dataset used for classifying different types of beans based on their morphological features. The dataset consists of **various shape-related attributes** extracted from bean images using computer vision techniques. Given these numerical attributes, the goal is to build a **classification model** that can accurately predict the bean type.  

In this **[notebook](https://github.com/capsuleismail/dry_bean_uci/blob/main/dry-bean-dataset-uci.ipynb)**, we explore and analyze the Dry Bean Dataset by answering key questions related to its structure, attributes, and classification potential. We perform **exploratory data analysis (EDA)** using **histograms, boxplots, and correlation matrices** to understand feature distributions and relationships. Additionally, we implement various **machine learning models**, compare their performance, and optimize hyperparameters using **Optuna** to enhance classification accuracy.  

Through this analysis, we aim to determine the **most effective model** for distinguishing between different bean types, leveraging advanced preprocessing techniques and **machine learning pipelines** to streamline the workflow.

These are the questions I've gone through on my notebooks:

1. **What is the Dry Bean Dataset?** <br/>
2. **How many instances (rows) and attributes (columns) are present in the dataset?** <br/>
3. **What are the different classes of beans in the dataset?** <br/>
4. **What are the main features (attributes) used to describe each bean?** <br/>
5. **Are all attributes numerical, or are there categorical attributes as well?** <br/>
6. **What type of classification problem is this dataset used for? (Binary or Multi-class?)** <br/>
7. **Which machine learning algorithms can be used to classify the bean types?** <br/>
8. **Use Histogram plots to understand the numerical features.** <br/>
9. **Use Boxplot plots to understand the numerical features.** <br/>
10. **Use Correlation plot to understand any relationship between variables.** <br/>
11. **What performance metrics can be used to evaluate classification models trained on this dataset?** <br/>
12. **Use a Pipeline to preprocess and modeling your data.** <br/>
13. **Compare between diffferent models which one is more accurate.** <br/>
14. **Tune hyperparameters using Optuna to improve accuracy with RandomForestClassifier.** <br/>


**Citation: Dry Bean [Dataset](https://doi.org/10.24432/C50S4B). (2020). UCI Machine Learning Repository.** <br/>
--------------------------------------------------------------------------------------------------------------------


### I. How to import the dataset via pip.
--------------------------------------------------------------------------------------------------------------------
```
pip install ucimlrepo
Import the dataset into your code 
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
dry_bean = fetch_ucirepo(id=602) 
  
# data (as pandas dataframes) 
X = dry_bean.data.features 
y = dry_bean.data.targets 
  
# metadata 
print(dry_bean.metadata) 
  
# variable information 
print(dry_bean.variables)

```


### II. All packages used for this notebook.
--------------------------------------------------------------------------------------------------------------------
```
import gc # Garbage Collector

import pandas as pd
import numpy as np
import os

# Time Modules
import calendar
from time import time
import datetime
from datetime import datetime, timedelta

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Plots
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize':(18, 12)})
%matplotlib inline

# Statistics 
from scipy.stats import norm
from scipy.stats import zscore
from scipy import stats

import warnings
warnings.filterwarnings('ignore')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import ConfusionMatrixDisplay

```
