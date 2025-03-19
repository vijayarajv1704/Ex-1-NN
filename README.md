<H3>ENTER YOUR NAME: Vijayaraj V</H3>
<H3>ENTER YOUR REGISTER NO: 212222230174</H3> 
<H3>EX. NO.1</H3>
<H3>DATE:</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

Import Libraries:
```
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
```

Read the dataset:
```
df=pd.read_csv("Churn_Modelling.csv")
df.head()
df.tail()
df.columns
```

Check the missing data:
```
df.isnull().sum()
df.duplicated()
```
Assigning Y
```
y = df.iloc[:, -1].values
print(y)
```

Check for duplicates:

```
df.duplicated()
```
Check for outliers:
```
df.describe()
```
Dropping string values data from dataset:
```
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
```
Checking datasets after dropping string values data from dataset:
```
data.head()
```
Normalize the dataset:
```
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
```

Split the dataset:
```
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
```
Training and testing model:
```
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))
```

## OUTPUT:

## DATA CHECKING:

![image](https://github.com/user-attachments/assets/03555884-861a-4428-8421-f81ef11b8686)

## MISSING DATA:

![image](https://github.com/user-attachments/assets/3a664406-3684-4c6d-87fa-24ce6e790a2e)


## DUPLICATES IDENTIFICATION:

![image](https://github.com/user-attachments/assets/f8c02477-e941-4d2b-9c31-ed02df4df973)


## VALUES OF 'Y':


![image](https://github.com/user-attachments/assets/fe63b364-b12e-4339-ae78-ac25fc300b9d)


## OUTLIERS:

![image](https://github.com/user-attachments/assets/1978dcb2-d997-42b7-8240-af3be52c0dc0)


## Checking datasets after dropping string values data from dataset:

![image](https://github.com/user-attachments/assets/9ef4a227-7645-4904-9e2f-2dbfc62791c6)


## NORMALIZE THE DATASET:

![image](https://github.com/user-attachments/assets/094928e8-a189-4d36-aa43-8670586e17d6)


## SPLIT THE DATASET:

![image](https://github.com/user-attachments/assets/388c2cbd-559c-4627-884d-03d2543940fa)


## TRAINING AND TESTING MODEL:

![image](https://github.com/user-attachments/assets/43ed6e58-83ac-425a-a3d5-cd6fa02c5340)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.
