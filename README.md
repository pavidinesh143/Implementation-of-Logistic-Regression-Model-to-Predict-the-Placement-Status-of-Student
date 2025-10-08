# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model

2. Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3. Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4. Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5. Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.


## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: PAVITHRA S
RegisterNumber:  212223220072

  import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

'''

## Output:

1.Placement Data

<img width="1475" height="261" alt="237874710-906881a8-378a-496f-87fa-e04a6b68b48c" src="https://github.com/user-attachments/assets/182657fd-445a-4128-b836-57ec26e9e36a" />

2.Salary Data

<img width="1303" height="258" alt="237874789-ac65ed31-9e75-4718-a5b0-f76b2bd4f4e6" src="https://github.com/user-attachments/assets/f4c5a5ff-cb63-47b2-a7f6-6cce9221b9fd" />

3. Checking the null function()

<img width="288" height="327" alt="237874999-80254414-d19a-40d5-bcde-a5b2d2fdc320" src="https://github.com/user-attachments/assets/fa164c98-b8a2-4e91-a240-ebb7d9b5170e" />

4.Data Duplicate

<img width="348" height="48" alt="237875076-1b7713de-f9f6-4a4f-9d2a-1d3afd05bd75" src="https://github.com/user-attachments/assets/56c9df45-6a86-4ade-b5af-96d0c0254af1" />

5.Print Data

<img width="1192" height="517" alt="237876957-e672b342-e3a3-4964-a1d9-f66d73e09f0d" src="https://github.com/user-attachments/assets/f39e696f-af26-48f9-93e0-6013603d1e0d" />

<img width="1192" height="517" alt="237926889-927a0e92-056d-4252-a9e6-1973a03922e6" src="https://github.com/user-attachments/assets/4d0e2c17-8fad-4759-b127-07b3719bfde6" />

6.Data Status

<img width="630" height="283" alt="237875326-6b5f9885-327f-49b8-805a-a9f2a20b99c0" src="https://github.com/user-attachments/assets/81dbdfb8-5c21-4910-ac25-efcc64076422" />

7.y_prediction array

<img width="952" height="70" alt="237875300-981ddbed-7686-4f43-a896-09cb8725d9ad" src="https://github.com/user-attachments/assets/00072c66-cafd-43da-bb42-839f95b34471" />

8.Accuracy value

<img width="353" height="46" alt="237875389-d1978977-ab9a-4f59-a2fc-1944fdf34a8e" src="https://github.com/user-attachments/assets/739dfaf5-e461-462e-968b-e5850bbaa615" />

9.Confusion matrix

<img width="306" height="67" alt="237875778-2eb60e93-5b60-44bd-9dc3-8d7afc0ef62d" src="https://github.com/user-attachments/assets/0c5b7ae8-b44f-444c-8ebd-80f968eaea12" />

10.Classification Report

<img width="687" height="213" alt="237875860-6320a38a-dda5-4d3a-94b9-2d9dba8426a5" src="https://github.com/user-attachments/assets/df813a5f-df13-43dd-9196-d1adab2153d2" />

11.Prediction of LR

<img width="1496" height="68" alt="237875963-057a45e1-08e3-4de9-af44-9322010e588a" src="https://github.com/user-attachments/assets/1150136c-7b41-4164-8d42-d32b0b6e54a4" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
