## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

  ```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
<img width="1033" height="614" alt="image" src="https://github.com/user-attachments/assets/119ee04a-16bc-4932-8695-4bf33ac017e7" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
<img width="1061" height="355" alt="image" src="https://github.com/user-attachments/assets/b3efd2a6-212e-40ff-9334-24e5c81d6181" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="1151" height="580" alt="image" src="https://github.com/user-attachments/assets/b948e27a-8c59-4e31-a068-428e0c9fa047" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="908" height="631" alt="image" src="https://github.com/user-attachments/assets/328a1cdc-391c-40bb-bbec-650aedc0a12a" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]])) # Orders in Alphabetical Order Blue , Green, Red
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="1064" height="684" alt="image" src="https://github.com/user-attachments/assets/08f6b428-1f74-4b0b-86b4-54d6f62718ec" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="942" height="494" alt="image" src="https://github.com/user-attachments/assets/ccc74520-b12c-4e1c-8420-f0852c5a6815" />

```
pip install --upgrade category_encoders
```
<img width="1388" height="499" alt="image" src="https://github.com/user-attachments/assets/7655e9d9-dec9-4aad-a4ab-9e8bbec22f99" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
<img width="1033" height="548" alt="image" src="https://github.com/user-attachments/assets/d43e5ecc-8f82-408a-a77f-e43a873e1b9f" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df=pd.concat([df,nd],axis=1)
df
```
<img width="988" height="572" alt="image" src="https://github.com/user-attachments/assets/efcce8a1-cb59-42b6-a7b5-9e74e8c682ed" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
<img width="980" height="624" alt="image" src="https://github.com/user-attachments/assets/cd1165d2-57d9-4df8-aa04-c4bfa82018cb" />

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
<img width="996" height="669" alt="image" src="https://github.com/user-attachments/assets/072fccae-df8d-4787-9400-f9c5d1e95e75" />

```
df.skew()
```
<img width="502" height="312" alt="image" src="https://github.com/user-attachments/assets/ee2acbbd-8454-43b4-b5a5-c75c72d62a18" />

```
np.log(df["Highly Positive Skew"])
```
<img width="591" height="617" alt="image" src="https://github.com/user-attachments/assets/39c81176-eb48-46aa-a876-7c035ad008ad" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="622" height="624" alt="image" src="https://github.com/user-attachments/assets/5b2e176a-8c16-4159-8fe1-5f2aa373791c" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="684" height="620" alt="image" src="https://github.com/user-attachments/assets/75468151-e8e4-4c4e-833e-d76c159e26f9" />

```
np.square(df["Highly Positive Skew"])
```
<img width="643" height="623" alt="image" src="https://github.com/user-attachments/assets/754015c9-d931-42c2-8332-b14c1d319b1f" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="1271" height="603" alt="image" src="https://github.com/user-attachments/assets/283a0e17-f120-4b1d-8b67-eec1013b96c1" />

```
df.skew()
```
<img width="610" height="347" alt="image" src="https://github.com/user-attachments/assets/a37dfe8b-2d87-4e04-8e94-c9d588ed61c6" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="1049" height="409" alt="image" src="https://github.com/user-attachments/assets/ec693cae-899f-447f-a532-70f20edc8c0f" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
<img width="1354" height="671" alt="image" src="https://github.com/user-attachments/assets/028e945c-5e09-433c-8b3b-f4266b88f122" />

```
import seaborn as sns
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT
plt.show()
```
<img width="995" height="707" alt="image" src="https://github.com/user-attachments/assets/e92e0d7b-9454-441c-bf46-fe3afc74ed85" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL
plt.show()
```
<img width="957" height="631" alt="image" src="https://github.com/user-attachments/assets/c2a36e28-3abd-4d6b-9503-fdd6592f9327" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="883" height="704" alt="image" src="https://github.com/user-attachments/assets/67e22499-5f23-4634-9dad-37dbb4160ecb" />

# RESULT:
    THE CODES ARE EXECUTED SUCCESSFULLY

       
