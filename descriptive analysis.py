import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# Load the dataset
data = pd.read_csv("C:\\Users\\sanda\\OneDrive - University of Central Florida\\UCF\\4_Spring_2024\\Courses\\STA6367 Statistical Methodology for Data Science II\\Final Project\\glasses\\train.csv")

# Continuous variables as predictors
cols = [0, 513]
df = data.drop(data.columns[cols], axis=1)

# Select the first 11 rows for train set
trainX = df.iloc[0:3600, :]
testX = df.iloc[3600:4501, :]

# Select the target variable
trainY = data.iloc[0:3600]['glasses']
testY = data.iloc[3600:4500]['glasses']



# Normalize trainX
scaler = StandardScaler()
trainX_norm = scaler.fit_transform(trainX)
X_train = np.concatenate((np.ones((trainX_norm.shape[0], 1)), trainX_norm), axis=1)

y_train = trainY

# Normalize testX
testX_norm = scaler.transform(testX)
X_test = np.concatenate((np.ones((testX_norm.shape[0], 1)), testX_norm), axis=1)
y_test = testY



##############################################
################ Pie plot ####################
##############################################

# This assumes 'target' is the name of the column with your binary classes
class_distribution = data['glasses'].value_counts()
print(class_distribution)

total_samples = len(data)
minority_class_percentage = min(class_distribution) / total_samples

print(f"Minority class percentage: {minority_class_percentage:.2%}")
if minority_class_percentage > 0.2:
    print("The dataset is reasonably balanced.")
else:
    print("The dataset is unbalanced.")
    

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = class_distribution.index
sizes = class_distribution.values
explode = (0.1, 0)  # only "explode" the 1st slice (i.e., the first class)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#plt.title('Class Distribution')
plt.show()

##############################################
################## VIF #######################
##############################################

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# Function to calculate VIF for each feature
def calculate_vif(dataframe):
    # Adding a constant column for intercept
    df = dataframe.copy()
    df = sm.add_constant(df)
    
    # Calculating VIF for each feature
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df.columns
    vif_data['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    
    return vif_data

# Assuming 'df' is your DataFrame containing all numerical columns
vif_scores = calculate_vif(df)
print(vif_scores)

# Plotting VIF scores vs column names
plt.figure(figsize=(10, 6))
plt.barh(range(1, 513), vif_scores['VIF'])
plt.axvline(x=5, color='red', linestyle='--')  # Draw a vertical line at x=5
plt.xlabel('VIF Score')
plt.ylabel('Feature Number')
#plt.title('VIF Scores of Features')
plt.show()
