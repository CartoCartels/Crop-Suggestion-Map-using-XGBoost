import os
os.popen("pip install geopandas")
# os.popen("pip install neuralprophet")
os.popen("pip install keplergl")
os.popen("pip install shiny")

import pickle
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
from shapely import wkt
from keplergl import KeplerGl

# from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
from datetime import datetime as d
from dateutil.relativedelta import relativedelta
plt.rcParams["figure.figsize"] = (80,13)

df_crop=pd.read_csv("cropData_Final.csv")
gdf_districts=pd.read_csv("districts-map.csv")


gdf_districts['geometry'] = gdf_districts['geometry'].apply(wkt.loads)
gdf_districts = gpd.GeoDataFrame(gdf_districts, geometry = 'geometry')


def caps_to_normal(district_name: str):
    return district_name.title()

df_crop['District']=df_crop['District_Name'].apply(caps_to_normal)
gdf_districts['District']=gdf_districts['District'].apply(caps_to_normal)
df_crop.drop(columns=['District_Name'],inplace=True)

"""### *Renaming districts*"""

def rename_districts(district_name):
    to_rename={
        'Thoothukkudi' : 'Thoothukudi',
        'Tirunelveli' : 'Thirunelveli',
        'Tiruchirappalli' : 'Trichirappalli',
        'Pudukkottai' : 'Pudukottai',
        'Sivaganga':'Sivagangai',
        'Kanniyakumari':'Kanyakumari'
    }
    if district_name in to_rename:
        return to_rename[district_name]
    return district_name

df_crop['District']=df_crop['District'].apply(rename_districts)
gdf_districts['District']=gdf_districts['District'].apply(rename_districts)


loc_data = pd.read_csv('TN_District_Lat_Lon.csv')


from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
from datetime import datetime
import random
import json

# Load the dataset
data = pd.read_csv('cropData_Final.csv')

"""###Label encoding

"""

le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()
le3 = preprocessing.LabelEncoder()

data['Season'] = le1.fit_transform(data['Season'])

data['District_Name']=le2.fit_transform(data['District_Name'])

data['Crop'] = le3.fit_transform(data['Crop'])

data.dropna(inplace=True)

oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(data.drop('Crop', axis=1), data['Crop'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


"""### C4.5 *ALGORITHM*"""

# Create a decision tree classifier using the C4.5 algorithm
clf = DecisionTreeClassifier(criterion='entropy') #hyperparameter tuning

# Train the classifier on the training data
clf.fit(X_train, y_train)

with open("clf.pkl", "wb") as f:
    pickle.dump(clf, f)


"""# XGBoost

### XGBoost Hyper parameter Tuning
"""

# Convert the training and testing data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the initial parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # Multiclass classification objective
    'num_class': len(data['Crop'].unique()),  # Number of classes
    'seed': 42,  # Random seed for reproducibility
}

# Perform hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5],
    'eta': [0.1, 0.2, 0.3],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(**params)

grid_search = GridSearchCV(
    estimator = xgb_model,
    param_grid = param_grid,
    scoring = 'accuracy',
    cv=3,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

"""### XGBoost algorithm"""

# Convert the training and testing data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set the parameters for XGBoost
params = {
    'objective': 'multi:softmax',  # Multiclass classification objective
    'num_class': len(data['Crop'].unique()),  # Number of classes
    'max_depth': 4,  # Maximum tree depth
    'eta': 0.3,  # Learning rate
    'seed': 42,  # Random seed for reproducibility
}

# Train the XGBoost model
num_rounds = 100  # Number of boosting rounds
model = xgb.train(params, dtrain, num_rounds)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)


def dateAfterMonths (months):
    return d.date(d.today()) + relativedelta(months=+months)

def month_to_season(months):
    if dateAfterMonths(months).month in [5,6,7]:
        return 0
    else:
        return 1

def getParam (months):
    new = pd.DataFrame()
    new['District_Name'] = df_crop['District'].unique()
    new['Crop_Year'] = pd.DataFrame([dateAfterMonths(months).year for i in new['District_Name']])
    new['Season'] = pd.DataFrame([month_to_season(months) for i in new['District_Name']])
    new['Area'] = pd.DataFrame([random.choice(data['Area']) for i in new['District_Name']])
    new['Production'] = pd.DataFrame([random.choice(data['Production']) for i in new['District_Name']])
    new_param = list()
    for i in df_crop['District'].unique():
        index = list(loc_data['DISTRICT']).index(i)
        wea = os.popen(f'curl -X GET "https://climate-api.open-meteo.com/v1/climate?latitude={loc_data.LATITUDE[index]}&longitude={loc_data.LONGITUDE[index]}&start_date=2023-06-22&end_date=2024-06-22&daily=temperature_2m_mean,shortwave_radiation_sum,relative_humidity_2m_mean,precipitation_sum&models=FGOALS_f3_H&min={str(dateAfterMonths(0))}&max={str(dateAfterMonths(months))}" -H "accect:application/json"').read()
        wea = json.loads(wea)
        new_param.append([wea['daily'][i][wea['daily']['time'].index(str(dateAfterMonths(0)))] for i in wea['daily'].keys()][1:])
    new_param = pd.DataFrame(new_param)
    new['T2M'] = new_param[0]
    new['RH2M'] = new_param[2]
    new['PREC'] = new_param[3]
    new['PAR'] = new_param[1]
    new['District_Name'] = le2.fit_transform(new['District_Name'])
    return new

def getPred (months):
    prediction = pd.DataFrame()
    prediction['District'] = df_crop['District'].unique()
    # prediction['Crop'] = clf.predict(new)
    try:
        with open("best_model.pkl", "rb") as f:
            best_model = pickle.load(f)
    except:
        pass
#    try:
#        with open("data.pkl", "rb") as f:
#            data = pickle.load(f)
#        if months not in data.keys():
#            new = getParam(months)
#        else:
#            new = data[months]
#    except:
#        pass
    prediction['Crop'] = best_model.predict(getParam(months))
    prediction['Crop'] = [le3.classes_[int(i)] for i in prediction['Crop']]
    return prediction


"""# ***Map coloring***"""

def getMap (months):
    gdf_pred=pd.merge(getPred(months),gdf_districts,on='District',how='outer')

    # Define a list of categorical colors
    colors = plt.cm.tab20.colors

    # Select a random subset of colors
    n_colors = len(gdf_pred['Crop'].dropna().unique())
    random_color_indices = np.random.choice(range(len(colors)), n_colors, replace=False)
    random_colors = [colors[i] for i in random_color_indices]

    # Plot the data using a categorical color map
    ax = gpd.GeoDataFrame(gdf_pred, geometry='geometry').plot(column='Crop', categorical=True, legend=True,
                                                            cmap=plt.cm.colors.ListedColormap(random_colors),
                                                            edgecolor='k', figsize=(70, 15))
    ax.set_title(f"Crop Prediction Map for {str(dateAfterMonths(months))}")
    # plt.show()
    return ax
