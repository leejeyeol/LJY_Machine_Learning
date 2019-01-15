# HoML 2.2
# Multivariate Regression
# Dataset : CaliforniaHousing http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_housing_data(path='/media/leejeyeol/74B8D3C8B8D38750/Data/CaliforniaHousing'):
    return pd.read_csv(os.path.join(path,'cal_housing.data'))


data = load_housing_data()
data.hist(bins=50, figsize=(20,15))
plt.show()
print(1)