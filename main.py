import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def prepare_country_stats(first_file, second_file):

	data1 = pd.DataFrame( first_file, columns=['Country', 'Value'])
	data2 = pd.DataFrame( second_file, columns=['Country', 'GDP per capita'])

	merged_data =pd.merge(data1,data2,  on='Country', how="left")
	#print_full(merged_data)
	return merged_data





# Load the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv",sep=',', engine='python')
#gdp_per_capita=pd.read_csv("gdp_per_capita.csv",sep=',', engine='python')

gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t',encoding='latin1', na_values="0")
gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)



# Prepare the data
country_stats = prepare_country_stats(oecd_bli.fillna(oecd_bli.mean()), gdp_per_capita.fillna(0))

X = np.c_[country_stats["GDP per capita"].fillna(0)] #fillna function replaces NaN into integer values
y = np.c_[country_stats['Value'].fillna(0)]


country_stats.plot(kind='scatter', x="GDP per capita", y='Value')


# Select a linear model
lin_reg_model = KNeighborsRegressor(n_neighbors=3) # Train the model with a KNeighborsRegressor with n=3
lin_reg_model.fit(X, y)

print("working...")
new_x= [[80271]]

print(lin_reg_model.predict(new_x))
plt.show()
# before fitting the model we change the NaN values to mean values or lese the fit function will not work
# Make a prediction for Cyprus
#X_new = [[22587]] # Cyprus' GDP per capita



#print(lin_reg_model.predict(valx))
#plt.show() # outputs [[ 5.96242338]]