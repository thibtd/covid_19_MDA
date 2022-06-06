# Modern Data Analytics [G0Z39a]

## Project: Covid 19 in the USA

The current study utilizes data from the New York Times to study the evolution of the Covid19 pandemic in the US

The mda_covid_011.ipynb notebook contains all the analysis that was conducted.

The mda_module_011.py contains all the necessary funtions for the notebook to work properly.

The covid-19-data folder contains the data sets taken from New York Times GitHub repo (https://github.com/nytimes/covid-19-data).

The data folder contains extra data that were used. Data were taken from the following sources:

- Vaccination data: https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh
 - Population data: https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh
- Geographical data: https://www.kaggle.com/datasets/washimahmed/usa-latlong-for-state-abbreviations
- Stock data: https://www.yahoofinanceapi.com

##  How to run the notebook with voila (the app deployment module):
the covid-19-data and the data folder that can be found on the google drive have to be unzipped and in the same folder as the notebooks. 
There are two ways of running the notebook with Voila: 
1) from the terminal: cd to the folder containing the code and the data then call $ voila mda_covid_011.ipynb 
2) If you have nbextensions, enable the voila/extension extension. Open the notebook in jupyter and click on the voila icon. 

However, it is preferable to use the 2nd option as the layout looks better. 

It takes about 2-3 minutes for voila to compile and launch the notebook. 
