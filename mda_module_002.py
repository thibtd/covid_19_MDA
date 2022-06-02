def counties_preprocessing(dataset):
    import pandas as pd
    import numpy as np
    
    """ The function pre-processes the counties' data sets."""
    
    # Missing values
    # For the counties New York City, Kansas City and Joplin the FIPS values are manually added.
    dataset.loc[dataset.county == "New York City", "fips"] = int(36061)
    dataset.loc[dataset.county == "Kansas City", "fips"] = int(20)
    dataset.loc[dataset.county == "Joplin", "fips"] = int(2937592)
    # Rows with NAs are dropped.
    dataset.dropna(axis=0, inplace=True)
    
    # Numerical variables are transformed to integers.
    dataset = dataset.astype({"fips": int, "cases": int, "deaths": int})
    
    # Date variable is split into Year, Month, Day variables
    dataset[["year", "month", "day"]] = dataset["date"].str.split("-", expand = True)
    
    # From the following source, the abbreviation code for the states are added.
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
    codes = list(df["code"])
    states = list(df["state"])
    codes_dict = {}
    
    for (state, code) in zip(states, codes):
        codes_dict.update({state:code})
    
    dataset['code'] = dataset['state'].map(codes_dict)
    
    # Column order is changed and dataset is sorted.
    col_order = ["date", "year", "month", "day", "state", "code", "county", "cases", "deaths"]
    dataset = dataset[col_order]
    dataset.sort_values(["date", "state"], inplace=True, ignore_index=True)
        
    return dataset

def per_state(dataset):
    import pandas as pd
    import numpy as np
    grouped_dataset = pd.DataFrame(dataset.groupby(["state", "code"])[["cases", "deaths"]].max()).reset_index()
    return grouped_dataset

def per_county(dataset):
    import pandas as pd
    import numpy as np
    grouped_dataset = pd.DataFrame(dataset.groupby(["state", "code", "county"])[["cases", "deaths"]].max()).reset_index()
    return grouped_dataset

def state_per_month(data):
    import os
    import pandas as pd
    import numpy as np
    
    """ The function pre-processes the states data set."""
    
    cwd = os.getcwd()
    
    dataset = data.copy(deep=True)
    
    # Numerical variables are transformed to integers.
    dataset = dataset.astype({"fips": int, "cases": int, "deaths": int})
    
    # Date variable is split into Year, Month, Day variables
    dataset[["year", "month", "day"]] = dataset["date"].str.split("-", expand = True)
    
    # dataset is grouped
    dataset = pd.DataFrame(dataset.groupby(["year", "month", "state"])[["cases", "deaths"]].max()).reset_index()
    dataset["date"] = dataset["year"]+"-"+dataset["month"]
    dataset.drop(["year", "month"], axis=1, inplace=True)
    
    # From the following source, the abbreviation code for the states are added.
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
    codes = list(df["code"])
    states = list(df["state"])
    codes_dict = {}
    
    for (state, code) in zip(states, codes):
        codes_dict.update({state:code})
    
    dataset['code'] = dataset['state'].map(codes_dict)
    
    # Latitude and Longitude data are added
    coords = pd.read_csv(cwd+"/statelatlong.csv")
    lat = list(coords["Latitude"])
    long = list(coords["Longitude"])
    name = list(coords["City"])
    lat_dict = {}
    long_dict = {}
    
    for (x,y,z) in zip(name, lat, long):
        lat_dict.update({x:y})
        long_dict.update({x:z})
    
    dataset['latitude'] = dataset['state'].map(lat_dict)
    dataset['longitude'] = dataset['state'].map(long_dict)
    
    col_order = ["date", "state", "code", "latitude", "longitude", "cases", "deaths"]
    dataset = dataset[col_order]
    
#     data_dict = {}
    
#     for (c, la, lo, s) in zip(codes, lat, long, states):
#         data_dict.update({s:[c, la, lo]})
    
#     for key, value in data_dict.items():
#         row = {'date': '2019-12', 'state': key, 'code': value[0], 'latitude': value[1], 'longitude': value[2], 'cases':0, 'deaths':0}
#         dataset = dataset.append(row, ignore_index = True)
    
    dataset.sort_values(["date", "state"], inplace=True, ignore_index=True)
    
    return dataset


def extra_data_retriever(dataset, df_to_merge):
    import pandas as pd
    
    cols_to_keep = ['Date', 'Recip_State', 'Administered_Dose1_Recip', 'Series_Complete_Yes', 'Census2019']
    new_dataset = dataset[cols_to_keep]
    
    new_dataset.rename(columns = {'Date':'date',
                                  'Recip_State':'code',
                                  'Administered_Dose1_Recip':'1_dose',
                                  'Series_Complete_Yes':'complete_dose', 'Census2019':'population'}, inplace = True)
    
    # Rows with NAs are dropped.
    new_dataset.dropna(axis=0, inplace=True)
    
    # Numerical variables are transformed to integers.
    new_dataset = new_dataset.astype({"1_dose": int, "complete_dose": int, "population": int})
    
    # Date variable is split into Year, Month, Day variables
    new_dataset[["month", "day", "year"]] = new_dataset["date"].str.split("/", expand = True)
    
       
    new_dataset = pd.DataFrame(new_dataset.groupby(["year", "month", "code"])[["1_dose", "complete_dose"]].max()).reset_index()
    new_dataset["date"] = new_dataset["year"]+"-"+new_dataset["month"]
    new_dataset.drop(["year", "month"], axis=1, inplace=True)
    
    new_dataset_merged = pd.merge(df_to_merge, new_dataset, on=['date', 'code'], how='outer')
    new_dataset_merged.sort_values("date", inplace=True)
    
    # From the following source, the abbreviation code for the states are added.
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
    codes = list(df["code"])
    states = list(df["state"])
    codes_dict = {}
    
    for (state, code) in zip(states, codes):
        codes_dict.update({code:state})
    
    new_dataset_merged['state'] = new_dataset_merged['code'].map(codes_dict)
    
    
    # Missing values
    new_dataset_merged.dropna(subset = ['state', 'code'], inplace=True)
    
    condition = new_dataset_merged["date"]<="2020-12"
    
    new_dataset_merged.loc[condition,'1_dose'] = new_dataset_merged.loc[condition,'1_dose'].fillna(0)
    new_dataset_merged.loc[condition,'complete_dose'] = new_dataset_merged.loc[condition,'complete_dose'].fillna(0)
    
    new_dataset_merged.dropna(axis=0, inplace=True)
    
    # Numerical variables are transformed to integers.
    new_dataset_merged = new_dataset_merged.astype({"cases": int, "deaths": int, "1_dose": int, "complete_dose": int})
    
    return new_dataset_merged
