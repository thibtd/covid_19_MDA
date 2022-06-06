def counties_preprocessing(dataset):
    import pandas as pd
    import numpy as np
    
    """ The function pre-processes the counties' data sets."""
    
    dataset_new = dataset.copy(deep=True)
    
    # Missing values
    # For the counties New York City, Kansas City and Joplin the FIPS values are manually added.
    dataset_new.loc[dataset_new.county == "New York City", "fips"] = int(36061)
    dataset_new.loc[dataset_new.county == "Kansas City", "fips"] = int(20)
    dataset_new.loc[dataset_new.county == "Joplin", "fips"] = int(2937592)
    # Rows with NAs are dropped.
    dataset_new.dropna(axis=0, inplace=True)
    
    # Numerical variables are transformed to integers.
    dataset_new = dataset_new.astype({"fips": int, "cases": int, "deaths": int})
    
    # Date variable is split into Year, Month, Day variables
    dataset_new[["year", "month", "day"]] = dataset_new["date"].str.split("-", expand = True)
    
    # From the following source, the abbreviation code for the states are added.
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
    codes = list(df["code"])
    states = list(df["state"])
    codes_dict = {}
    
    for (state, code) in zip(states, codes):
        codes_dict.update({state:code})
    
    dataset_new['code'] = dataset_new['state'].map(codes_dict)
    
    # Column order is changed and dataset is sorted.
    col_order = ["date", "year", "month", "day", "state", "code", "county", "cases", "deaths"]
    dataset_new = dataset_new[col_order]
    dataset_new.sort_values(["date", "state"], inplace=True, ignore_index=True)
        
    return dataset_new

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
    coords = pd.read_csv(cwd+"/data/statelatlong.csv")
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


def timeseries_process(inputdf, level):
    import pandas as pd
    import numpy as np
    
    df = inputdf.copy(deep=True)
    df['date']=pd.to_datetime(df['date'])
    
    # level: "us", "state", "county"
    
    if level == "us":
        df['daily_cases'] = df['cases'].diff()
        df['daily_deaths'] = df['deaths'].diff()
        df.fillna(0, axis=0, inplace=True)
        df = df.astype({"daily_cases": int, "daily_deaths": int})
        neg_dailycases = df[df['daily_cases']<0].index
        df.drop(neg_dailycases, axis=0, inplace=True)
        neg_dailydeaths = df[df['daily_deaths']<0].index
        df.drop(neg_dailydeaths, axis=0, inplace=True)
    
    elif level == "state":
        df.sort_values(['state','date'], inplace=True)
        states = df['state'].unique()
        df['daily_cases'] = np.zeros(df.shape[0])
        df['daily_deaths'] = np.zeros(df.shape[0])
        
        for state in states:
            df['daily_cases'][df['state']==state]=df.cases[df['state']==state].diff()
            df['daily_deaths'][df['state']==state]=df.deaths[df['state']==state].diff()
        
        df.fillna(0, axis=0, inplace=True)
        df = df.astype({"daily_cases": int, "daily_deaths": int})
        neg_dailycases = df[df['daily_cases']<0].index
        df.drop(neg_dailycases, axis=0, inplace=True)
        neg_dailydeaths = df[df['daily_deaths']<0].index
        df.drop(neg_dailydeaths, axis=0, inplace=True)
    
    elif level == "county":
        df.drop(["year", "month", "day", "code"], axis=1, inplace=True)
        df.sort_values(['county','date'], inplace=True)
        counties = df['county'].unique()
        df['daily_cases'] = np.zeros(df.shape[0])
        df['daily_deaths'] = np.zeros(df.shape[0])
        
        df['daily_cases'] = df[['county','cases']].groupby(by=["county"]).diff()
        df['daily_deaths'] = df[['county','deaths']].groupby(by=["county"]).diff()
        
        df.fillna(0, axis=0, inplace=True)
        df = df.astype({"daily_cases": int, "daily_deaths": int})
        neg_dailycases = df[df['daily_cases']<0].index
        df.drop(neg_dailycases, axis=0, inplace=True)
        neg_dailydeaths = df[df['daily_deaths']<0].index
        df.drop(neg_dailydeaths, axis=0, inplace=True)
    
    return df


def plot(df, level, x="date", y="daily_cases", state='Alabama', county="Abbeville"):
    import plotly.express as px
    import pandas as pd
    import numpy as np
    
    title_dictionary = {"daily_cases": "Daily Cases", "daily_deaths": "Daily Deaths"}
    
    if level == "us":
        title = '{} in the USA'.format(title_dictionary[y])
        ax = px.area(df, x, y, title=title, labels={y:title_dictionary[y], "date":"Date"})
        ax.show()
    elif level == "state":
        title = '{} in {}'.format(title_dictionary[y], state)
        ax = px.area(df[df['state']==state], x, y, title=title, labels={y:title_dictionary[y], "date":"Date"})
        ax.show()
    elif level == "county":
        title = '{} in {}'.format(title_dictionary[y], county)
        ax = px.area(df[df['county']==county], x, y, title=title, labels={y:title_dictionary[y], "date":"Date"})
        ax.show()
        
        
def com_filter(case, level, state, county):
    output.clear_output()
    if level == 'us':
        with output:
            plot(us_timeseries, level="us", y=case)
    elif level == "state":
        with output:
            display(dropdown_state)
            plot(state_timeseries, level="state", y=case, state=state)
    elif level == "county":
        with output:
            display(dropdown_county)
            plot(counties_timeseries, level="county", y=case, county=county)

def dropdown_case_eventhandler(change):
    com_filter(change.new, dropdown_level.value, dropdown_state.value, dropdown_county.value)

def dropdown_level_eventhandler(change):
    com_filter(dropdown_case.value, change.new, dropdown_state.value, dropdown_county.value)    
    
def dropdown_state_eventhandler(change):
    com_filter(dropdown_case.value, dropdown_level.value, change.new, dropdown_county.value)
    
def dropdown_county_eventhandler(change):
    com_filter(dropdown_case.value, dropdown_level.value, dropdown_state.value, change.new)        

    
def population_data(dataset):
    import pandas as pd
    
    cols_to_keep = ['Date','Recip_County', 'Recip_State', 'Census2019']
    new_dataset = dataset[cols_to_keep]
    
    new_dataset.rename(columns = {'Date':'date',
                                  'Recip_County':'county',
                                  'Recip_State':'code',
                                  'Census2019':'population'}, inplace = True)
    
    # Rows with NAs are dropped.
    new_dataset.dropna(axis=0, inplace=True)
    
    # Numerical variables are transformed to integers.
    new_dataset = new_dataset.astype({"population": int})
    
    # Date variable is split into Year, Month, Day variables
    new_dataset[["month", "day", "year"]] = new_dataset["date"].str.split("/", expand = True)
    new_dataset.drop(["day", "month"], axis=1, inplace=True)
    new_dataset = new_dataset.astype({"year": int})
    
    new_dataset = pd.DataFrame(new_dataset.groupby(["year", "county", "code"])[["population"]].max()).reset_index()
    new_dataset2 = pd.DataFrame(new_dataset.groupby(["year", "code"])[["population"]].sum()).reset_index()
    
    new_dataset20 = new_dataset2[new_dataset2["year"]==2020].reset_index()
    new_dataset21 = new_dataset2[new_dataset2["year"]==2021].reset_index()
    new_dataset22 = new_dataset2[new_dataset2["year"]==2022].reset_index()
    
    def population_dictionary(df):
        import pandas as pd
        
        codes = list(df["code"])
        populations = list(df["population"])
        population_dict = {}
        
        for c, p in zip(codes, populations):
            population_dict.update({c:int(p)})
        
        return population_dict
    
    return(population_dictionary(new_dataset20), population_dictionary(new_dataset21), population_dictionary(new_dataset22))


def cluster_process(df):
    import pandas as pd
    import os
    
    cwd = os.getcwd()
    
    df20 = df[df["date"]<"2021-01"]
    df21 = df[(df["date"]>="2021-01") & (df["date"]<"2022-01")]
    df22 = df[df["date"]>="2022-01"]
    
    df20_processed = pd.DataFrame(df20.groupby(["state", "code"])[["latitude", "longitude", "cases", "deaths", "1_dose", "complete_dose"]].max()).reset_index()
    df21_processed = pd.DataFrame(df21.groupby(["state", "code"])[["latitude", "longitude", "cases", "deaths", "1_dose", "complete_dose"]].max()).reset_index()
    df22_processed = pd.DataFrame(df22.groupby(["state", "code"])[["latitude", "longitude", "cases", "deaths", "1_dose", "complete_dose"]].max()).reset_index()
    
    df20_processed.set_index("state", inplace=True)
    df21_processed.set_index("state", inplace=True)
    df22_processed.set_index("state", inplace=True)
    
    extra_data = pd.read_csv(cwd+"/data/extra_data.csv")
    pop20, pop21, pop22 = population_data(extra_data)
    
    df20_processed['population'] = df20_processed['code'].map(pop20)
    df21_processed['population'] = df21_processed['code'].map(pop21)
    df22_processed['population'] = df22_processed['code'].map(pop22)
    
    df20_processed.drop(["code"], axis=1, inplace=True)
    df20_processed.drop(["Hawaii"], axis=0, inplace=True)
    df21_processed.drop(["code"], axis=1, inplace=True)
    df22_processed.drop(["code"], axis=1, inplace=True)
    
    
    def risk_categorizer20(ratio):
        if ratio<0.03:
            return "Green"
        elif 0.03<=ratio<0.06:
            return "Yellow"
        elif 0.06<=ratio<0.09:
            return "Orange"
        elif 0.09<=ratio<0.12:
            return "Red"
        elif ratio>=0.12:
            return "Black"
    
    def risk_categorizer21(ratio):
        if ratio<0.12:
            return "Green"
        elif 0.12<=ratio<0.15:
            return "Yellow"
        elif 0.15<=ratio<0.18:
            return "Orange"
        elif 0.18<=ratio<0.21:
            return "Red"
        elif ratio>=0.21:
            return "Black"
    
    def risk_categorizer22(ratio):
        if ratio<0.21:
            return "Green"
        elif 0.21<=ratio<0.24:
            return "Yellow"
        elif 0.24<=ratio<0.27:
            return "Orange"
        elif 0.27<=ratio<0.3:
            return "Red"
        elif ratio>=0.3:
            return "Black"
    
    df20_processed["risk_category"] = (df20_processed["cases"]/df20_processed["population"]).apply(risk_categorizer20)
    df21_processed["risk_category"] = (df21_processed["cases"]/df21_processed["population"]).apply(risk_categorizer21)
    df22_processed["risk_category"] = (df22_processed["cases"]/df22_processed["population"]).apply(risk_categorizer22)
    
    return(df20_processed, df21_processed, df22_processed)


def cluster_algorithm(df, algorithm):
    from sklearn.pipeline import Pipeline
    import sklearn.cluster as cl
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    import pandas as pd
    import numpy as np
    
    data = np.array(df.loc[:,["latitude", "longitude", "cases", "deaths", "1_dose", "complete_dose", "population"]].values)
    true_label_names = np.array(df.loc[:,["risk_category"]].values)
    
    label_encoder = LabelEncoder()
    true_labels = label_encoder.fit_transform(true_label_names)
    
    n_clusters = len(label_encoder.classes_)
    
    preprocessor = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=0)),
        ]
    )
    
    clusterer = Pipeline(
        [
            ("KMeans", cl.KMeans(n_clusters=n_clusters, init="k-means++", n_init=100, max_iter=500, random_state=0)),
            # ("hierarchical", cl.AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')),
            # ("spectral", cl.SpectralClustering(n_clusters=n_clusters)),
            # ("minibatch", cl.MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, batch_size=1024, random_state=0)),
            # ("feature", cl.FeatureAgglomeration(n_clusters=n_clusters, affinity='euclidean', linkage='ward'))
            
        ]
    )
    
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("clusterer", clusterer)
        ]
    )
    
    pipe.fit(data)
    
    pcadf = pd.DataFrame(pipe["preprocessor"].transform(data), columns=["PC1", "PC2"], index=df.index)
    pcadf["predicted_cluster"] = pipe["clusterer"][algorithm].labels_
    pcadf["true_label"] = label_encoder.inverse_transform(true_labels)
    
    Z = pipe["clusterer"][algorithm].labels_
    
    preprocessed_data = pipe["preprocessor"].transform(data)
    predicted_labels = pipe["clusterer"][algorithm].labels_
    silhouette = silhouette_score(preprocessed_data, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    
    return pcadf, Z, silhouette, ari


def anova_process(df):
    import pandas as pd
    import os
    
    cwd = os.getcwd()
    
    df20 = df[df["date"]<"2021-01"]
    df21 = df[(df["date"]>="2021-01") & (df["date"]<"2022-01")]
    df22 = df[df["date"]>="2022-01"]
    
    df20[["year", "month"]] = df20["date"].str.split("-", expand = True)
    df20.drop(["date"], axis=1, inplace=True)
    df20 = df20[df20["state"] != "Hawaii"]
    df20 = df20.astype({"year": int, "month": int})
    
    df21[["year", "month"]] = df21["date"].str.split("-", expand = True)
    df21.drop(["date"], axis=1, inplace=True)
    df21 = df21.astype({"year": int, "month": int})
    
    df22[["year", "month"]] = df22["date"].str.split("-", expand = True)
    df22.drop(["date"], axis=1, inplace=True)
    df22 = df22.astype({"year": int, "month": int})
    
    df20_processed = pd.DataFrame(df20.groupby(["state", "code", "month", "year"])[["latitude", "longitude", "cases", "deaths", "1_dose", "complete_dose"]].max()).reset_index()
    df21_processed = pd.DataFrame(df21.groupby(["state", "code", "month", "year"])[["latitude", "longitude", "cases", "deaths", "1_dose", "complete_dose"]].max()).reset_index()
    df22_processed = pd.DataFrame(df22.groupby(["state", "code", "month", "year"])[["latitude", "longitude", "cases", "deaths", "1_dose", "complete_dose"]].max()).reset_index()
    extra_data = pd.read_csv(cwd+"/data/extra_data.csv")
    pop20, pop21, pop22 = population_data(extra_data)
    
    df20_processed['population'] = df20_processed['code'].map(pop20)
    df21_processed['population'] = df21_processed['code'].map(pop21)
    df22_processed['population'] = df22_processed['code'].map(pop22)
    
    df20_processed.drop(["code"], axis=1, inplace=True)
    df21_processed.drop(["code"], axis=1, inplace=True)
    df22_processed.drop(["code"], axis=1, inplace=True)
    
    df21_processed["month"]+=12
    df22_processed["month"]+=24
    
    states20 = sorted(df20_processed["state"].unique())
    tempdf_list20 = []
    for s in states20:
        tempdf = df20_processed[df20_processed['state'] == s]
        tempdf['monthly_cases'] = abs(tempdf['cases'].diff())
        tempdf['monthly_deaths'] = abs(tempdf['deaths'].diff())
        tempdf['monthly_1dose'] = abs(tempdf['1_dose'].diff())
        tempdf['monthly_completedose'] = abs(tempdf['complete_dose'].diff())
        
        tempdf_list20.append(tempdf)
    
    df20_processed = pd.concat(tempdf_list20)
    
    is_NaN20 = df20_processed.isnull()
    row_has_NaN20 = is_NaN20.any(axis=1)
    rows_with_NaN20 = df20_processed[row_has_NaN20]
    rows_with_NaN20
    NaN_index20 = list(rows_with_NaN20.index)
    
    for i in NaN_index20:
        df20_processed.loc[i, "monthly_cases"] = df20_processed.loc[i, "cases"]
        df20_processed.loc[i, "monthly_deaths"] = df20_processed.loc[i, "deaths"]
        df20_processed.loc[i, "monthly_1dose"] = df20_processed.loc[i, "1_dose"]
        df20_processed.loc[i, "monthly_completedose"] = df20_processed.loc[i, "complete_dose"]

    states21 = sorted(df21_processed["state"].unique())
    tempdf_list21 = []
    for s in states21:
        tempdf = df21_processed[df21_processed['state'] == s]
        tempdf['monthly_cases'] = abs(tempdf['cases'].diff())
        tempdf['monthly_deaths'] = abs(tempdf['deaths'].diff())
        tempdf['monthly_1dose'] = abs(tempdf['1_dose'].diff())
        tempdf['monthly_completedose'] = abs(tempdf['complete_dose'].diff())
        tempdf_list21.append(tempdf)
    
    df21_processed = pd.concat(tempdf_list21)
    
    is_NaN21 = df21_processed.isnull()
    row_has_NaN21 = is_NaN21.any(axis=1)
    rows_with_NaN21 = df21_processed[row_has_NaN21]
    rows_with_NaN21
    NaN_index21 = list(rows_with_NaN21.index)
    
    for i in NaN_index21:
        df21_processed.loc[i, "monthly_cases"] = df21_processed.loc[i, "cases"]
        df21_processed.loc[i, "monthly_deaths"] = df21_processed.loc[i, "deaths"]
        df21_processed.loc[i, "monthly_1dose"] = df21_processed.loc[i, "1_dose"]
        df21_processed.loc[i, "monthly_completedose"] = df21_processed.loc[i, "complete_dose"]
    
    newtempdf_list21 = []
    for s21, s20 in zip(states21, states20):
        tempdf21 = df21_processed[df21_processed["state"]==s21]
        tempdf20 = df20_processed[df20_processed["state"]==s20]
        tempdf21.loc[min(tempdf21.index), "monthly_cases"] = abs(tempdf21.loc[min(tempdf21.index), "cases"]-tempdf20.loc[max(tempdf20.index), "cases"])
        tempdf21.loc[min(tempdf21.index), "monthly_deaths"] = abs(tempdf21.loc[min(tempdf21.index), "deaths"]-tempdf20.loc[max(tempdf20.index), "deaths"])
        tempdf21.loc[min(tempdf21.index), "monthly_1dose"] = abs(tempdf21.loc[min(tempdf21.index), "1_dose"]-tempdf20.loc[max(tempdf20.index), "1_dose"])
        tempdf21.loc[min(tempdf21.index), "monthly_completedose"] = abs(tempdf21.loc[min(tempdf21.index), "complete_dose"]-tempdf20.loc[max(tempdf20.index), "complete_dose"])
        newtempdf_list21.append(tempdf21)
    
    df21_processed_new = pd.concat(newtempdf_list21)
    
    states22 = sorted(df22_processed["state"].unique())
    tempdf_list22 = []
    for s in states22:
        tempdf = df22_processed[df22_processed['state'] == s]
        tempdf['monthly_cases'] = abs(tempdf['cases'].diff())
        tempdf['monthly_deaths'] = abs(tempdf['deaths'].diff())
        tempdf['monthly_1dose'] = abs(tempdf['1_dose'].diff())
        tempdf['monthly_completedose'] = abs(tempdf['complete_dose'].diff())
        tempdf_list22.append(tempdf)    
        
    df22_processed = pd.concat(tempdf_list22)
    
    is_NaN22 = df22_processed.isnull()
    row_has_NaN22 = is_NaN22.any(axis=1)
    rows_with_NaN22 = df22_processed[row_has_NaN22]
    rows_with_NaN22
    NaN_index22 = list(rows_with_NaN22.index)
    
    for i in NaN_index22:
        df22_processed.loc[i, "monthly_cases"] = df22_processed.loc[i, "cases"]
        df22_processed.loc[i, "monthly_deaths"] = df22_processed.loc[i, "deaths"]
        df22_processed.loc[i, "monthly_1dose"] = df22_processed.loc[i, "1_dose"]
        df22_processed.loc[i, "monthly_completedose"] = df22_processed.loc[i, "complete_dose"]
    
    newtempdf_list22 = []
    for s22, s21 in zip(states22, states21):
        tempdf22 = df22_processed[df22_processed["state"]==s22]
        tempdf21 = df21_processed[df21_processed["state"]==s21]
        tempdf22.loc[min(tempdf22.index), "monthly_cases"] = abs(tempdf22.loc[min(tempdf22.index), "cases"]-tempdf21.loc[max(tempdf21.index), "cases"])
        tempdf22.loc[min(tempdf22.index), "monthly_deaths"] = abs(tempdf22.loc[min(tempdf22.index), "deaths"]-tempdf21.loc[max(tempdf21.index), "deaths"])
        tempdf22.loc[min(tempdf22.index), "monthly_1dose"] = abs(tempdf22.loc[min(tempdf22.index), "1_dose"]-tempdf21.loc[max(tempdf21.index), "1_dose"])
        tempdf22.loc[min(tempdf22.index), "monthly_completedose"] = abs(tempdf22.loc[min(tempdf22.index), "complete_dose"]-tempdf21.loc[max(tempdf21.index), "complete_dose"])
        newtempdf_list22.append(tempdf22)
    
    df22_processed_new = pd.concat(newtempdf_list22)
    
    lmem = pd.concat([df20_processed, df21_processed_new, df22_processed_new])
    lmem = lmem.astype({"monthly_cases": int, "monthly_deaths": int, "monthly_1dose": int, "monthly_completedose": int})
        
    return(lmem)



#compute csr 
def comp_csr(data,date_end,state=True):
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import datetime as dt
    if state:
        date = dt.datetime.strptime(date_end,"%Y-%m-%d")
        csr={}
        data = data.dropna()
        states = data['state'].unique()
        for s in states:
            time = data[(data['date']==date)&(data['state']==s)]
            rate = (time['deaths'].values[0]/time['Pop'].values[0])*1000
            csr[s]=rate
        title = 'Cause-specific mortality ratio per 1000 people per state on {}'.format(date_end)
        ax = px.bar(x = csr.keys(), y=csr.values(), title=title)
        ax.show()
    else :
        date = dt.datetime.strptime(date_end,"%Y-%m-%d")
        pop = 329500000 #us population in 2020 
        time = data[data['date']==date]
        rate = (time['deaths'].values[0]/pop)*1000
        return round(rate,3)
    

    
    
    
def comp_cfr(data,date_end,state=True):
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import datetime as dt
# mucfr = death(t)/infected(t) *100 
    if state:
        data = data.dropna()
        states = data['state'].unique()
        date = dt.datetime.strptime(date_end,"%Y-%m-%d")
        cfr={}
        for s in states:
            time = data[(data['date']==date)&(data['state']==s)]
            rate = (time['deaths'].values[0]/time['cases'].values[0])*100
            cfr[s]=rate
        title = 'Case to fatality ratio in percent per state on {}'.format(date_end)
        ax = px.bar(x = cfr.keys(), y=cfr.values(), title=title)
        ax.show()
    else:
        date = dt.datetime.strptime(date_end,"%Y-%m-%d")
        time = data[data['date']==date]
        rate = (time['deaths'].values[0]/time['cases'].values[0])*100
        return round(rate,3) 
    
#compute changes over 7 days in % after each of those events
def changes(data,events):
    import datetime as dt
    import pandas as pd
    import numpy as np
    changes= list()
    for e in events:
        e_start = data['Close'][(data.index ==e)].values[0]
        e_end_d = dt.datetime.strptime(e, "%Y-%m-%d")+ dt.timedelta(days=7)
        e_end = data['Close'][(data.index ==e_end_d)].values[0]
        e_change=((e_end-e_start)/e_start)*100
        changes.append(e_change)
    return changes

#get stock data from yahoo finance
def get_stocks(stock1,stock2,start_date,end_date,name1,name2):
    import plotly.express as px
    import pandas as pd
    import numpy as np
    import yfinance as yf
    ticker1 = yf.Ticker(stock1)
    dfStock1 = ticker1.history(start=start_date, end=end_date)
    dfStock1['name'] = name1
    ticker2 = yf.Ticker(stock2)
    dfStock2 = ticker2.history(start=start_date, end=end_date)
    dfStock2['name'] = name2
    dfstocks = dfStock1.append(dfStock2)
    return [dfStock1,dfStock2,dfstocks]