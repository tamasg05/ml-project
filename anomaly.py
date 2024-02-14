import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity


def reading_ds(file='worldwide_covid_data_2023_09.csv'):
    df_covid = pd.read_csv(file )

    # remove NAs
    df_covid.isna().sum()

    df_covid = df_covid.dropna()
    #print(df_covid.isna().sum())

    df_covid.reset_index(inplace = True, drop = True)

    return df_covid

def df_for_recommendation():

    df_covid = reading_ds()
    # selecting columns
    df = df_covid.iloc[:, [0, 5, 6, 8]]
    df.columns = ['country', 'total_cases_per_1m', 'death_per_1m', 'test_per_1m']

    country_list = df['country']

    df = df.set_index('country') # to have numeric cols only
    return (df, country_list)

def my_cosine_similarity(df, country_list):
    minMax_scale = MinMaxScaler()
    scaled_df = minMax_scale.fit_transform(df)

    cosine_sim_m = cosine_similarity(scaled_df, scaled_df)

    cosine_sim_df = pd.DataFrame(cosine_sim_m)
    cosine_sim_df.columns = country_list
    cosine_sim_df['country'] = country_list

    cosine_sim_df = cosine_sim_df.set_index('country') # to have numeric cols only
    return cosine_sim_df   

def get_similar_countries(cosine_sim_df, country_name, n_firsts=10):
  li = cosine_sim_df.loc[country_name, :]
  li = li.sort_values(ascending=False)

  # first n elements
  print(f'\n###### {country_name}, similar countries ####: \n {li[:n_firsts]}')


def plot_outlier_one_feature(df):
    sns.boxplot(df, x = 'Deaths/ 1M pop', color=".8", linecolor="#137", linewidth=.75 ).set(title="Covid Data")
    plt.show()

def print_outlier_death_rate(df, threshold = 4800):
    # selecting oulier countries based on the death cases projected on 1M population
    w = df.loc[df['Deaths/ 1M pop'] > threshold, :]
    print(f'Outliers for death rate: \n{w.sort_values('Deaths/ 1M pop', ascending=False)}')

def isolation_forest(df, df_full):
    # Isolation forest to locate the outliers, i.e. anomalies
    # https://www.analyticsvidhya.com/blog/2021/07/anomaly-detection-using-isolation-forest-a-complete-guide/
    # The variabe contamination marks the procent of the data we assume to be anomalouos
    # it would be not bad to try Extended Isolation Forest

    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(df)
    df_full['if_labels'] = pd.DataFrame(anomalies)
    
    df_full_subset = df_full.loc[df_full['if_labels'] == -1, :]
    return df_full_subset

    

def main_outlier_detection():
    df = reading_ds()
    plot_outlier_one_feature(df)
    print_outlier_death_rate(df)

    # Outliers with several features
    ################################

    # Selecting relative features
    df_mod = df[['Deaths/ 1M pop', 'Tests/ 1M pop']]

    # Feature Transformation
    minMax_scale = MinMaxScaler()
    df_covid_scaled = minMax_scale.fit_transform(df_mod)

    # isolation forest
    df_covid_scaled = pd.DataFrame(df_covid_scaled)
    anomalous_countries = isolation_forest(df_covid_scaled, df)

    # listing by death /1M population
    anomalous_countries = anomalous_countries.sort_values('Deaths/ 1M pop', ascending=False)
    print(f'######### Anomalous contries listed by death per 1M population: \n{anomalous_countries.to_string()} \n\n') # to_string() for getting the whole table printed

    # listing by tests /1M population
    anomalous_countries = anomalous_countries.sort_values('Tests/ 1M pop', ascending=False)
    print(f'######### Anomalous contries listed by tests per 1M population: \n{anomalous_countries.to_string()} \n\n') # to_string() for getting the whole table printed

def main_recommender_engine():
    df, country_list = df_for_recommendation()
    cos_m = my_cosine_similarity(df, country_list)
    print('Type in the country for which you are looking for similarities:')
    for line in sys.stdin:
        line = line.rstrip()
        if 'q' == line:
            break
        non_occurence = country_list.isin([line]).value_counts()[False]
        length = country_list.shape
        if non_occurence == length:
            print('Country not in the list. Type in a country or exit by typing \'q\'!')
            continue

        get_similar_countries(cos_m, line, 20) 

    print("Exiting....")   


if __name__ == '__main__':
    #main_outlier_detection()
    main_recommender_engine()


