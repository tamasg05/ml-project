import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest


def reading_ds(file='worldwide_covid_data_2023_09.csv'):
    df_covid = pd.read_csv(file )

    # remove NAs
    df_covid.isna().sum()

    df_covid = df_covid.dropna()
    print(df_covid.isna().sum())

    df_covid.reset_index(inplace = True, drop = True)
    df_covid.head()

    return df_covid

def plot_outlier_one_feature(df):
    sns.boxplot(df, x = 'Deaths/ 1M pop', color=".8", linecolor="#137", linewidth=.75 ).set(title="Covid Data")
    plt.show()

def print_outlier_death_rate(threshold = 4800):
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

    




df = reading_ds()
plot_outlier_one_feature(df)
print_outlier_death_rate()

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



