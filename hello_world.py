#Loading the titanic data set
import seaborn as sns
import pandas as pd



print('Hello ')
sns.get_dataset_names()
#df = sns.load_dataset('titanic')

# if the data set is avaibale in seaborn again, we can use it from there.
# it is the simplest solution as as shown above. However, if it does not wok,
# we can proceed e.g. as shown below.

# if you need to download it, you can use the link: https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv
# this csv file seems to have the same stucture as the ne in seaborn. The version on Kaggle deviates a bit with respect to some columns.
# The raw link: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv and this is what we need to obtain
# if we directly download it through pandas.


file_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
df=pd.read_csv(file_url)

df.head()
