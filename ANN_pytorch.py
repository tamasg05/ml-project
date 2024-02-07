#Loading the titanic data set
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt

## ANN section
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def get_titanic_data():
  sns.get_dataset_names()
  #df = sns.load_dataset('titanic')

  # if the data set is avaibale in seaborn again, we can use it from there.
  # it is the simplest solution as as shown above. However, if it does not work always,
  # we can proceed e.g. as shown below.

  # if you need to download it, you can use the link: https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv
  # this csv file seems to have the same stucture as the ne in seaborn. The version on Kaggle deviates a bit with respect to some columns.
  # The raw link: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv and this is what we need to obtain
  # if we directly download it through pandas.


  file_url = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv'
  df=pd.read_csv(file_url)

  df.head()
  return df

def clean_titanic_data(df):

  # the deck column has a lot of NAs, replacing them with the median

  print(f'df columns data types: \n{df.dtypes}')
  print(f'\n\ndf.info() :-------------------------')
  df.info()

  # converting deck to category type
  df['deck'] = df['deck'].astype('category')

  # replacing NAs in column Deck
  df['deck'] = df["deck"].cat.add_categories("None").fillna("None")

  # replacing NAs with the median age in column Age
  df['age'] = df["age"].fillna(df['age'].median())

  # to check
  df.isna().sum()

  # dropping the remaining rows with NAs
  df = df.dropna()
  df.isna().sum()

  return df

def coding_data(df):
  # the deck column is a categorical variable with few number of categories -> one-hot encoding
  col = "deck"
  transformed_as_df = pd.get_dummies(df[col])
  coded_column_names = [col + "_" + column for column in transformed_as_df.columns]
  transformed_as_df.columns = coded_column_names

  df = pd.concat([df, transformed_as_df], axis=1)

  # label encoding boolean columns
  label_encoders = {}
  for col in ["adult_male", "alone"]:
      label_encoders[col] = LabelEncoder()
      df[col] = label_encoders[col].fit_transform(df[col])
  
  return df

def selecting_features(df, features = ['pclass', 'adult_male', 'alone', 'fare', 'deck_A', 'deck_B', 'deck_C', 'deck_D', 'deck_E', 'deck_F', 'deck_G', 'deck_None', 'survived']):
  # Selecting the relevant features
  df_selected_features = df[features]
  return df_selected_features

def splitting_scaling_features(df_selected_features, test_size=0.3):
  # Scaling the data
  # Before scaling, let's split the data into train and test sets, so that the test and train data could be independent while scaling

  # getting the inputs
  X = df_selected_features.iloc[:, :-1]
  # getting the output vars, i.e. the target
  y = df_selected_features.iloc[:, -1]

  # split the dataset
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=0) # random_state=0, i.e. setting the seed for random number generation, to be able to replicate the experiments

  # Scaling the data, fare has a lot of outliers; consequently, a non-standard scaling method is used

  # Normalization
  min_max_scaler = MinMaxScaler()
  col = 'fare'
  X_train[col] = min_max_scaler.fit_transform(X_train[[col]])
  X_test[col] = min_max_scaler.transform(X_test[[col]])

  # resetting indexes
  X_train.reset_index(inplace=True, drop=True)
  X_test.reset_index(inplace=True, drop=True)
  y_train.reset_index(inplace=True, drop=True)
  y_test.reset_index(inplace=True, drop=True)

  return (X_train, X_test, y_train, y_test)


#########################################################
## ANN section

def creating_nns():
  # Building the model
  # we have 12 input feature in the data set
  model_rlu = nn.Sequential(
      nn.Linear(12, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 12),
      nn.ReLU(),
      nn.Linear(12, 1),
      nn.Sigmoid()
  )

  model_th = nn.Sequential(
      nn.Linear(12, 128),
      nn.Tanh(),
      nn.Linear(128, 64),
      nn.Tanh(),
      nn.Linear(64, 64),
      nn.Tanh(),
      nn.Linear(64, 12),
      nn.Tanh(),
      nn.Linear(12, 1),
      nn.Sigmoid()
  )

  # by default it would expect float32 but I had float64 below
  # model_rlu.double()

  # Loss and optimizer
  criterion = nn.BCELoss()

  return (model_rlu, model_th, criterion)


# Training the model and checking accuracy with cross-validation
def full_gd(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=1000):

  # converting data frames to torch tensors
  X_train_t = torch.tensor(X_train.astype(np.float32).values)
  y_train_t = torch.tensor(y_train.astype(np.float32).values.reshape(-1, 1))

  X_test_t = torch.tensor(X_test.astype(np.float32).values)
  y_test_t = torch.tensor(y_test.astype(np.float32).values.reshape(-1, 1))

  # Stuff to store
  train_losses = np.zeros(epochs)
  ar_ac_train  = np.zeros(epochs)
  ar_ac_test  = np.zeros(epochs)


  for it in range(epochs):
    # zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)

    # Backward and optimize
    loss.backward()
    optimizer.step()

    # Save losses
    train_losses[it] = loss.item()

    # test accuracy on the test data and train data
    # and save them
    y_test_pred = model(X_test_t)
    ac_test = (y_test_pred.round() == y_test_t).float().mean()
    ac_train = (outputs.round() == y_train_t).float().mean()

    ar_ac_train[it] = ac_train
    ar_ac_test[it] = ac_test

#    if (it + 1) % 10 == 0:
#      print(f'Epoch {it+1}/{epochs}, Train Loss: {loss.item():.4f}, train accuracy: {ac_train}, test accuracy: {ac_test}')

  return (train_losses, ar_ac_train, ar_ac_test)

def train_nn(model, epoch, split):
  optimizer = torch.optim.Adam(model.parameters())
  kfold = StratifiedKFold(n_splits=split, shuffle=True)
  i=0
  arr_train = np.zeros(split)
  arr_test = np.zeros(split)
  for train, test in kfold.split(X_train, y_train):
    # create model, train, and get accuracy
    _, ac_trains_m, ac_tests_m = full_gd(model, criterion, optimizer, X_train.iloc[train], y_train.iloc[train], X_train.iloc[test], y_train.iloc[test], epoch)
    print(f'RLU, Accuracy, training: {ac_trains_m.mean()}, test: {ac_tests_m.mean()} ')
    arr_train[i] = ac_trains_m.mean()
    arr_test[i]  = ac_tests_m.mean()
    i = i + 1

  print(f'Accuracy for k-fold cross-validation, training: {arr_train.mean()}, test: {arr_test.mean()} \n')
  return (arr_train.mean(), arr_test.mean())

def plot_accuracy(num_epoch, ac_train_for_each_apoch, ac_test_for_each_apoch, title='Accuracy as a Function of the Epochs'):

  plt.figure(figsize=(8, 6))
  plt.plot(range(0, num_epoch), ac_test_for_each_apoch, label='Test Accuracy', color='blue', linewidth=2)
  plt.plot(range(0, num_epoch), ac_train_for_each_apoch, label='Train Accuracy', color='green', linewidth=1)
  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.legend()
  plt.grid(True)
  plt.show()
  plt.close()


titanic = get_titanic_data()
df = clean_titanic_data(titanic)
df_coded = coding_data(df)
df_features = selecting_features(df_coded)
X_train, X_test, y_train, y_test = splitting_scaling_features(df_features)
m_rlu, m_th, criterion = creating_nns()

# cross-validation
num_epoch = 30
split=5
train_nn(m_rlu, num_epoch, split)
train_nn(m_th, num_epoch, split)


###### Testing on the Test Data ########################################
optimizer = torch.optim.Adam(m_rlu.parameters())
_, ac_trains_rlu, ac_tests_rlu = full_gd(m_rlu, criterion, optimizer, X_train, y_train, X_test, y_test, num_epoch)

# illustrating the results
plot_accuracy(num_epoch, ac_trains_rlu, ac_tests_rlu, 'Accuracy as a Function of the Epochs')

