"""Kaggle_Titanic, 2/22/17, Sajad Azami"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from preprocessing import data_preparation as dp
from visualization import scatter

sns.set_style('white')
__author__ = 'sajjadaazami@gmail.com (Sajad Azami)'

# Loading dataset
train_data_set = dp.read_data('./data_set/train.csv')
test_data_set = dp.read_data('./data_set/test.csv')

# Filling missing values
train_data_set['Age'] = train_data_set['Age'].fillna(train_data_set['Age'].median())
test_data_set['Age'] = test_data_set['Age'].fillna(test_data_set['Age'].median())
train_data_set["Fare"] = train_data_set["Fare"].fillna(train_data_set["Fare"].median())
test_data_set["Fare"] = test_data_set["Fare"].fillna(test_data_set["Fare"].median())
train_data_set["Embarked"] = train_data_set["Embarked"].fillna("S")
test_data_set["Embarked"] = test_data_set["Embarked"].fillna("S")

# Categorical to numerical
# Convert the Embarked classes to integer form
train_data_set.loc[train_data_set["Embarked"] == "S", "Embarked"] = 0
train_data_set.loc[train_data_set["Embarked"] == "C", "Embarked"] = 1
train_data_set.loc[train_data_set["Embarked"] == "Q", "Embarked"] = 2
test_data_set.loc[test_data_set["Embarked"] == "S", "Embarked"] = 0
test_data_set.loc[test_data_set["Embarked"] == "C", "Embarked"] = 1
test_data_set.loc[test_data_set["Embarked"] == "Q", "Embarked"] = 2
# Convert the male and female groups to integer form
train_data_set.loc[train_data_set["Sex"] == "male", "Sex"] = 0
train_data_set.loc[train_data_set["Sex"] == "female", "Sex"] = 1
test_data_set.loc[test_data_set["Sex"] == "male", "Sex"] = 0
test_data_set.loc[test_data_set["Sex"] == "female", "Sex"] = 1

# convert from float to int
train_data_set['Fare'] = train_data_set['Fare'].astype(int)
test_data_set['Fare'] = test_data_set['Fare'].astype(int)
train_data_set['Age'] = train_data_set['Age'].astype(int)
test_data_set['Age'] = test_data_set['Age'].astype(int)
train_data_set['Embarked'] = train_data_set['Embarked'].astype(int)
test_data_set['Embarked'] = test_data_set['Embarked'].astype(int)
train_data_set['Sex'] = train_data_set['Sex'].astype(int)
test_data_set['Sex'] = test_data_set['Sex'].astype(int)

# Scatter plot numerical data
scatter.bar_plot_feature_vs_label(train_data_set, 'Survived',
                                  ['Pclass', 'SibSp', 'Parch'], 2, 2)

# Drop unnecessary columns, these columns won't be useful in analysis and prediction
train_data_set = train_data_set.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_data_set_ids = test_data_set['PassengerId']
test_data_set = test_data_set.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Peaks for survived/not survived passengers by their age
facet = sns.FacetGrid(train_data_set, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data_set['Age'].max()))
facet.add_legend()
plt.show()

# Split train and test
train_Y = train_data_set['Survived'].ix[0:690]
train_X = train_data_set.drop('Survived', axis=1).ix[0:690]

test_Y = train_data_set['Survived'].ix[691:]
test_X = train_data_set.drop('Survived', axis=1).ix[691:]

print(train_X.info())
print(test_X.info())

# Logistic Regression model
log_reg = LogisticRegression()
log_reg.fit(train_X, train_Y)
Y_pred = log_reg.predict(test_data_set)
submission = pd.DataFrame(pd.concat([test_data_set_ids, pd.DataFrame(Y_pred)], axis=1))
submission.to_csv(path_or_buf='logistic_regression_result.csv', index=False)

# Random Forest model
forest = RandomForestClassifier(max_features='sqrt')
parameter_grid = {
    'max_depth': [4, 5, 6, 7, 8],
    'n_estimators': [200, 210, 240, 250],
    'criterion': ['gini', 'entropy']
}

cross_validation = StratifiedKFold(train_Y, n_folds=5)
grid_search = GridSearchCV(forest,
                           param_grid=parameter_grid,
                           cv=cross_validation)
grid_search.fit(train_X, train_Y)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
Y_pred = grid_search.predict(test_data_set).astype(int)
submission = pd.DataFrame(pd.concat([test_data_set_ids, pd.DataFrame(Y_pred)], axis=1))
submission.to_csv(path_or_buf='random_forest.csv', index=False)
