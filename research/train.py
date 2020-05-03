# Importing the dependencies
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def train():
    # Loading the dataset
    df = pd.read_csv('../input/football_data.csv', index_col=False)

    # Dropping unnecessary columns
    df.drop(['Div', 'Date', 'HomeTeam', 'AwayTeam', 'Referee', 'HTR'],
            axis=1,
            inplace=True)

    # normalization
    data = df.drop(['FTR'], axis=1)
    x = data.values
    scaled = (data - data.min()) / (data.max() - data.min())
    data = pd.DataFrame(scaled)
    data = data.join(other=df['FTR'], how='left')

    # Create categorical features
    data['FTR'].replace(to_replace='H', value=0, inplace=True)
    data['FTR'].replace(to_replace='A', value=1, inplace=True)
    data['FTR'].replace(to_replace='D', value=2, inplace=True)
    data['FTR'] = to_categorical(data['FTR'].values)

    # Create model
    X = data.drop(['FTR'], axis=1)
    Y = data['FTR']

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        Y,
                                                        test_size=0.1,
                                                        random_state=0)

    random_forest_model = RandomForestClassifier()
    model = random_forest_model.fit(X_train, y_train)
    joblib.dump(model, '../models/model.joblib', compress=True)


if __name__ == '__main__':
    train()
