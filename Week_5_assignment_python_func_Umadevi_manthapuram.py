import pandas as pd
from pycaret.classification import ClassificationExperiment

def load_data(filepath):
    """
    Loads diabetes data into a DataFrame from a string filepath.
    """
    return pd.read_csv(filepath)

def make_predictions(df):
    """
    Uses the pycaret best model to make predictions on data in the df dataframe.
    """
    classifier = ClassificationExperiment()
    model = classifier.load_model('pycaret_model')
    predictions = classifier.predict_model(model, df)
    churn_prob = predictions["Churn"]  # Assuming single row prediction
    return churn_prob



if __name__ == "__main__":
    df = load_data('/Users/arungajjela/Documents/Uma/MSDS600/MSDS600 Week2/prepared_churn_data.csv')
    predictions = make_predictions(df)
    print('predictions:')
    print(predictions)