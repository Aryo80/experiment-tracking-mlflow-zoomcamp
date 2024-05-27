import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Set the MLflow tracking URI
mlflow.set_tracking_uri(uri="http://127.0.0.1:8585")

# Set the MLflow experiment
mlflow.set_experiment("nyc-green-taxi")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    # Start MLflow run
    with mlflow.start_run():
        # Load data
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        # Define and train the model
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        
        # Log the trained model as an artifact
        mlflow.sklearn.log_model(rf, "random_forest_model")

        # Log hyperparameters
        mlflow.log_params({
            "max_depth": rf.max_depth,
            "random_state": rf.random_state
        })

        # Evaluate the model
        y_pred = rf.predict(X_val)
        rmse = (mean_squared_error(y_val, y_pred))**0.5

        # Log evaluation metric
        mlflow.log_metric("RMSE", rmse)        

if __name__ == '__main__':
    run_train()
