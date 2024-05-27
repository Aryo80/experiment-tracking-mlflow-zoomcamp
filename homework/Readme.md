## Train a model with autolog
We will train a RandomForestRegressor (from Scikit-Learn) on the taxi dataset.

We have prepared the training script train.py for this exercise, which can be also found in the folder homework.

The script will:
load the datasets produced by the previous step,
train the model on the training set,
calculate the RMSE score on the validation set.


![image](https://github.com/Aryo80/experiment-tracking-mlflow-zoomcamp/assets/55058593/83f29fad-90c5-488e-91a1-1a5ebf68853d)

## Tune model hyperparameters
Now let's try to reduce the validation error by tuning the hyperparameters of the RandomForestRegressor using hyperopt. We have prepared the script hpo.py for this exercise.

Your task is to modify the script hpo.py and make sure that the validation RMSE is logged to the tracking server for each run of the hyperparameter optimization (you will need to add a few lines of code to the objective function) and run the script without passing any parameters.

After that, open UI and explore the runs from the experiment called random-forest-hyperopt to answer the question below.
The idea is to just log the information that you need to answer the question below, including:

the list of hyperparameters that are passed to the objective function during the optimization,
the RMSE obtained on the validation set (February 2023 data).

![image](https://github.com/Aryo80/experiment-tracking-mlflow-zoomcamp/assets/55058593/b0b157e5-476a-494d-9e15-739cbb70fd38)


## Promote the best model to the model registry
The results from the hyperparameter optimization are quite good. So, we can assume that we are ready to test some of these models in production. In this exercise, you'll promote the best model to the model registry. We have prepared a script called register_model.py, which will check the results from the previous step and select the top 5 runs. After that, it will calculate the RMSE of those models on the test set (March 2023 data) and save the results to a new experiment called random-forest-best-models.

Your task is to update the script register_model.py so that it selects the model with the lowest RMSE on the test set and registers it to the model registry.

![image](https://github.com/Aryo80/experiment-tracking-mlflow-zoomcamp/assets/55058593/eb253315-0724-494c-b85c-7ed57b0b316b)




![image](https://github.com/Aryo80/experiment-tracking-mlflow-zoomcamp/assets/55058593/91a0933e-348b-46ea-8357-0f952e006d3c)
