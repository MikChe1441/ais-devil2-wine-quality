import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import json
import dagshub
import mlflow

dagshub.init(repo_owner='MikChe1441', repo_name='ais-devil2-wine-quality', mlflow=True)


mlflow.autolog()
mlflow.set_experiment("Wine_Quality_CatBoost_Exp")

df = pd.read_parquet('data/winequality.parquet')
X = df.drop('quality', axis=1)
y = df['quality']

X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.2,random_state=7)


with mlflow.start_run(run_name = 'Catboost_3'):
    clf = CatBoostClassifier(cat_features=['wine_color'],
                             auto_class_weights='Balanced',
                             iterations=7000,
                             depth=8,
                             verbose=False)
    clf.fit(X_train, y_train,eval_set=(X_test, y_test),verbose=False)
    y_pred = clf.predict(X_test)

    with open('models/wine_model3.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)

    report_dict = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("accuracy", report_dict["accuracy"])
    mlflow.log_metric("macro_f1", report_dict["macro avg"]["f1-score"])

    with open('models/wine_quality_model.metadata.json', 'w') as f:
        json.dump(report_dict, f, indent=2)

    mlflow.log_artifact('models/wine_model3.pkl')