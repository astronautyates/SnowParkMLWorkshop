from datetime import datetime
from airflow.decorators import dag, task, task_group
from astro import sql as aql
from astro.files import File
from astro.sql.table import Table
import os
from astronomer.providers.snowflake.utils.snowpark_helpers import SnowparkTable

demo_database = "DEMO"
demo_schema = "DEMO"


@dag(
    dag_id="snowparkmodelbasic",
    default_args={
        "temp_data_output": "table",
        "temp_data_db": demo_database,
        "temp_data_schema": demo_schema,
        "temp_data_overwrite": True,
        "database": demo_database,
        "schema": demo_schema,
    },
    schedule_interval=None,
    start_date=datetime(2023, 4, 1),
)
def snowparkmodelbasic():
    _SNOWFLAKE_CONN_ID = "snowflake_default"

    load_file = aql.load_file(
        task_id=f"load_from_file",
        input_file=File(f"include/data/roses_raw.csv"),
        output_table=Table(
            name="TEST_TABLE",
            metadata={"database": "DEMO", "schema": "DEMO"},
            conn_id=_SNOWFLAKE_CONN_ID,
        ),
        if_exists="replace",
    )

    @task.snowpark_python()
    def transform_table(df: SnowparkTable):
        from snowflake.snowpark.functions import col

        filtered_df = df.filter(col("LEAF_SIZE_CM") >= 2)
        return filtered_df

    @task.snowpark_virtualenv(
        conn_id=_SNOWFLAKE_CONN_ID,
        requirements=["pandas", "scikit-learn"],
    )
    def feature_eng_in_snowpark(raw_table: SnowparkTable) -> SnowparkTable:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        df = raw_table.to_pandas()

        # converting column names to str for the Scaler
        df.columns = [str(col).replace("'", "").replace('"', "") for col in df.columns]

        df = pd.get_dummies(df, columns=["BLOOMING_MONTH"], drop_first=True)
        X = df.drop(["ROSE_TYPE", "INDEX"], axis=1)

        y = df["ROSE_TYPE"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )

        train_data = pd.concat([X_train_scaled, y_train], axis=1)
        test_data = pd.concat([X_test_scaled, y_test], axis=1)

        return train_data, test_data

    @task.snowpark_python()
    def create_model_reg(demo_database, demo_schema):
        from snowflake.ml.registry import model_registry

        model_registry.create_model_registry(
            session=snowpark_session,
            database_name=demo_database,
            schema_name=demo_schema,
        )

    @task.snowpark_virtualenv(requirements=["scikit-learn", "astro_provider_snowflake"])
    def train_classifier(tables, database_name, schema_name):
        from snowflake.ml.registry import model_registry
        from snowflake.ml.modeling.ensemble import RandomForestClassifier
        from uuid import uuid4
        from snowflake.ml.modeling.pipeline import Pipeline
        from snowflake.ml.modeling.metrics import accuracy_score, confusion_matrix

        train_data = tables[0]
        test_data = tables[1]

        feature_cols = [
            "PETAL_SIZE_CM",
            "STEM_LENGTH_CM",
            "LEAF_SIZE_CM",
            "BLOOMING_MONTH_May",
            "BLOOMING_MONTH_June",
            "BLOOMING_MONTH_July",
        ]
        label_col = ["ROSE_TYPE"]

        pipe = Pipeline(
            steps=[
                (
                    "classifier",
                    RandomForestClassifier(
                        input_cols=feature_cols,
                        label_cols=label_col,
                        n_estimators=1000,
                        random_state=23,
                    ),
                )
            ]
        )

        rf_model = pipe.fit(train_data)

        registry = model_registry.ModelRegistry(
            session=snowpark_session,
            database_name=database_name,
            schema_name=schema_name,
        )

        model_id = registry.log_model(
            model=rf_model,
            model_version=uuid4().urn,
            model_name="Roses",
            tags={"stage": "dev", "model_type": "RandomForestClassifier"},
        )

        model_predict = rf_model.predict(test_data)

        model_predict_snow = snowpark_session.create_dataframe(model_predict)

        acc = accuracy_score(
            df=model_predict_snow,
            y_true_col_names="ROSE_TYPE",
            y_pred_col_names="OUTPUT_ROSE_TYPE",
        )
        conf = confusion_matrix(
            df=model_predict_snow,
            y_true_col_name="ROSE_TYPE",
            y_pred_col_name="OUTPUT_ROSE_TYPE",
        )

        print("Accuracy: ", acc)
        print("Confusion Matrix: ", conf)

        return {"model_pred": model_predict}

    model_trained = (
        train_classifier(
            feature_eng_in_snowpark(transform_table(load_file)),
            demo_database,
            demo_schema,
        ),
    )

    model_reg = create_model_reg(demo_database=demo_database, demo_schema=demo_schema)

    model_reg >> model_trained


snowparkmodelbasic()
