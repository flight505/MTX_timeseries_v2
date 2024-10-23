import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import (
    MissingValuesFiller,
    Scaler,
    StaticCovariatesTransformer,
)
from darts.metrics import mae, mape, r2_score, rmse
from darts.models import ExponentialSmoothing, TFTModel
from darts.utils.timeseries_generation import (
    datetime_attribute_timeseries,  # This is the correct import for datetime attributes
)
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# Load the data
df = pd.read_csv(
    "data/df_blood_preprocessed.csv",
)

# Convert date columns to datetime
df["ds"] = pd.to_datetime(df["ds"])
df["first_infusion_date"] = pd.to_datetime(df["first_infusion_date"])
df["next_infusion"] = pd.to_datetime(df["next_infusion"])

# Define data types for consistency
dtype_map = {
    "ds": "datetime64[ns]",
    "first_infusion_date": "datetime64[ns]",
    "next_infusion": "datetime64[ns]",
    "unique_id": str,  # Ensure unique_id is a string for anonymization
    "sex": int,
    "age_at_diagdate": float,
    "weight_change_cycles": float,
    "component": str,
    "value": float,
    "unit": str,
    "days_since_first_infusion": int,
    "infno_day": int,
    "infno": int,
}

df = df.astype(dtype_map)

# print the unique components
print(df["component"].unique())


# Anonymize unique_id by mapping digits to letters
def anonymize_unique_id(df):
    # Create a mapping dictionary to replace digits with letters
    digit_to_letter = {str(i): chr(ord("a") + i - 1) for i in range(1, 10)}
    digit_to_letter["0"] = "j"

    # Apply the mapping to the 'unique_id' column
    df["unique_id"] = (
        df["unique_id"]
        .astype(str)
        .apply(lambda x: "".join(digit_to_letter.get(d, "k") for d in x))
    )

    return df


df = anonymize_unique_id(df)

# Drop unnecessary columns
df = df.drop(columns=["unit", "transer", "transth"])

# Define the target variable
TARGET_COLUMN = "Neutrophilocytes_B"


# TimeSeriesPreprocessor class to handle data preprocessing
class TimeSeriesPreprocessor:
    def __init__(
        self,
        target_column=None,
        complete_timeline=False,
        encode_temporal_distance=False,
        add_decay=False,
        fill_residual_nan=False,
        exclude_columns=None,
    ):
        self.target_column = target_column
        self.complete_timeline = complete_timeline
        self.encode_temporal_distance = encode_temporal_distance
        self.add_decay = add_decay
        self.decay_params = {}
        self.fill_residual_nan = fill_residual_nan
        self.exclude_columns = exclude_columns or []

    def _aggregate_daily(self, X):
        X["ds_date"] = X["ds"].dt.date
        grouped = X.groupby(
            ["unique_id", "ds_date", "age_at_diagdate", "sex", "infno", "component"]
        )
        daily_data = grouped.agg({"value": "mean"}).reset_index()
        return daily_data

    def _initial_preparation(self, X):
        X = self._aggregate_daily(X)
        X["ds_date"] = pd.to_datetime(X["ds_date"])
        X["normalized_time"] = X.groupby("unique_id")["ds_date"].transform(
            lambda x: (x - x.min()).dt.days
        )

        X = X.pivot_table(
            index=[
                "unique_id",
                "normalized_time",
                "age_at_diagdate",
                "sex",
                "infno",
                "ds_date",
            ],
            columns="component",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Rename the target column to 'y'
        if self.target_column in X.columns:
            X = X.rename(columns={self.target_column: "y"})
        else:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame."
            )

        return X

    def _calculate_infno_day(self, df):
        df["infno_first_time"] = df.groupby(["unique_id", "infno"])[
            "normalized_time"
        ].transform("min")
        df["infno_day"] = df["normalized_time"] - df["infno_first_time"]
        df.drop(columns=["infno_first_time", "ds_date"], inplace=True)
        return df

    def prepare_and_process(self, df):
        """Main method to prepare and process the data"""
        # Make a copy
        df = df.copy()

        # First, pivot the data to wide format
        df_wide = df.pivot_table(
            index=["unique_id", "ds", "age_at_diagdate", "sex", "infno"],
            columns="component",
            values="value",
            aggfunc="first",
        ).reset_index()

        # Create time series list
        series_list = []
        covariates_list = []

        for unique_id in df_wide["unique_id"].unique():
            # Get data for this ID
            id_data = df_wide[df_wide["unique_id"] == unique_id].copy()

            # Sort by date
            id_data = id_data.sort_values("ds")

            # Separate numeric and non-numeric columns
            numeric_cols = id_data.select_dtypes(include=[np.number]).columns
            non_numeric_cols = id_data.select_dtypes(exclude=[np.number]).columns

            # Set index for resampling
            id_data.set_index("ds", inplace=True)

            # Resample numeric columns
            numeric_resampled = id_data[numeric_cols].resample("D").mean()

            # Forward fill non-numeric columns (except 'ds' which is now index)
            non_numeric_resampled = id_data[non_numeric_cols].resample("D").ffill()

            # Combine resampled data
            id_data = pd.concat([numeric_resampled, non_numeric_resampled], axis=1)

            # Forward and backward fill to handle any remaining NaN values
            id_data = id_data.ffill().bfill()

            # Reset index to get 'ds' back as a column
            id_data.reset_index(inplace=True)

            # Create target series
            target_series = TimeSeries.from_dataframe(
                id_data,
                time_col="ds",
                value_cols=[self.target_column],
                freq="D",
                fill_missing_dates=True,
            )

            # Create covariates
            static_covs = ["age_at_diagdate", "sex"]
            covariate_cols = [
                col
                for col in id_data.columns
                if col
                not in ["ds", "unique_id", self.target_column, "infno"] + static_covs
            ]

            covariate_series = TimeSeries.from_dataframe(
                id_data,
                time_col="ds",
                value_cols=covariate_cols,
                freq="D",
                static_covariates={cov: id_data[cov].iloc[0] for cov in static_covs},
                fill_missing_dates=True,
            )

            series_list.append(target_series)
            covariates_list.append(covariate_series)

        return series_list, covariates_list

    def _fill_residual_nan(self, df):
        """Fill remaining NaN values using forward fill and backward fill within each group"""
        feature_columns = [col for col in df.columns if col not in ["unique_id", "ds"]]

        # First, forward fill
        df[feature_columns] = df.groupby("unique_id")[feature_columns].ffill()

        # Then, backward fill to handle leading NaNs
        df[feature_columns] = df.groupby("unique_id")[feature_columns].bfill()

        return df

    def _complete_timeline(self, df):
        unique_ids = df["unique_id"].unique()
        complete_timeline = []

        for uid in unique_ids:
            uid_df = df[df["unique_id"] == uid]
            min_time = uid_df["normalized_time"].min()
            max_time = uid_df["normalized_time"].max()

            timeline = pd.DataFrame(
                {"unique_id": uid, "normalized_time": np.arange(min_time, max_time + 1)}
            )
            complete_timeline.append(timeline)

        complete_timeline = pd.concat(complete_timeline, ignore_index=True)

        df = pd.merge(
            complete_timeline, df, on=["unique_id", "normalized_time"], how="left"
        )
        static_columns = ["unique_id", "age_at_diagdate", "sex"]
        for col in static_columns:
            df[col] = df.groupby("unique_id")[col].ffill().bfill()

        dynamic_columns = ["infno"]
        for col in dynamic_columns:
            df[col] = df.groupby("unique_id")[col].ffill()

        return df

    def _simple_missingness_encoding(self, df):
        missingness_indicators = (
            df.loc[:, ~df.columns.isin(self.exclude_columns)].isna().astype(int)
        )
        missingness_indicators = missingness_indicators.add_suffix("_missing")
        df = pd.concat([df, missingness_indicators], axis=1)
        return df

    def _encode_temporal_distance(self, df):
        feature_columns = [
            col
            for col in df.columns
            if col not in self.exclude_columns and not col.endswith("_delta")
        ]
        temporal_distance_data = {}
        for col in feature_columns:
            if not col.endswith("_delta"):
                # Calculate temporal distance
                temporal_distance_series = (
                    df.groupby("unique_id")[col]
                    .apply(
                        lambda x: x.isna()
                        .astype(int)
                        .groupby(x.notna().astype(int).cumsum())
                        .cumsum()
                    )
                    .astype(float)
                )
                temporal_distance_data[col + "_delta"] = temporal_distance_series

        temporal_distance_df = pd.DataFrame(temporal_distance_data)
        temporal_distance_df = temporal_distance_df.set_index(df.index)

        df = pd.concat([df, temporal_distance_df], axis=1)
        return df

    def _estimate_decay_parameters(self, df):
        decay_params = {}
        for feature in [
            col
            for col in df.columns
            if not col.endswith(("_delta", "_missing"))
            and col not in self.exclude_columns
        ]:
            df[feature + "_last"] = df.groupby("unique_id")[feature].ffill()
            feature_df = df[[feature + "_delta", feature, feature + "_last"]].dropna()
            X_train = feature_df[feature + "_delta"].values
            y_train = feature_df[feature].values
            y_last = feature_df[feature + "_last"].values
            X_train_normalized = (X_train - X_train.min()) / (
                X_train.max() - X_train.min()
            )

            def objective(params):
                W, b = params
                decay_factor = np.exp(-np.maximum(0, W * X_train_normalized + b))
                loss = np.mean(np.abs(y_train - (y_last * decay_factor)))
                return loss

            initial_params = [0.0, 0.0]
            result = minimize(
                objective,
                initial_params,
                method="L-BFGS-B",
                bounds=[(0, None), (None, None)],
            )
            if result.success:
                W_opt, b_opt = result.x
                decay_params[feature] = {"W": W_opt, "b": b_opt}
            df.drop(columns=[feature + "_last"], inplace=True)
        self.decay_params = decay_params

    def _apply_decay(self, df):
        df_copy = df.copy()
        last_columns = {}
        for feature in self.decay_params:
            last_columns[feature + "_last"] = df_copy.groupby("unique_id")[
                feature
            ].ffill()
        last_df = pd.DataFrame(last_columns)
        for feature in self.decay_params:
            W = self.decay_params[feature]["W"]
            b = self.decay_params[feature]["b"]
            decay_factor = np.exp(-np.maximum(0, W * df_copy[feature + "_delta"] + b))
            df_copy[feature] = last_df[feature + "_last"] * decay_factor
            df_copy[feature] = df_copy[feature].fillna(last_df[feature + "_last"])
        df_copy = df_copy.drop(
            columns=[col for col in df_copy if col.endswith("_last")]
        )
        return df_copy

    def _remove_constant_columns(self, df):
        # Remove constant columns
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
        df = df.drop(columns=constant_columns)
        return df


# Initialize the preprocessor with appropriate parameters
preprocessor = TimeSeriesPreprocessor(
    target_column=TARGET_COLUMN,
    complete_timeline=True,
    encode_temporal_distance=True,
    add_decay=True,
    fill_residual_nan=True,
    exclude_columns=[
        "unique_id",
        "normalized_time",
        "sex",
        "age_at_diagdate",
        "infno",
        "infno_day",
        "ds_date",
    ],
)

print("\nFirst few rows of the DataFrame:")
print(df.head())

# Apply preprocessing to the DataFrame
processed_df = preprocessor.prepare_and_process(df)

# Update the list of component names after preprocessing
exclude_columns = {
    "unique_id",
    "normalized_time",
    "sex",
    "age_at_diagdate",
    "infno",
    "infno_day",
    "ds_date",
}

component_names = [col for col in processed_df.columns if col not in exclude_columns]
print(f"Number of components: {len(component_names)}")


# Function to split data into train, validation, and test sets
def split_data(unique_ids, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets.

    Args:
        unique_ids: Array of unique patient identifiers
        test_size: Proportion of data to use for testing
        val_size: Proportion of data to use for validation
    """
    np.random.seed(42)  # For reproducibility

    # Sort IDs to ensure reproducibility
    unique_ids = np.sort(unique_ids)

    # Calculate split sizes
    n_samples = len(unique_ids)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)

    # Create splits ensuring no data leakage
    test_ids = unique_ids[-n_test:]
    val_ids = unique_ids[-(n_test + n_val) : -n_test]
    train_ids = unique_ids[: -(n_test + n_val)]

    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()


# Function to create TimeSeries objects from a DataFrame
def create_time_series(patient_data, value_cols, static_covariates):
    """Create TimeSeries objects from patient data.

    Args:
        patient_data: DataFrame containing patient time series data
        value_cols: List of columns containing target values
        static_covariates: DataFrame/Series containing static covariates
    """
    # Convert normalized_time to datetime
    patient_data = patient_data.copy()
    patient_data["date"] = pd.to_datetime(patient_data["normalized_time"], unit="D")
    patient_data = patient_data.sort_values("date")

    # Create TimeSeries with proper frequency and handling of missing values
    ts = TimeSeries.from_dataframe(
        df=patient_data,
        time_col="date",
        value_cols=value_cols,
        static_covariates=static_covariates,
        fill_missing_dates=True,
        freq="D",
        fillna_value=None,  # Let Darts handle missing values
    )

    return ts


# Prepare the data for model training
unique_ids = processed_df["unique_id"].unique()
train_ids, val_ids, test_ids = split_data(unique_ids)

# Initialize lists for storing time series
train_list, val_list, test_list = [], [], []
train_covariates_list, val_covariates_list, test_covariates_list = [], [], []

for unique_id in unique_ids:
    patient_data = processed_df[processed_df["unique_id"] == unique_id]
    static_covariates = patient_data[["sex", "age_at_diagdate"]].iloc[0]
    value_cols = ["y"]
    # Create main series
    patient_series = create_time_series(patient_data, value_cols, static_covariates)
    # Create past covariates series
    covariate_cols = ["infno", "infno_day"]
    covariate_series = create_time_series(patient_data, covariate_cols, None)
    # Append to the appropriate list
    if unique_id in train_ids:
        train_list.append(patient_series)
        train_covariates_list.append(covariate_series)
    elif unique_id in val_ids:
        val_list.append(patient_series)
        val_covariates_list.append(covariate_series)
    else:
        test_list.append(patient_series)
        test_covariates_list.append(covariate_series)


# Convert series to float32
def convert_to_float32(ts_list):
    return [ts.astype(np.float32) for ts in ts_list]


train_list = convert_to_float32(train_list)
val_list = convert_to_float32(val_list)
test_list = convert_to_float32(test_list)
train_covariates_list = convert_to_float32(train_covariates_list)
val_covariates_list = convert_to_float32(val_covariates_list)
test_covariates_list = convert_to_float32(test_covariates_list)

# Create pipelines for scaling
pipeline_main = Pipeline(
    [
        Scaler(),
    ],
    name="main_pipeline",
)

pipeline_covariates = Pipeline(
    [
        Scaler(),
    ],
    name="covariates_pipeline",
)

# Fit and transform with explicit fit_transform calls
train_scaled = pipeline_main.fit_transform(train_list)
val_scaled = pipeline_main.transform(val_list)
test_scaled = pipeline_main.transform(test_list)

# Fit and transform the covariate series
train_covariates_scaled = pipeline_covariates.fit_transform(train_covariates_list)
val_covariates_scaled = pipeline_covariates.transform(val_covariates_list)
test_covariates_scaled = pipeline_covariates.transform(test_covariates_list)


# Check for NaNs in scaled data
def check_for_nans(series_list, name="series"):
    """Check for NaNs in a list of time series."""
    for i, series in enumerate(series_list):
        if series.pd_dataframe().isna().any().any():
            raise ValueError(f"NaNs found in {name} at index {i}")


# Check scaled data
check_for_nans(train_scaled, "train_scaled")
check_for_nans(val_scaled, "val_scaled")
check_for_nans(test_scaled, "test_scaled")


# Create time features using datetime_attribute_timeseries
def create_datetime_features(series):
    features = {}
    for attribute in ["month", "year", "day_of_week"]:
        features[attribute] = datetime_attribute_timeseries(series, attribute=attribute)
    return features


# Scale the data
scaler = Scaler()
train_scaled = scaler.fit_transform(train_list)
val_scaled = scaler.transform(val_list)
test_scaled = scaler.transform(test_list)

# Create datetime features for covariates
train_covariates = create_datetime_features(train_list)
val_covariates = create_datetime_features(val_list)
test_covariates = create_datetime_features(test_list)

# Create TFT model with quantile forecasts
model = TFTModel(
    input_chunk_length=24,
    output_chunk_length=7,
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=50,
    likelihood="quantile",
    optimizer_kwargs={"lr": 1e-3},
    pl_trainer_kwargs={
        "accelerator": "auto",
        "devices": 1,
        "gradient_clip_val": 0.1,
    },
    random_state=42,
)

# Train the model
model.fit(
    series=train_scaled,
    past_covariates=train_covariates,
    val_series=val_scaled,
    val_past_covariates=val_covariates,
    verbose=True,
)

# Make probabilistic predictions
predictions_95 = model.predict(
    n=7,
    series=val_scaled,
    past_covariates=val_covariates,
    num_samples=100,
    quantiles=[0.05, 0.5, 0.95],
)

# Backtesting
backtest = model.historical_forecasts(
    series=val_scaled,
    past_covariates=val_covariates,
    start=0.5,
    forecast_horizon=7,
    stride=1,
    retrain=False,
    verbose=True,
)


# Plotting functions
def plot_predictions(series, predictions, title):
    plt.figure(figsize=(10, 6))
    series.plot(label="actual")
    predictions.plot(label=["5%", "median", "95%"])
    plt.title(title)
    plt.legend()
    plt.show()


def plot_backtest(series, backtest_predictions, title):
    plt.figure(figsize=(10, 6))
    series.plot(label="actual")
    backtest_predictions.plot(label="backtest")
    plt.title(title)
    plt.legend()
    plt.show()


# Plot results
plot_predictions(
    val_list[0],
    scaler.inverse_transform(predictions_95[0]),
    "TFT Predictions with Confidence Intervals",
)

plot_backtest(
    val_list[0], scaler.inverse_transform(backtest[0]), "TFT Backtest Results"
)

# Print metrics
print("\nValidation Metrics:")
print(f"MAPE: {mape(val_list, scaler.inverse_transform(predictions_95)):.2f}%")
print(f"RMSE: {rmse(val_list, scaler.inverse_transform(predictions_95)):.2f}")
print(f"MAE: {mae(val_list, scaler.inverse_transform(predictions_95)):.2f}")

# Save the model
model.save("best_tft_model")
