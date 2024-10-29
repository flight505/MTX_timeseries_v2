import logging
import warnings
from datetime import datetime

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
from darts.models import TFTModel
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from darts.explainability import TFTExplainer
from darts.utils.likelihood_models import QuantileRegression
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("timeseries_processing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

logger.info("Loading data...")
# Load the data
df = pd.read_csv(
    "data/df_blood_preprocessed.csv",
)

# Convert date columns to datetime
logger.info("Converting date columns...")
df["ds"] = pd.to_datetime(df["ds"])
df["first_infusion_date"] = pd.to_datetime(df["first_infusion_date"])
df["next_infusion"] = pd.to_datetime(df["next_infusion"])

# Define data types for consistency
dtype_map = {
    "ds": "datetime64[ns]",
    "first_infusion_date": "datetime64[ns]",
    "next_infusion": "datetime64[ns]",
    "unique_id": str,
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

logger.info("Converting data types...")
df = df.astype(dtype_map)


def apply_minimum_days_cutoff(df, cycle_min_days):
    """
    Applies a cutoff to df based on the minimum_days_to_next_cycle for each cycle,
    excluding records where cycle_day is greater than the minimum_days_to_next_cycle for their respective cycle.

    Parameters:
    df (pd.DataFrame): The main DataFrame containing patient cycle information.
    cycle_min_days (pd.DataFrame): DataFrame containing the minimum_days_to_next_cycle for each cycle.

    Returns:
    pd.DataFrame: The modified df_blood DataFrame after applying the cutoff.
    """
    df = df.copy()
    # Ensure 'infno' is float for matching
    df["infno"] = df["infno"].astype(int)

    # Merge the minimum_days_to_next_cycle information into df
    df = df.merge(
        cycle_min_days[["infno", "minimum_days_to_next_cycle"]], on="infno", how="left"
    )

    # Apply the cutoff based on minimum_days_to_next_cycle for each cycle
    df = df[df["infno_day"] <= df["minimum_days_to_next_cycle"]]

    # Print the shape and unique patient count
    print(
        f"Shape after applying cutoff based on minimum_days_to_next_cycle: {df.shape}"
    )
    print(f"Unique patients after applying cutoff: {df['nopho_nr'].nunique()}")

    return df


# Example usage
cycle_min_days = pd.DataFrame(
    {
        "infno": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "minimum_days_to_next_cycle": [20.0, 20.0, 55.0, 41.0, 47.0, 49.0, 46.0, 47.0],
    }
)

# Assuming df_blood is already defined and loaded
df_blood = apply_minimum_days_cutoff(df, cycle_min_days)


df = df.drop(columns=["unit", "transer", "transth"])

TARGET_COLUMN = "Neutrophilocytes_B"


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
        self.beta_params = {}
        self.fill_residual_nan = fill_residual_nan
        self.exclude_columns = exclude_columns or []
        logger.info(
            f"Initialized TimeSeriesPreprocessor with target column: {target_column}"
        )

    def _calculate_delta_matrix(self, df, col):
        """Calculate delta matrix according to Equation 1"""
        logger.debug(f"Calculating delta matrix for column: {col}")
        df = df.sort_values(["unique_id", "normalized_time"])
        mask = (~df[col].isna()).astype(int)
        time_diff = df.groupby("unique_id")["normalized_time"].diff()

        delta = pd.Series(index=df.index, dtype=float)

        for uid in tqdm(
            df["unique_id"].unique(), desc=f"Processing delta matrix for {col}"
        ):
            mask_group = mask[df["unique_id"] == uid]
            time_diff_group = time_diff[df["unique_id"] == uid]
            delta_group = pd.Series(index=mask_group.index, dtype=float)
            delta_group.iloc[0] = 0

            for t in range(1, len(mask_group)):
                if mask_group.iloc[t - 1] == 0:
                    delta_group.iloc[t] = (
                        time_diff_group.iloc[t] + delta_group.iloc[t - 1]
                    )
                else:
                    delta_group.iloc[t] = time_diff_group.iloc[t]

            delta[delta_group.index] = delta_group

        return delta

    def _calculate_decay_parameter(self, delta, W_gamma=0.1, b_gamma=0.0):
        """Calculate temporal decay parameter gamma according to Equation 2"""
        logger.debug("Calculating decay parameter")
        return np.exp(-np.maximum(0, W_gamma * delta + b_gamma))

    def _initialize_missing_values(self, df, col, gamma):
        """Initialize missing values according to Equation 3"""
        logger.debug(f"Initializing missing values for column: {col}")
        mask = (~df[col].isna()).astype(int)
        col_mean = df[col].mean()
        last_observed = df.groupby("unique_id")[col].ffill()
        initialized_values = gamma * last_observed + (1 - gamma) * col_mean
        df[col] = df[col].fillna(initialized_values)
        return df

    def _calculate_beta_parameter(self, gamma, mask, W_beta=0.1, b_beta=0.0):
        """Calculate trade-off parameter beta according to Equation 4"""
        logger.debug("Calculating beta parameter")
        combined = np.column_stack([gamma, mask])
        return sigmoid(np.dot(combined, W_beta) + b_beta)

    def prepare_and_process(self, df):
        logger.info("Starting data preparation and processing...")
        df = df.copy()
        df = self._initial_preparation(df)

        if self.complete_timeline:
            logger.info("Completing timeline...")
            df = self._complete_timeline(df)

        feature_columns = [
            col
            for col in df.columns
            if col not in self.exclude_columns
            and not col.endswith(("_delta", "_missing"))
        ]

        logger.info("Processing features with temporal knowledge...")
        for col in tqdm(feature_columns, desc="Processing features"):
            if df[col].isna().any():
                delta = self._calculate_delta_matrix(df, col)
                gamma = self._calculate_decay_parameter(delta)
                df = self._initialize_missing_values(df, col, gamma)
                mask = (~df[col].isna()).astype(int)
                beta = self._calculate_beta_parameter(gamma, mask)
                self.decay_params[col] = {"gamma": gamma, "delta": delta}
                self.beta_params[col] = beta

        if self.encode_temporal_distance:
            logger.info("Encoding temporal distance...")
            df = self._encode_temporal_distance(df)

        if self.fill_residual_nan:
            logger.info("Filling residual NaN values...")
            df = self._fill_residual_nan(df)

        logger.info("Data preparation and processing completed")
        return df

    def _initial_preparation(self, X):
        logger.info("Performing initial data preparation...")
        X = self._aggregate_daily(X)
        X["ds_date"] = pd.to_datetime(X["ds_date"])
        X["normalized_time"] = X.groupby("unique_id")["ds_date"].transform(
            lambda x: (x - x.min()).dt.days
        )

        # Calculate infno_day here
        X["infno_first_time"] = X.groupby(["unique_id", "infno"])[
            "normalized_time"
        ].transform("min")
        X["infno_day"] = X["normalized_time"] - X["infno_first_time"]
        X.drop(columns=["infno_first_time"], inplace=True)

        X = X.pivot_table(
            index=[
                "unique_id",
                "normalized_time",
                "age_at_diagdate",
                "sex",
                "infno",
                "infno_day",
                "ds_date",
            ],
            columns="component",
            values="value",
            aggfunc="first",
        ).reset_index()

        if self.target_column in X.columns:
            X = X.rename(columns={self.target_column: "y"})
        else:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame."
            )

        return X

    def _aggregate_daily(self, X):
        logger.info("Aggregating daily data...")
        X["ds_date"] = X["ds"].dt.date
        grouped = X.groupby(
            ["unique_id", "ds_date", "age_at_diagdate", "sex", "infno", "component"]
        )
        daily_data = grouped.agg({"value": "mean"}).reset_index()
        return daily_data

    def _complete_timeline(self, df):
        logger.info("Completing timeline...")
        unique_ids = df["unique_id"].unique()
        complete_timeline = []

        for uid in tqdm(unique_ids, desc="Processing timelines"):
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

        dynamic_columns = ["infno", "infno_day"]
        for col in dynamic_columns:
            df[col] = df.groupby("unique_id")[col].ffill()

        return df

    def _encode_temporal_distance(self, df):
        logger.info("Encoding temporal distance...")
        feature_columns = [
            col
            for col in df.columns
            if col not in self.exclude_columns and not col.endswith("_delta")
        ]
        temporal_distance_data = {}

        for col in tqdm(feature_columns, desc="Processing temporal distances"):
            if not col.endswith("_delta"):
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

    def _fill_residual_nan(self, df):
        logger.info("Filling residual NaN values...")
        feature_columns = [col for col in df.columns if col not in ["unique_id", "ds"]]
        df[feature_columns] = df.groupby("unique_id")[feature_columns].ffill()
        df[feature_columns] = df.groupby("unique_id")[feature_columns].bfill()
        return df


# Initialize preprocessor
logger.info("Initializing preprocessor...")
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

# Process the data
logger.info("Processing data...")
processed_df = preprocessor.prepare_and_process(df)


# Function to split data into train, validation, and test sets
def split_data(unique_ids, test_size=0.2, val_size=0.1):
    logger.info("Splitting data into train, validation, and test sets...")
    np.random.seed(42)
    unique_ids = np.sort(unique_ids)
    n_samples = len(unique_ids)
    n_test = int(n_samples * test_size)
    n_val = int(n_samples * val_size)
    test_ids = unique_ids[-n_test:]
    val_ids = unique_ids[-(n_test + n_val) : -n_test]
    train_ids = unique_ids[: -(n_test + n_val)]
    return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()


# Function to create TimeSeries objects
def create_time_series(patient_data, value_cols, static_covariates):
    logger.debug(f"Creating time series for columns: {value_cols}")
    patient_data = patient_data.copy()
    patient_data["date"] = pd.to_datetime(patient_data["normalized_time"], unit="D")
    patient_data = patient_data.sort_values("date")

    ts = TimeSeries.from_dataframe(
        df=patient_data,
        time_col="date",
        value_cols=value_cols,
        static_covariates=static_covariates,
        fill_missing_dates=True,
        freq="D",
        fillna_value=None,
    )

    return ts


# Prepare data for model training
logger.info("Preparing data for model training...")
unique_ids = processed_df["unique_id"].unique()
train_ids, val_ids, test_ids = split_data(unique_ids)

# Initialize lists for storing time series
train_list, val_list, test_list = [], [], []
train_covariates_list, val_covariates_list, test_covariates_list = [], [], []

logger.info("Creating time series for each patient...")
for unique_id in tqdm(unique_ids, desc="Processing patients"):
    patient_data = processed_df[processed_df["unique_id"] == unique_id]
    static_covariates = patient_data[["sex", "age_at_diagdate"]].iloc[0]
    value_cols = ["y"]

    # Create main series
    patient_series = create_time_series(patient_data, value_cols, static_covariates)

    # Create past covariates series
    covariate_cols = ["infno", "infno_day"]
    covariate_series = create_time_series(patient_data, covariate_cols, None)

    # Append to appropriate list
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
logger.info("Converting series to float32...")


def convert_to_float32(ts_list):
    return [ts.astype(np.float32) for ts in ts_list]


train_list = convert_to_float32(train_list)
val_list = convert_to_float32(val_list)
test_list = convert_to_float32(test_list)
train_covariates_list = convert_to_float32(train_covariates_list)
val_covariates_list = convert_to_float32(val_covariates_list)
test_covariates_list = convert_to_float32(test_covariates_list)

# Create pipelines for scaling
logger.info("Creating scaling pipelines...")
pipeline_main = Pipeline([Scaler()])
pipeline_covariates = Pipeline([Scaler()])

# Scale the data
logger.info("Scaling the data...")
train_scaled = pipeline_main.fit_transform(train_list)
val_scaled = pipeline_main.transform(val_list)
test_scaled = pipeline_main.transform(test_list)

train_covariates_scaled = pipeline_covariates.fit_transform(train_covariates_list)
val_covariates_scaled = pipeline_covariates.transform(val_covariates_list)
test_covariates_scaled = pipeline_covariates.transform(test_covariates_list)


# Check for NaNs in scaled data
def check_for_nans(series_list, name="series"):
    logger.info(f"Checking for NaNs in {name}...")
    for i, series in enumerate(series_list):
        if series.pd_dataframe().isna().any().any():
            raise ValueError(f"NaNs found in {name} at index {i}")


# Check scaled data
check_for_nans(train_scaled, "train_scaled")
check_for_nans(val_scaled, "val_scaled")
check_for_nans(test_scaled, "test_scaled")


# Create datetime features
def create_datetime_features(series):
    logger.info("Creating datetime features...")
    features = {}
    for attribute in ["month", "year", "day_of_week"]:
        features[attribute] = datetime_attribute_timeseries(series, attribute=attribute)
    return features


# Create TFT model with quantile forecasts
logger.info("Creating TFT model...")
model = TFTModel(
    input_chunk_length=24,
    output_chunk_length=7,
    hidden_size=64,
    lstm_layers=1,
    num_attention_heads=4,
    dropout=0.1,
    batch_size=32,
    n_epochs=50,
    likelihood=QuantileRegression(quantiles=[0.1, 0.5, 0.9]),
    optimizer_kwargs={"lr": 1e-3},
    add_encoders={"cyclic": {"future": ["month", "day_of_week"]}},
    pl_trainer_kwargs={
        "accelerator": "auto",
        "devices": 1,
        "callbacks": [EarlyStopping(monitor="val_loss", patience=5, mode="min")],
        "gradient_clip_val": 0.1,
    },
    random_state=42,
)

# Train the model
logger.info("Training the model...")
model.fit(
    series=train_scaled,
    past_covariates=train_covariates_scaled,
    val_series=val_scaled,
    val_past_covariates=val_covariates_scaled,
    verbose=True,
)

# Make probabilistic predictions
logger.info("Making predictions...")
predictions_90 = model.predict(
    n=7,
    series=val_scaled,
    past_covariates=val_covariates_scaled,
    num_samples=100,
)

# Backtesting
logger.info("Performing backtesting...")
backtest = model.historical_forecasts(
    series=val_scaled,
    past_covariates=val_covariates_scaled,
    start=0.5,
    forecast_horizon=7,
    stride=1,
    retrain=False,
    verbose=True,
)

# Create TFT explainer
logger.info("Creating TFT explainer...")
explainer = TFTExplainer(model)
explainability_result = explainer.explain()


# Plotting functions
def plot_predictions(series, predictions, title):
    logger.info(f"Plotting predictions: {title}")
    plt.figure(figsize=(10, 6))
    series.plot(label="actual")
    predictions.plot(label=["10%", "median", "90%"])
    plt.axhline(y=0.5, color="r", linestyle="--", label="Critical threshold (0.5)")
    plt.title(title)
    plt.legend()
    plt.show()


def plot_backtest(series, backtest_predictions, title):
    logger.info(f"Plotting backtest: {title}")
    plt.figure(figsize=(10, 6))
    series.plot(label="actual")
    backtest_predictions.plot(label="backtest")
    plt.axhline(y=0.5, color="r", linestyle="--", label="Critical threshold (0.5)")
    plt.title(title)
    plt.legend()
    plt.show()


# Plot results
logger.info("Plotting results...")
plot_predictions(
    val_list[0],
    pipeline_main.inverse_transform(predictions_90[0]),
    "TFT Predictions with Confidence Intervals",
)

plot_backtest(
    val_list[0], pipeline_main.inverse_transform(backtest[0]), "TFT Backtest Results"
)

# Plot feature importance and attention
logger.info("Plotting feature importance and attention...")
explainer.plot_variable_selection(explainability_result)
explainer.plot_attention(explainability_result, plot_type="time")

# Print metrics
logger.info("Calculating and printing metrics...")
print("\nValidation Metrics:")
print(f"MAPE: {mape(val_list, pipeline_main.inverse_transform(predictions_90)):.2f}%")
print(f"RMSE: {rmse(val_list, pipeline_main.inverse_transform(predictions_90)):.2f}")
print(f"MAE: {mae(val_list, pipeline_main.inverse_transform(predictions_90)):.2f}")
print(f"R2: {r2_score(val_list, pipeline_main.inverse_transform(predictions_90)):.2f}")

# Save the model
logger.info("Saving the model...")
model.save("best_tft_model")

logger.info("Processing completed successfully")
