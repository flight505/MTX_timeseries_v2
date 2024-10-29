import logging
import math
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import (
    MissingValuesFiller,
    Scaler,
    StaticCovariatesTransformer,
)
from darts.explainability import TFTExplainer
from darts.metrics import mae, mape, r2_score, rmse
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression
from darts.utils.statistics import check_seasonality, plot_acf
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from torch.autograd import Variable
from torch.nn.parameter import Parameter
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
    print(f"Unique patients after applying cutoff: {df['unique_id'].nunique()}")

    return df


# Example usage
cycle_min_days = pd.DataFrame(
    {
        "infno": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "minimum_days_to_next_cycle": [20.0, 20.0, 55.0, 41.0, 47.0, 49.0, 46.0, 47.0],
    }
)

# Assuming df_blood is already defined and loaded
df = apply_minimum_days_cutoff(df, cycle_min_days)


df = df.drop(
    columns=["unit", "transer", "transth"]
)  # transfusions feature is did the patient recive transfusion within 21 days of infno, these might leed to dataleakaged


# print missing values
print(df.isnull().sum())

logger.info("Performing data preparation...")
# Pivot the data while preserving all temporal information
df_pivot = df.pivot_table(
    index=[
        "unique_id",
        "ds",  # Using original datetime column
        "age_at_diagdate",
        "sex",
        "infno",
    ],
    columns="component",
    values="value",
    aggfunc="first",  # Take first value if there are duplicates at exact same timestamp
).reset_index()

# Calculate normalized time in days with decimal points
df_pivot["normalized_time"] = df_pivot.groupby("unique_id")["ds"].transform(
    lambda x: (x - x.min()).dt.total_seconds() / (24 * 60 * 60)
)

# Calculate infno_day (keeping fractional days)
df_pivot["infno_first_time"] = df_pivot.groupby(["unique_id", "infno"])[
    "normalized_time"
].transform("min")
df_pivot["infno_day"] = df_pivot["normalized_time"] - df_pivot["infno_first_time"]
df_pivot.drop(columns=["infno_first_time"], inplace=True)

# First, explicitly define which columns we want to impute
# These are all columns except the metadata/identifier columns
COLUMNS_TO_IMPUTE = df_pivot.drop(
    columns=[
        "unique_id",
        "ds",
        "age_at_diagdate",
        "sex",
        "infno",
        "infno_day",
        "normalized_time",
    ]
).columns.tolist()

logger.info(f"Preparing to impute the following columns: {COLUMNS_TO_IMPUTE}")

# Prepare data for CATSI
values = df_pivot[COLUMNS_TO_IMPUTE].values
masks = ~np.isnan(values)
values[np.isnan(values)] = 0

# Create evaluation masks and values for training
eval_masks = masks.copy()
np.random.seed(42)
eval_masks = eval_masks & (
    np.random.random(eval_masks.shape) > 0.2
)  # Hold out 20% of observed values
evals = values.copy()

# Prepare data dictionary for CATSI
data = {
    "values": torch.tensor(values, dtype=torch.float32).unsqueeze(0),
    "masks": torch.tensor(masks.astype(np.float32), dtype=torch.float32).unsqueeze(0),
    "deltas": torch.zeros_like(torch.tensor(values, dtype=torch.float32)).unsqueeze(0),
    "lengths": torch.tensor([values.shape[0]], dtype=torch.float32),
    "min_vals": torch.tensor(np.nanmin(values, axis=0), dtype=torch.float32)
    .unsqueeze(0)
    .unsqueeze(0),
    "max_vals": torch.tensor(np.nanmax(values, axis=0), dtype=torch.float32)
    .unsqueeze(0)
    .unsqueeze(0),
    "evals": torch.tensor(evals, dtype=torch.float32).unsqueeze(0),
    "eval_masks": torch.tensor(
        eval_masks.astype(np.float32), dtype=torch.float32
    ).unsqueeze(0),
}

# save data as csv
df_pivot.to_csv("data/df_pivot.csv", index=False)

TARGET_COLUMN = "Neutrophilocytes_B"
logger.info(f"Creating target column from component: {TARGET_COLUMN}")


class MLPFeatureImputation(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super(MLPFeatureImputation, self).__init__()

        self.W = Parameter(torch.Tensor(input_size, hidden_size, input_size))
        self.b = Parameter(torch.Tensor(input_size, hidden_size))
        self.nonlinear_regression = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        m = torch.ones(input_size, hidden_size, input_size)
        stdv = 1.0 / math.sqrt(input_size)
        for i in range(input_size):
            m[i, :, i] = 0
        self.register_buffer("m", m)
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        hidden = torch.cat(
            tuple(
                F.linear(x, self.W[i] * Variable(self.m[i]), self.b[i]).unsqueeze(2)
                for i in range(len(self.W))
            ),
            dim=2,
        )
        z_h = self.nonlinear_regression(hidden)
        return z_h.squeeze(-1)


class InputTemporalDecay(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.eye(input_size, input_size)
        self.register_buffer("m", m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        return torch.exp(-gamma)


class RNNContext(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.GRUCell(input_size, hidden_size)

    def forward(self, input, seq_lengths):
        T_max = input.shape[1]  # batch x time x dims

        h = torch.zeros(input.shape[0], self.hidden_size).to(input.device)
        hn = torch.zeros(input.shape[0], self.hidden_size).to(input.device)

        for t in range(T_max):
            h = self.rnn_cell(input[:, t, :], h)
            padding_mask = (
                ((t + 1) <= seq_lengths).float().unsqueeze(1).to(input.device)
            )
            hn = padding_mask * h + (1 - padding_mask) * hn

        return hn


class CATSI(nn.Module):
    def __init__(self, num_vars, hidden_size=64, context_hidden=32):
        super().__init__()
        self.num_vars = num_vars
        self.hidden_size = hidden_size

        self.context_mlp = nn.Sequential(
            nn.Linear(3 * self.num_vars + 1, 2 * context_hidden),
            nn.ReLU(),
            nn.Linear(2 * context_hidden, context_hidden),
        )
        self.context_rnn = RNNContext(2 * self.num_vars, context_hidden)

        self.initial_hidden = nn.Linear(2 * context_hidden, 2 * hidden_size)
        self.initial_cell_state = nn.Tanh()

        self.rnn_cell_forward = nn.LSTMCell(
            2 * num_vars + 2 * context_hidden, hidden_size
        )
        self.rnn_cell_backward = nn.LSTMCell(
            2 * num_vars + 2 * context_hidden, hidden_size
        )

        self.decay_inputs = InputTemporalDecay(input_size=num_vars)

        self.recurrent_impute = nn.Linear(2 * hidden_size, num_vars)
        self.feature_impute = MLPFeatureImputation(num_vars)

        self.fuse_imputations = nn.Linear(2 * num_vars, num_vars)

    def forward(self, data):
        seq_lengths = data["lengths"]

        values = data["values"]  # pts x time_stamps x vars
        masks = data["masks"]
        deltas = data["deltas"]

        # compute context vector, h0 and c0
        T_max = values.shape[1]
        padding_masks = torch.cat(
            tuple(
                ((t + 1) <= seq_lengths).float().unsqueeze(1).to(values.device)
                for t in range(T_max)
            ),
            dim=1,
        )
        padding_masks = padding_masks.unsqueeze(2).repeat(
            1, 1, values.shape[2]
        )  # pts x time_stamps x vars

        data_means = values.sum(dim=1) / masks.sum(dim=1)  # pts x vars
        data_variance = ((values - data_means.unsqueeze(1)) ** 2).sum(dim=1) / (
            masks.sum(dim=1) - 1
        )
        data_stdev = data_variance**0.5
        data_missing_rate = 1 - masks.sum(dim=1) / padding_masks.sum(dim=1)
        data_stats = torch.cat(
            (
                seq_lengths.unsqueeze(1).float(),
                data_means,
                data_stdev,
                data_missing_rate,
            ),
            dim=1,
        )

        # normalization
        min_max_norm = data["max_vals"] - data["min_vals"]
        normalized_values = (values - data["min_vals"]) / min_max_norm
        normalized_means = (
            data_means - data["min_vals"].squeeze(1)
        ) / min_max_norm.squeeze(1)

        if self.training:
            normalized_evals = (data["evals"] - data["min_vals"]) / min_max_norm

        x_prime = torch.zeros_like(normalized_values)
        x_prime[:, 0, :] = normalized_values[:, 0, :]
        for t in range(1, T_max):
            x_prime[:, t, :] = normalized_values[:, t - 1, :]

        gamma = self.decay_inputs(deltas)
        x_decay = gamma * x_prime + (1 - gamma) * normalized_means.unsqueeze(1)
        x_complement = (
            masks * normalized_values + (1 - masks) * x_decay
        ) * padding_masks

        context_mlp = self.context_mlp(data_stats)
        context_rnn = self.context_rnn(
            torch.cat((x_complement, deltas), dim=-1), seq_lengths
        )
        context_vec = torch.cat((context_mlp, context_rnn), dim=1)
        h = self.initial_hidden(context_vec)
        c = self.initial_cell_state(h)

        inputs = torch.cat(
            [x_complement, masks, context_vec.unsqueeze(1).repeat(1, T_max, 1)], dim=-1
        )

        h_forward, c_forward = h[:, : self.hidden_size], c[:, : self.hidden_size]
        h_backward, c_backward = h[:, self.hidden_size :], c[:, self.hidden_size :]
        hiddens_forward = h[:, : self.hidden_size].unsqueeze(1)
        hiddens_backward = h[:, self.hidden_size :].unsqueeze(1)
        for t in range(T_max - 1):
            h_forward, c_forward = self.rnn_cell_forward(
                inputs[:, t, :], (h_forward, c_forward)
            )
            h_backward, c_backward = self.rnn_cell_backward(
                inputs[:, T_max - 1 - t, :], (h_backward, c_backward)
            )
            hiddens_forward = torch.cat(
                (hiddens_forward, h_forward.unsqueeze(1)), dim=1
            )
            hiddens_backward = torch.cat(
                (h_backward.unsqueeze(1), hiddens_backward), dim=1
            )

        rnn_imp = self.recurrent_impute(
            torch.cat((hiddens_forward, hiddens_backward), dim=2)
        )
        feat_imp = self.feature_impute(x_complement).squeeze(-1)

        # imputation fusion
        beta = torch.sigmoid(self.fuse_imputations(torch.cat((gamma, masks), dim=-1)))
        imp_fusion = beta * feat_imp + (1 - beta) * rnn_imp
        final_imp = masks * normalized_values + (1 - masks) * imp_fusion

        rnn_loss = F.mse_loss(
            rnn_imp * masks, normalized_values * masks, reduction="sum"
        )
        feat_loss = F.mse_loss(
            feat_imp * masks, normalized_values * masks, reduction="sum"
        )
        fusion_loss = F.mse_loss(
            imp_fusion * masks, normalized_values * masks, reduction="sum"
        )
        total_loss = rnn_loss + feat_loss + fusion_loss

        if self.training:
            rnn_loss_eval = F.mse_loss(
                rnn_imp * data["eval_masks"],
                normalized_evals * data["eval_masks"],
                reduction="sum",
            )
            feat_loss_eval = F.mse_loss(
                feat_imp * data["eval_masks"],
                normalized_evals * data["eval_masks"],
                reduction="sum",
            )
            fusion_loss_eval = F.mse_loss(
                imp_fusion * data["eval_masks"],
                normalized_evals * data["eval_masks"],
                reduction="sum",
            )
            total_loss_eval = rnn_loss_eval + feat_loss_eval + fusion_loss_eval

        def rescale(x):
            return torch.where(
                padding_masks == 1, x * min_max_norm + data["min_vals"], padding_masks
            )

        feat_imp = rescale(feat_imp)
        rnn_imp = rescale(rnn_imp)
        final_imp = rescale(final_imp)

        out_dict = {
            "loss": total_loss / masks.sum(),
            "verbose_loss": [
                ("rnn_loss", rnn_loss / masks.sum(), masks.sum()),
                ("feat_loss", feat_loss / masks.sum(), masks.sum()),
                ("fusion_loss", fusion_loss / masks.sum(), masks.sum()),
            ],
            "loss_count": masks.sum(),
            "imputations": final_imp,
            "feat_imp": feat_imp,
            "hist_imp": rnn_imp,
        }
        if self.training:
            out_dict["loss_eval"] = total_loss_eval / data["eval_masks"].sum()
            out_dict["loss_eval_count"] = data["eval_masks"].sum()
            out_dict["verbose_loss"] += [
                (
                    "rnn_loss_eval",
                    rnn_loss_eval / data["eval_masks"].sum(),
                    data["eval_masks"].sum(),
                ),
                (
                    "feat_loss_eval",
                    feat_loss_eval / data["eval_masks"].sum(),
                    data["eval_masks"].sum(),
                ),
                (
                    "fusion_loss_eval",
                    fusion_loss_eval / data["eval_masks"].sum(),
                    data["eval_masks"].sum(),
                ),
            ]

        return out_dict


def train_and_save_catsi(
    data, values, masks, df_pivot, save_path="data/catsi_output.npz"
):
    """Train CATSI model and save its output"""
    # Calculate num_vars from the shape of values
    num_vars = values.shape[1]

    logger.info("Initializing CATSI model...")
    catsi_model = CATSI(num_vars=num_vars, hidden_size=64, context_hidden=32)
    catsi_model = catsi_model.float()
    optimizer = torch.optim.Adam(catsi_model.parameters(), lr=0.001)

    num_epochs = 10
    logger.info(f"Starting CATSI training for {num_epochs} epochs...")
    catsi_model.train()

    best_loss = float("inf")
    patience = 3
    patience_counter = 0

    progress_bar = tqdm(range(num_epochs), desc="Training CATSI")
    for epoch in progress_bar:
        optimizer.zero_grad()
        output = catsi_model(data)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "best_loss": f"{best_loss:.4f}"}
        )
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Get imputed values and create df_pivot_imputed
    catsi_model.eval()
    with torch.no_grad():
        output = catsi_model(data)
        imputed_values = output["imputations"].squeeze(0).numpy()

    # Save the imputed values and related data
    np.savez(
        save_path,
        imputed_values=imputed_values,
        masks=masks,
        original_values=values,
        min_vals=data["min_vals"].squeeze(0).numpy(),
        max_vals=data["max_vals"].squeeze(0).numpy(),
    )

    # Create df_pivot_imputed with the imputed values
    feature_cols = df_pivot.columns.drop(
        [
            "unique_id",
            "ds",
            "age_at_diagdate",
            "sex",
            "infno",
            "infno_day",
            "normalized_time",
        ]
    )
    df_pivot_imputed = df_pivot.copy()
    df_pivot_imputed[feature_cols] = imputed_values

    return df_pivot_imputed


def load_or_train_catsi(
    data, values, masks, df_pivot, save_path="data/catsi_output.npz"
):
    """Load CATSI output if exists, otherwise train and save"""
    try:
        logger.info("Attempting to load saved CATSI output...")
        loaded = np.load(save_path)
        imputed_values = loaded["imputed_values"]

        # Create df_pivot_imputed with the loaded imputed values
        feature_cols = df_pivot.columns.drop(
            [
                "unique_id",
                "ds",
                "age_at_diagdate",
                "sex",
                "infno",
                "infno_day",
                "normalized_time",
            ]
        )
        df_pivot_imputed = df_pivot.copy()
        df_pivot_imputed[feature_cols] = imputed_values

        logger.info("Successfully loaded CATSI output")
        return df_pivot_imputed
    except (FileNotFoundError, IOError):
        logger.info("No saved CATSI output found. Training new model...")
        return train_and_save_catsi(data, values, masks, df_pivot, save_path)


# Use the functions
logger.info("Processing CATSI imputation...")
df_pivot_imputed = load_or_train_catsi(data, values, masks, df_pivot)

# Calculate imputation statistics for each column
for col in COLUMNS_TO_IMPUTE:
    missing = df_pivot[col].isna().sum()
    total = len(df_pivot[col])
    logger.info(f"Column {col}: {missing} missing values ({missing/total*100:.2f}%)")

# Now we can proceed with the time series modeling
logger.info("Preparing data for time series modeling...")

# Convert the wide format back to long format for time series processing
df_long = df_pivot_imputed.melt(
    id_vars=[
        "unique_id",
        "ds",
        "age_at_diagdate",
        "sex",
        "infno",
        "infno_day",
        "normalized_time",
    ],
    var_name="component",
    value_name="value",
)

# Sort the data
df_long = df_long.sort_values(["unique_id", "ds"])

# Create wide format with target column
logger.info(f"Creating wide format with target column: {TARGET_COLUMN}")
df_wide = df_long.pivot_table(
    index=[
        "unique_id",
        "ds",
        "age_at_diagdate",
        "sex",
        "infno",
        "infno_day",
        "normalized_time",
    ],
    columns="component",
    values="value",
).reset_index()

# Rename the target column to 'y' for consistency with the time series creation
df_wide["y"] = df_wide[TARGET_COLUMN]


def split_data(df_wide, test_size=0.2, val_size=0.1):
    """
    Split data maintaining temporal order and patient cohorts.
    Latest data points go to test set, earlier ones to validation, and earliest to train.
    """
    logger.info(
        "Splitting data into train, validation, and test sets (temporal order)..."
    )

    # Get the last timestamp for each patient
    patient_last_dates = df_wide.groupby("unique_id")["ds"].max().reset_index()
    patient_last_dates = patient_last_dates.sort_values("ds")

    # Calculate split points based on last timestamps
    n_patients = len(patient_last_dates)
    n_test = int(n_patients * test_size)
    n_val = int(n_patients * val_size)

    # Split patients based on their last timestamp
    test_ids = patient_last_dates.tail(n_test)["unique_id"].tolist()
    val_ids = patient_last_dates.iloc[-(n_test + n_val) : -n_test]["unique_id"].tolist()
    train_ids = patient_last_dates.iloc[: -(n_test + n_val)]["unique_id"].tolist()

    # Log split information
    logger.info(
        f"Train set: {len(train_ids)} patients ({100 * len(train_ids)/n_patients:.1f}%)"
    )
    logger.info(
        f"Validation set: {len(val_ids)} patients ({100 * len(val_ids)/n_patients:.1f}%)"
    )
    logger.info(
        f"Test set: {len(test_ids)} patients ({100 * len(test_ids)/n_patients:.1f}%)"
    )

    # Log temporal ranges
    train_dates = df_wide[df_wide["unique_id"].isin(train_ids)]["ds"]
    val_dates = df_wide[df_wide["unique_id"].isin(val_ids)]["ds"]
    test_dates = df_wide[df_wide["unique_id"].isin(test_ids)]["ds"]

    logger.info(f"Train period: {train_dates.min()} to {train_dates.max()}")
    logger.info(f"Validation period: {val_dates.min()} to {val_dates.max()}")
    logger.info(f"Test period: {test_dates.min()} to {test_dates.max()}")

    return train_ids, val_ids, test_ids


# Get unique IDs first
unique_ids = df_wide["unique_id"].unique()
logger.info(f"Total number of unique patients: {len(unique_ids)}")

# Split the data temporally
train_ids, val_ids, test_ids = split_data(df_wide, test_size=0.2, val_size=0.1)

# Initialize lists to store time series
train_list = []
val_list = []
test_list = []
train_covariates_list = []
val_covariates_list = []
test_covariates_list = []


# Function to create TimeSeries objects
def create_time_series(patient_data, value_cols, static_covariates):
    logger.debug(f"Creating time series for columns: {value_cols}")
    patient_data = patient_data.copy()
    # Issue: We're creating a new date from normalized_time, but we already have 'ds'
    # patient_data["date"] = pd.to_datetime(patient_data["normalized_time"], unit="D")
    patient_data = patient_data.sort_values("ds")  # Use original datetime

    ts = TimeSeries.from_dataframe(
        df=patient_data,
        time_col="ds",  # Use original datetime column
        value_cols=value_cols,
        static_covariates=static_covariates,
        fill_missing_dates=True,
        freq="D",
        fillna_value=None,
    )
    return ts


# Prepare data for model training
logger.info("Preparing data for model training...")
unique_ids = df_long["unique_id"].unique()
train_ids, val_ids, test_ids = split_data(unique_ids)

# Initialize lists for storing time series
train_list, val_list, test_list = [], [], []
train_covariates_list, val_covariates_list, test_covariates_list = [], [], []

logger.info("Creating time series for each patient...")
for unique_id in tqdm(unique_ids, desc="Processing patients"):
    # Get patient data from df_wide instead of df_long since we already have the target
    patient_data = df_wide[df_wide["unique_id"] == unique_id].copy()
    static_covariates = pd.DataFrame(
        {
            "sex": [patient_data["sex"].iloc[0]],
            "age_at_diagdate": [patient_data["age_at_diagdate"].iloc[0]],
        }
    )

    # Create main series with target variable
    value_cols = ["y"]
    patient_series = create_time_series(patient_data, value_cols, static_covariates)

    # Create covariates series
    covariate_cols = ["infno", "infno_day"]
    covariate_series = create_time_series(patient_data, covariate_cols, None)

    # Scale the series before adding to lists
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
