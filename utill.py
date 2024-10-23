import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# class names should be in CamelCase
# function names should be in snake_case
# variable names should be in snake_case
# constants should be in CAPITAL_SNAKE_CASE

# class names in the file
# 1. DataFrameValidator
# 2. MetricsBloodCoveragePlotter
# 3. BloodCoveragePlotter
# 4. InfusionMapper
# 5. CycleFinder


class DataFrameValidator:
    def __init__(self, df: pd.DataFrame) -> None:
        """
        A class used to validate the columns of a DataFrame.

        This class takes a DataFrame and a dictionary of expected values for each column.
        It provides a method to validate the DataFrame, checking that each column contains only the expected values.

        Attributes:
        df (DataFrame): The DataFrame to validate.
        expected_values (dict): A dictionary where the keys are column names and the values are lists of expected values for that column.

        Methods:
        validate_column(column_name, expected_vals): Validates a single column of the DataFrame.
        validate(): Validates all columns of the DataFrame.
        """

        self.df = df
        self.expected_values = {
            "saeabdominal": [np.nan, 1, 2, 3],
            "saeanaphylaxis": [np.nan, 1, 2, 3],
            "saeavasosteo": [np.nan, 1, 2, 3],
            "saecnsbleeding": [np.nan, 1, 2, 3],
            "saecomma": [np.nan, 1, 2, 3],
            "saedialysis": [np.nan, 1, 2, 3],
            "saeheartfail": [np.nan, 1, 2, 3],
            "saehyperglycemia": [np.nan, 1, 2, 3],
            "saehypertension": [np.nan, 1, 2, 3],
            "saeintcare": [np.nan, 1, 2, 3],
            "saeliverdysf": [np.nan, 1, 2, 3],
            "saeother": [np.nan, 1, 3, 5],
            "saepancretitis": [np.nan, 1, 2, 3],
            "saeparalysis": [np.nan, 1, 2, 3],
            "saepres": [np.nan, 1, 2, 3],
            "saeseizures": [np.nan, 1, 2, 3],
            "saesepticshock": [np.nan, 1, 2, 3],
            "saethrombosis": [np.nan, 1, 2, 3],
            "saevod": [np.nan, 1, 2, 3],
            "toxabdominal": [np.nan, 1, 2, 3],
            "toxanaphylaxis": [np.nan, 1, 2],
            "toxavasosteo": [np.nan, 1, 2, 3],
            "toxcnsbleeding": [np.nan, 1, 2],
            "toxcomma": [np.nan, 1],
            "toxdialysis": [np.nan, 1, 2, 4],
            "toxevent": [np.nan, 2, 3, 4, 5, 6, 7],
            "toxeventdate": "Datetime",  # Special case, handled differently
            "toxfungalinf": [np.nan, 1, 2, 3, 4, 5],
            "toxheartfail": [np.nan, 1],
            "toxhyperlipidemia": [np.nan, 1, 2, 3, 4],
            "toxhypertension": [np.nan, 1],
            "toxintcare": [np.nan, 1, 2, 3],
            "toxliverdysf": [np.nan, 1, 2, 3],
            "toxnone": [
                np.nan,
                list(range(1, 49)),
            ],  # Assuming 1 to 48 are valid values
            "toxother": [1, 2, 3, 4, 5, 6, 7],
            "toxpancretitis": [np.nan, 1, 2, 3, 5],
            "toxparalysis": [np.nan, 1, 2, 3, 4, 5, 6, 7],
            "toxpneumoinf": [np.nan, 1, 2],
            "toxpres": [np.nan, 1, 2, 3],
            "toxseizures": [np.nan, 1, 2, 3],
            "toxsusar": [np.nan, 1, 2, 3],
            "toxthrombosis": [np.nan, 1, 2, 3],
            "toxvod": [np.nan, 1, 2],
        }

    def validate_column(self, column_name, expected_vals) -> bool:
        if column_name == "toxeventdate":
            return self.df[
                column_name
            ].isnull().all() or pd.api.types.is_datetime64_any_dtype(
                self.df[column_name]
            )
        elif column_name == "toxnone":
            column_values = self.df[column_name].dropna().unique()
            # Flatten the expected_vals list
            expected_vals = [np.nan] + list(range(1, 49))
            return all(value in expected_vals for value in column_values)
        else:
            return self.df[column_name].dropna().isin(expected_vals).all()

    def validate(self):
        all_columns_passed = True
        for column, values in self.expected_values.items():
            if not self.validate_column(column, values):
                print(f"Column {column} has invalid data.")
                all_columns_passed = False
        if all_columns_passed:
            print("All SAE and TOX in All2008 columns passed.")


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MetricsBloodCoveragePlotter:
    def __init__(self, df_blood: pd.DataFrame) -> None:
        self.df_blood = df_blood

    def get_nopho_infno_pairs(self) -> pd.DataFrame:
        unique_pairs = self.df_blood.drop_duplicates(subset=["nopho_nr", "infno"])[
            [
                "nopho_nr",
                "infno",
                "next_infusion",
                "most_recent_infusion_date",
                "first_infusion_date",
            ]
        ]
        return unique_pairs

    def plot_blood_coverage(self):
        unique_pairs = self.get_nopho_infno_pairs()

        # Reconstructing blood_coverage_data
        blood_coverage_data = []

        # Adjust the handling for infno = 0
        for infno in range(0, 10):
            blood_subset = unique_pairs[unique_pairs["infno"] == infno]

            for idx, row in blood_subset.iterrows():
                patient_id = row["nopho_nr"]
                if row["infno"] == 0:
                    # Use the first_infusion_date to look back 30 days as the start date for pre-treatment samples
                    # Assuming first_infusion_date is available and represents the date of the first infusion
                    start_date = row["first_infusion_date"] - pd.DateOffset(days=30)
                    end_date = row[
                        "first_infusion_date"
                    ]  # The day before the first infusion

                else:
                    # For infno > 0, use most_recent_infusion_date as the start date
                    # Ensure that most_recent_infusion_date is not NaT, otherwise, skip this row
                    if pd.isnull(row["most_recent_infusion_date"]):
                        continue
                    start_date = row["most_recent_infusion_date"]
                    end_date = start_date + pd.DateOffset(days=20)

                # Filter blood data for this patient and timeframe
                patient_blood_data = self.df_blood[
                    (self.df_blood["nopho_nr"] == patient_id)
                    & (self.df_blood["sample_time"] >= start_date)
                    & (
                        self.df_blood["sample_time"] < end_date
                    )  # Use < end_date to exclude the infusion day itself
                ]

                unique_tests = patient_blood_data["component"].nunique()
                total_samples = len(patient_blood_data)

                blood_coverage_data.append(
                    {
                        "nopho_nr": patient_id,
                        "infno": infno,
                        "unique_tests_count": unique_tests,
                        "total_samples": total_samples,
                    }
                )

        blood_coverage_data = pd.DataFrame(blood_coverage_data)

        # Creating the heatmap data
        heatmap_data_unique = blood_coverage_data.pivot(
            index="nopho_nr", columns="infno", values="unique_tests_count"
        )
        heatmap_data_total = blood_coverage_data.pivot(
            index="nopho_nr", columns="infno", values="total_samples"
        )

        # Ensure both dataframes have the same shape and fill NaN values
        heatmap_data_unique = heatmap_data_unique.reindex_like(
            heatmap_data_total
        ).fillna(0)
        heatmap_data_total = heatmap_data_total.fillna(0)

        # Combine unique tests count and total samples for annotations
        combined_annotations = (
            heatmap_data_unique.astype(str) + "/" + heatmap_data_total.astype(str)
        )

        # Plotting
        plt.figure(figsize=(12, 100))  # Adjust the figure size as needed
        sns.heatmap(
            heatmap_data_unique,  # Use for color scale
            cmap="Blues",
            cbar=False,
            annot=combined_annotations,
            fmt="",
            annot_kws={"fontsize": 8},  # Adjust text size as needed
            linewidths=0.5,
            cbar_kws={"label": "Unique Tests Count"},
        )
        plt.title(
            "Number of Unique Blood Tests/Total Blood Samples for Each Patient during Each Infno"
        )
        plt.ylabel("Patient ID (nopho_nr)")
        plt.xlabel("Infno")  # Adjust x-axis label as needed
        plt.tight_layout()  # Adjust subplot parameters to give some padding
        plt.show()

    # plot_blood_coverage(df_mtx, df_blood)


class InfusionMapper:
    def __init__(self, df_mtx: pd.DataFrame, df_samples: pd.DataFrame) -> None:
        self.df_mtx = df_mtx.copy()
        self.df_samples = df_samples.copy()

        # Ensure datetime columns are converted to date for consistent comparison
        self.df_samples["sample_date"] = self.df_samples["sample_time"].dt.date
        self.df_mtx["inf_date"] = self.df_mtx["mtx_inf_datetime"].dt.date

    def map_infno_to_samples(self) -> pd.DataFrame:
        self.df_samples = self.df_samples.sort_values(by=["nopho_nr", "sample_time"])
        self.df_mtx = self.df_mtx.sort_values(by=["nopho_nr", "mtx_inf_datetime"])

        self._precalculate_infusion_mappings()
        self._map_infusion_details_to_samples()
        self._calculate_most_recent_infusion()

        return self.df_samples

    def _precalculate_infusion_mappings(self):
        self.df_mtx["next_infusion"] = self.df_mtx.groupby("nopho_nr")[
            "inf_date"
        ].shift(-1)
        self.df_samples["first_infusion_date"] = self.df_samples["nopho_nr"].map(
            self.df_mtx.groupby("nopho_nr")["inf_date"].min().to_dict()
        )

    def _map_infusion_details_to_samples(self):
        for index, sample in self.df_samples.iterrows():
            patient_infusions = self.df_mtx[
                self.df_mtx["nopho_nr"] == sample["nopho_nr"]
            ]
            last_infusion = patient_infusions[
                patient_infusions["inf_date"] <= sample["sample_date"]
            ].tail(1)

            if not last_infusion.empty:
                self.df_samples.at[index, "infno"] = last_infusion["infno"].iloc[0]
                self.df_samples.at[index, "days_since_first_infusion"] = (
                    sample["sample_date"]
                    - pd.to_datetime(sample["first_infusion_date"]).date()
                ).days

                next_infusion = patient_infusions[
                    patient_infusions["inf_date"] > sample["sample_date"]
                ].head(1)

                if not next_infusion.empty:
                    self.df_samples.at[index, "next_infusion"] = next_infusion[
                        "inf_date"
                    ].iloc[0]
                    self.df_samples.at[index, "days_to_next_infusion"] = (
                        next_infusion["inf_date"].iloc[0] - sample["sample_date"]
                    ).days
                else:
                    self.df_samples.at[index, "next_infusion"] = pd.NaT
                    self.df_samples.at[index, "days_to_next_infusion"] = pd.NA
            else:
                self.df_samples.at[index, "infno"] = 0
                self.df_samples.at[index, "next_infusion"] = pd.to_datetime(
                    sample["first_infusion_date"]
                )
                self.df_samples.at[index, "days_to_next_infusion"] = (
                    pd.to_datetime(sample["first_infusion_date"]).date()
                    - sample["sample_date"]
                ).days

                # Now compare this numeric value with 0 directly (not with pd.Timedelta)
                if self.df_samples.at[index, "days_to_next_infusion"] < 0:
                    self.df_samples.at[index, "days_to_next_infusion"] = abs(
                        self.df_samples.at[index, "days_to_next_infusion"]
                    )

    def _calculate_most_recent_infusion(self):
        for index, sample in self.df_samples.iterrows():
            patient_infusions = self.df_mtx[
                (self.df_mtx["nopho_nr"] == sample["nopho_nr"])
                & (self.df_mtx["inf_date"] <= sample["sample_date"])
            ]

            if not patient_infusions.empty:
                most_recent_infusion_date = patient_infusions["inf_date"].max()
                self.df_samples.at[index, "most_recent_infusion_date"] = (
                    most_recent_infusion_date
                )

                days_since_last_inf = (
                    sample["sample_date"] - most_recent_infusion_date
                ).days
                self.df_samples.at[index, "days_since_last_inf"] = days_since_last_inf


# Note: Ensure 'sample_time' and 'mtx_inf_datetime' in the original df_samples and df_mtx DataFrames are in datetime format for the above conversions to work correctly.


class CycleFinder:
    def __init__(self, df, component=None, threshold=None, mode="drop_below"):
        self.df = df
        self.component = component
        self.threshold = threshold
        self.mode = mode

    @staticmethod
    def interpolate_crossing_time(ts1, val1, ts2, val2, threshold):
        if val1 == val2:
            return ts1 if val1 <= threshold else ts2

        fraction = (threshold - val1) / (val2 - val1)
        crossing_time = ts1 + (ts2 - ts1) * fraction
        return crossing_time

    @staticmethod
    def find_cycles_with_interpolation(
        series, infno_series, threshold=None, mode="drop_below"
    ):
        cycles = []
        in_cycle = False
        start_time, end_time, min_time, max_time = None, None, None, None
        min_val, max_val = float("inf"), float("-inf")
        start_infno = None  # Initialize start_infno here

        prev_time, prev_val = None, None

        # If the series has only one data point, skip processing
        if len(series) <= 1:
            return cycles

        # Initializing edge case handling based on mode
        if mode == "drop_below" and series.iloc[0] < threshold:
            start_time = series.index[0]
            in_cycle = True
        elif mode == "rise_above" and series.iloc[0] > threshold:
            start_time = series.index[0]
            in_cycle = True

        for idx, (time, value) in enumerate(series.items()):
            infno = infno_series.iloc[idx]
            # Update next_time and next_val for the current iteration
            if idx + 1 < len(series):
                next_time, next_val = series.index[idx + 1], series.iloc[idx + 1]
            else:
                next_time, next_val = None, None

            # Handling values exactly on the threshold
            if value == threshold:
                if prev_val is not None and next_val is not None:
                    # Determine the direction of crossing based on previous and next values
                    if mode == "drop_below":
                        if prev_val > threshold and next_val < threshold:
                            value = (
                                threshold - 0.0001
                            )  # Adjust value slightly below threshold
                        elif prev_val < threshold and next_val > threshold:
                            value = (
                                threshold + 0.0001
                            )  # Adjust value slightly above threshold
                    elif mode == "rise_above":
                        if prev_val < threshold and next_val > threshold:
                            value = (
                                threshold + 0.0001
                            )  # Adjust value slightly above threshold
                        elif prev_val > threshold and next_val < threshold:
                            value = (
                                threshold - 0.0001
                            )  # Adjust value slightly below threshold

            if prev_time is not None:
                if mode == "drop_below":
                    crossing_below = prev_val > threshold and value < threshold
                    crossing_above = prev_val < threshold and value > threshold
                else:  # mode == "rise_above"
                    crossing_below = prev_val < threshold and value > threshold
                    crossing_above = prev_val > threshold and value < threshold

                if crossing_below:
                    interpolated_time = CycleFinder.interpolate_crossing_time(
                        prev_time, prev_val, time, value, threshold
                    )
                    if not in_cycle:
                        start_time = interpolated_time
                        in_cycle = True
                        min_val = value
                        min_time = time
                        start_infno = infno

                elif crossing_above:
                    interpolated_time = CycleFinder.interpolate_crossing_time(
                        prev_time, prev_val, time, value, threshold
                    )
                    if in_cycle:
                        end_time = interpolated_time
                        if mode == "drop_below":
                            extreme_val, extreme_time = min_val, min_time
                        else:  # mode == "rise_above"
                            extreme_val, extreme_time = max_val, max_time

                        cycles.append(
                            {
                                "t_start": start_time,
                                "t_end": end_time,
                                "t_extreme": extreme_time,
                                "duration": (end_time - start_time).total_seconds()
                                / 3600,  # in hours
                                "doc": extreme_val - threshold,
                                "start_infno": start_infno,
                                "end_infno": infno,
                            }
                        )
                        in_cycle = False
                        start_time, end_time, min_time, max_time = (
                            None,
                            None,
                            None,
                            None,
                        )
                        min_val, max_val = float("inf"), float("-inf")

            if in_cycle:
                if mode == "drop_below" and value < min_val:
                    min_val = value
                    min_time = time
                elif mode == "rise_above" and value > max_val:
                    max_val = value
                    max_time = time

            prev_time, prev_val = time, value

        # Handle edge case where series ends in a cycle
        if in_cycle:
            end_time = series.index[-1]
            if mode == "drop_below":
                extreme_val, extreme_time = min_val, min_time
            else:  # mode == "rise_above"
                extreme_val, extreme_time = max_val, max_time

            # Handle edge case where series ends in a cycle
            if in_cycle:
                end_time = series.index[-1]
                if mode == "drop_below":
                    extreme_val, extreme_time = min_val, min_time
                else:  # mode == "rise_above"
                    extreme_val, extreme_time = max_val, max_time

                cycles.append(
                    {
                        "t_start": start_time,
                        "t_end": end_time,
                        "t_extreme": extreme_time,
                        "duration": (end_time - start_time).total_seconds()
                        / 3600,  # in hours
                        "doc": extreme_val - threshold,
                        "start_infno": start_infno,
                        "end_infno": infno,
                    }
                )

        return cycles

    def identify_cycles(self) -> pd.DataFrame:
        df_component = self.df[self.df["component"] == self.component]

        # Ensure 'diagdate' is of datetime type in self.df
        self.df["diagdate"] = pd.to_datetime(self.df["diagdate"])

        all_cycles_interpolated = []
        for nopho_nr, group in df_component.groupby("nopho_nr"):
            cycles = self.find_cycles_with_interpolation(
                group.set_index("sample_time")["reply_num"],
                group.set_index("sample_time")["infno"],  # Pass the infno series
                threshold=self.threshold,
                mode=self.mode,
            )
            for cycle in cycles:
                cycle["nopho_nr"] = nopho_nr
            all_cycles_interpolated.extend(cycles)

        df_cycles = pd.DataFrame(all_cycles_interpolated)

        # Check if df_cycles is empty
        if df_cycles.empty:
            print(f"No cycles found for component {self.component}.")
            # remove the component from the dictionary of dataframes
            return df_cycles  # or raise an exception

        # Check if 'nopho_nr' exists in df_cycles and self.df
        if "nopho_nr" not in df_cycles.columns:
            raise KeyError("'nopho_nr' not found in df_cycles")
        if "nopho_nr" not in self.df.columns:
            raise KeyError("'nopho_nr' not found in self.df")

        # Create a dictionary from df with nopho_nr as keys and diagdate as values
        if not hasattr(self, "diagdate_dict"):
            self.diagdate_dict = (
                self.df.drop_duplicates(subset="nopho_nr")
                .set_index("nopho_nr")["diagdate"]
                .to_dict()
            )

        # Map diagdate to each cycle using the diagdate_dict
        df_cycles["diagdate"] = df_cycles["nopho_nr"].map(self.diagdate_dict)

        # Check if any 'diagdate' values are NaN
        if df_cycles["diagdate"].isnull().any():
            print("Not all 'nopho_nr' values in df_blood are present in df_cycles.")
            df_cycles.dropna(subset=["diagdate"], inplace=True)  # or raise an exception

        # Ensure 'diagdate' and 't_start' are of datetime type in df_cycles after mapping
        df_cycles["diagdate"] = pd.to_datetime(df_cycles["diagdate"])
        df_cycles["t_start"] = pd.to_datetime(df_cycles["t_start"])

        # Now calculate 'days_from_diagnosis'
        df_cycles["days_from_diagnosis"] = (
            df_cycles["t_start"] - df_cycles["diagdate"]
        ).dt.days

        return df_cycles


"""
This class is responsible for identifying cycles in a time series where the values rise above or drop below a specified threshold.

Attributes:
- df (pd.DataFrame): The DataFrame containing the time series data. It should have columns for 'component', 'sample_time', 'reply_num', and 'infno'.
- component (str): The component for which to identify cycles.
- threshold (float): The threshold value to identify cycles.
- mode (str): The mode of operation. If "drop_below", cycles are identified when values drop below the threshold. If "rise_above", cycles are identified when values rise above the threshold.

Methods:
- interpolate_crossing_time(ts1, val1, ts2, val2, threshold): Interpolates the time at which the values cross the threshold between two time points.
- find_cycles_with_interpolation(series, infno_series): Identifies cycles in a time series using linear interpolation to estimate the exact time points where the values cross the threshold.
- identify_cycles(): Identifies cycles for the specified component in the DataFrame.
"""

"""
This function identifies cycles in a time series where the values rise above or drop below a specified threshold.
It uses linear interpolation to estimate the exact time points where the values cross the threshold.

Parameters:
- series (pd.Series): The time series data. The index should be the time points and the values should be the measurements.
- infno_series (pd.Series): The series of infusion numbers corresponding to the time series data.
- threshold (float): The threshold value to identify cycles.
- mode (str): The mode of operation. If "drop_below", cycles are identified when values drop below the threshold. If "rise_above", cycles are identified when values rise above the threshold.
The formula used for the interpolation is:

Interpolate the x value given y. The formula is:
.. math::

x = \\frac{{x_1 (y_0 - y) + x_0(y - y_1)}}{{y_0 - y_1}}

given the assumption that :math:`y_0 \\neq y_1` and :math:`x_1 \\neq x_0`.

Parameters:
- x0, y0: coordinates of the first point
- x1, y1: coordinates of the second point
- y: the y value at which to interpolate x

Returns:
- list: A list of dictionaries. Each dictionary represents a cycle and contains the start time, end time, time of extreme value, duration, deviation from threshold, start infusion number, and end infusion number.
"""

"""
This function identifies cycles for a specific component in a DataFrame where the values rise above or drop below a specified threshold.
It uses the find_cycles_with_interpolation function to identify the cycles.

Parameters:
- df (pd.DataFrame): The DataFrame containing the time series data. It should have columns for 'component', 'sample_time', 'reply_num', and 'infno'.
- component (str): The component for which to identify cycles.
- threshold (float): The threshold value to identify cycles.
- mode (str): The mode of operation. If "drop_below", cycles are identified when values drop below the threshold. If "rise_above", cycles are identified when values rise above the threshold.

Returns:
- pd.DataFrame: A DataFrame where each row represents a cycle. The columns are 't_start', 't_end', 't_extreme', 'duration', 'doc', 'start_infno', 'end_infno', and 'nopho_nr'.
"""

"""
Identifies cycles for a specific component in a DataFrame where the values rise above or drop below a specified threshold.
It uses the find_cycles_with_interpolation function to identify the cycles.

Parameters:
- df (pd.DataFrame): The DataFrame containing the time series data. It should have columns for 'component', 'sample_time', 'reply_num', and 'infno'.
- component (str): The component for which to identify cycles.
- threshold (float): The threshold value to identify cycles.
- mode (str): The mode of operation. If "drop_below", cycles are identified when values drop below the threshold. If "rise_above", cycles are identified when values rise above the threshold.

Returns:
- pd.DataFrame: A DataFrame where each row represents a cycle. The columns are 't_start', 't_end', 't_extreme', 'duration', 'doc', 'start_infno', 'end_infno', and 'nopho_nr'.
"""
