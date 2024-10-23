# read data present data,

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


# Re-defining the function to plot individual parameters
def plot_individual_param(
    date, values, y_label, lower_bound=None, upper_bound=None, title=None
):
    plt.figure(figsize=(10, 6))
    plt.plot(date, values, marker="o")
    if lower_bound is not None:
        plt.axhline(
            y=lower_bound,
            color="r",
            linestyle="--",
            label=f"Lower Bound ({lower_bound})",
        )
    if upper_bound is not None:
        plt.axhline(
            y=upper_bound,
            color="g",
            linestyle="--",
            label=f"Upper Bound ({upper_bound})",
        )
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.show()


# Plotting individual parameters


def main() -> None:

    blood_data = read_csv("Blood Test Results.csv")

    # Defining function to plot individual parameters with reference lines
    def plot_individual_param(
        date, values, y_label, lower_bound=None, upper_bound=None, title=None
    ):
        plt.figure(figsize=(10, 6))
        plt.plot(date, values, marker="o")
        if lower_bound is not None:
            plt.axhline(
                y=lower_bound,
                color="r",
                linestyle="--",
                label=f"Lower Bound ({lower_bound})",
            )
        if upper_bound is not None:
            plt.axhline(
                y=upper_bound,
                color="g",
                linestyle="--",
                label=f"Upper Bound ({upper_bound})",
            )
        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel(y_label)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.show()

    list_blood_count = [
        "Basofilocytter",
        "Eosinofilocytter",
        "Leukocytter",
        "Lymfocytter",
        "Metamyelo.+Myelo.+Promyelocytter",
        "Monocytter",
        "Neutrofilocytter",
        "Trombocytter",
    ]

    # Plotting individual parameters

    # Blood Cell Counts
    plot_blood_counts = [
        plot_individual_param(
            blood_data["Date"],
            blood_data["Basofilocytter"],
            "Basofilocytter (10^9/L)",
            0.01,
            0.10,
            "Basofilocytter (10^9/L) Over Time",
        ),
        plot_individual_param(
            blood_data["Date"],
            blood_data["Eosinofilocytter"],
            "Eosinofilocytter (10^9/L)",
            0.01,
            0.50,
            "Eosinofilocytter (10^9/L) Over Time",
        ),
        plot_individual_param(
            blood_data["Date"],
            blood_data["Leukocytter"],
            "Leukocytter (10^9/L)",
            3.5,
            8.8,
            "Leukocytter (10^9/L) Over Time",
        ),
        plot_individual_param(
            blood_data["Date"],
            blood_data["Lymfocytter"],
            "Lymfocytter (10^9/L)",
            1.0,
            3.5,
            "Lymfocytter (10^9/L) Over Time",
        ),
        plot_individual_param(
            blood_data["Date"],
            blood_data["Metamyelo.+Myelo.+Promyelocytter"],
            "Metamyelo.+Myelo.+Promyelocytter (10^9/L)",
            0.0,
            0.10,
            "Metamyelo.+Myelo.+Promyelocytter (10^9/L) Over Time",
        ),
        plot_individual_param(
            blood_data["Date"],
            blood_data["Monocytter"],
            "Monocytter (10^9/L)",
            0.20,
            0.80,
            "Monocytter (10^9/L) Over Time",
        ),
        plot_individual_param(
            blood_data["Date"],
            blood_data["Neutrofilocytter"],
            "Neutrofilocytter (10^9/L)",
            1.60,
            5.90,
            "Neutrofilocytter (10^9/L) Over Time",
        ),
        plot_individual_param(
            blood_data["Date"],
            blood_data["Trombocytter"],
            "Trombocytter (10^9/L)",
            145,
            390,
            "Trombocytter (10^9/L) Over Time",
        ),
    ]

    # Blood Cell Counts
    # plot_individual_param(blood_data['Date'], blood_data['Basofilocytter'], 'Basofilocytter (10^9/L)', 0.01, 0.10, 'Basofilocytter (10^9/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Eosinofilocytter'], 'Eosinofilocytter (10^9/L)', 0.01, 0.50, 'Eosinofilocytter (10^9/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Leukocytter'], 'Leukocytter (10^9/L)', 3.5, 8.8, 'Leukocytter (10^9/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Lymfocytter'], 'Lymfocytter (10^9/L)', 1.0, 3.5, 'Lymfocytter (10^9/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Metamyelo.+Myelo.+Promyelocytter'], 'Metamyelo.+Myelo.+Promyelocytter (10^9/L)', 0.0, 0.10, 'Metamyelo.+Myelo.+Promyelocytter (10^9/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Monocytter'], 'Monocytter (10^9/L)', 0.20, 0.80, 'Monocytter (10^9/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Neutrofilocytter'], 'Neutrofilocytter (10^9/L)', 1.60, 5.90, 'Neutrofilocytter (10^9/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Trombocytter'], 'Trombocytter (10^9/L)', 145, 390, 'Trombocytter (10^9/L) Over Time')

    # # Nutrient Levels
    # plot_individual_param(blood_data['Date'], blood_data['Jern'], 'Jern (µmol/L)', 9, 34, 'Jern (µmol/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Vitamin B12'], 'Vitamin B12 (pmol/L)', 250, None, 'Vitamin B12 (pmol/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Calcium'], 'Calcium (mmol/L)', 2.15, 2.51, 'Calcium (mmol/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Kalium'], 'Kalium (mmol/L)', 3.5, 4.4, 'Kalium (mmol/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Natrium'], 'Natrium (mmol/L)', 137, 144, 'Natrium (mmol/L) Over Time')

    # # Metabolic Parameters
    # plot_individual_param(blood_data['Date'], blood_data['HbA1c'], 'HbA1c (mmol/mol)', None, 48, 'HbA1c (mmol/mol) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['HDL'], 'HDL (mmol/L)', 1.0, None, 'HDL (mmol/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['LDL'], 'LDL (mmol/L)', None, 3.0, 'LDL (mmol/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Urat'], 'Urat (mmol/L)', 0.23, 0.48, 'Urat (mmol/L) Over Time')

    # # Hormone Levels
    # plot_individual_param(blood_data['Date'], blood_data['17-OH-Progesteron'], '17-OH-Progesteron (nmol/L)', None, 8.0, '17-OH-Progesteron (nmol/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['Androstendion'], 'Androstendion (nmol/L)', 1.70, 6.90, 'Androstendion (nmol/L) Over Time')
    # plot_individual_param(blood_data['Date'], blood_data['T3'], 'T3 (nmol/L)', 1.4, 2.8, 'T3 (nmol/L) Over Time')

    # Nutrient Levels
    st.selectbox(list_blood_count)


# selection function

# preprocessing


# model building

# AI infrence on unseen data - Zero shoot
if __name__ in "__main__"():
    main
