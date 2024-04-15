import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA
import os
import numpy as np

# Get the directory path of the current script file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory (one level up)
parent_dir = os.path.dirname(current_dir)

# Construct the file paths to the CSV files
manpower_file_path = os.path.join(parent_dir, 'data', 'Manpower_Working.csv')
surge_path = os.path.join(parent_dir, 'data', 'Surge_Amt.csv')

# Read the CSV files
manpower_df = pd.read_csv(manpower_file_path)
surge_df = pd.read_csv(surge_path)


def arima_dynamic_forecast(train_days, num_steps, p, d, q, surge_amt, var_to_pred):
    # Extract the specified column from the training data
    combined_train_data = train_days[var_to_pred]

    # Initialize history with combined training data
    history = [x for x in combined_train_data]

    predictions = []

    # Iterate over the number of time steps to make predictions
    for i in range(num_steps):
        # Filter out non-numeric elements from history
        history_numeric = [x for x in history if isinstance(x, (int, float))]

        if not history_numeric:
            # Handle the case where history contains no numeric elements
            # You can choose to handle this case based on your requirements
            # For example, you can use a default value or skip the prediction
            yhat = 0  # Replace with your desired default value
        else:
            # Fit ARIMA model with seasonal differencing
            model = ARIMA(history_numeric, order=(p, d, q))
            model_fit = model.fit()

            # Forecast the next value
            yhat = model_fit.forecast()[0]
            # Apply surge percentage increase to the entire forecasted value
            yhat *= (1 + surge_amt / 100)

        # Round down to the nearest whole number
        yhat_rounded = np.floor(yhat)
        # Append the forecasted value to predictions
        predictions.append(yhat_rounded)

        # Update the history with the forecasted value
        history.append(yhat)

    return predictions


def calculate_new_cases_pred(train_days, num_steps, p, d, q, surge_amt):
    # Calculate new_cases_pred using ARIMA or other methods
    new_cases_pred = arima_dynamic_forecast(
        train_days, num_steps, p, d, q, surge_amt, var_to_pred="New Cases")
    # Check if there are any NaN values in new_cases_pred
    if np.isnan(new_cases_pred).any():
        # Replace NaN values with a default value (e.g., 0)
        new_cases_pred[np.isnan(new_cases_pred)] = 0
    return new_cases_pred


def total_cases_to_clear(final_end_bal, curr_open_bal, remain_days, new_cases_pred, num_steps):
    if remain_days == 0:
        case_to_clear = ((curr_open_bal-final_end_bal) +
                         sum(new_cases_pred))/num_steps
        return case_to_clear
    else:
        remove_pred_for = num_steps - remain_days
        new_cases_remain_pred = new_cases_pred[remove_pred_for:]
        case_to_clear = ((curr_open_bal-final_end_bal) +
                         sum(new_cases_remain_pred))/remain_days
        return case_to_clear


def avg_load_per_role(clear_per_day, csa_input, cse_input, temps_input):
    avg_load_cse = clear_per_day / \
        (cse_input+(csa_input*0.67)+(temps_input*0.54))
    avg_load_csa = avg_load_cse*0.67
    avg_load_temps = avg_load_cse*0.54
    return int(avg_load_cse), int(avg_load_csa), int(avg_load_temps)


def render(num_steps, train_days, p, d, q):
    st.title("Manpower Allocation Simulation")

    # Get the directory path of the current script file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the parent directory (one level up)
    parent_dir = os.path.dirname(current_dir)

    # Construct the file paths to the CSV files
    surge_path = os.path.join(parent_dir, 'data', 'Surge_Amt.csv')

    surge_df = pd.read_csv(surge_path)
    last_row_surge = surge_df.iloc[-1]
    surge_amt = last_row_surge['surge_amt']

    new_cases_pred = calculate_new_cases_pred(
        train_days, num_steps, p, d, q, surge_amt)

    # Check for NaN values
    if manpower_df.isnull().values.any():
        st.error("Error: NaN values found in manpower data.")
        return
    if surge_df.isnull().values.any():
        st.error("Error: NaN values found in surge data.")
        return

    with st.form(key='manpower_simulation_form'):
        with st.container():
            st.subheader("Requirements")
            # Sidebar inputs
            final_end_bal_input = st.number_input(
                "Target end balance on last day of week/month", min_value=0, step=1)
            curr_open_bal_input = st.number_input(
                "Today's open balance", min_value=0, step=1)
            remain_days_input = st.number_input(
                "How many days remaining in the week/month would you like to forecast for", min_value=0, step=1)

        # Define the form submit button
        submit_button = st.form_submit_button(label='Run Simulation')

        if submit_button:
            last_row = manpower_df.iloc[-1]
            cse_input = last_row['CSE']
            csa_input = last_row['CSA']
            temps_input = last_row['Temps']

            # Call functions from forecasting.py
            clear_per_day = total_cases_to_clear(
                final_end_bal_input, curr_open_bal_input, remain_days_input, new_cases_pred, num_steps)
            avg_load_cse, avg_load_csa, avg_load_temps = avg_load_per_role(
                clear_per_day, csa_input, cse_input, temps_input)

            with st.container(border=True):
                st.subheader("Results")
                # Display results
                st.write("Average load per role per day:")
                st.write(f"- CSE: {avg_load_cse}")
                st.write(f"- CSA: {avg_load_csa}")
                st.write(f"- Temps: {avg_load_temps}")


if __name__ == "__main__":
    render()
