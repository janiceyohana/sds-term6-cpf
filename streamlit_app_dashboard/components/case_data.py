import streamlit as st
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import itertools
import numpy as np
import datetime
import os
import time
import uuid

# Get the directory path of the current script file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Navigate to the parent directory (one level up)
parent_dir = os.path.dirname(current_dir)

# Construct the file paths to the CSV files
manpower_file_path = os.path.join(parent_dir, 'data', 'Manpower_Working.csv')
productivity_file_path = os.path.join(
    parent_dir, 'data', 'Case_Closure_(Oct22-Mar24).csv')
cases_file_path = os.path.join(parent_dir, 'data', '2022-2024_Stats.csv')
surge_path = os.path.join(parent_dir, 'data', 'Surge_Amt.csv')

# Read the CSV files
manpower_df = pd.read_csv(manpower_file_path)
df_productivity = pd.read_csv(productivity_file_path)
cases_df = pd.read_csv(cases_file_path)
surge_df = pd.read_csv(surge_path)

# Set the default values based on the last row of the DataFrame
last_row = manpower_df.iloc[-1]
default_num_cse = last_row['CSE']
default_num_csa = last_row['CSA']
default_num_temps = last_row['Temps']

surge_amt = []

# Suppress deprecation warnings for st.experimental_get_query_params()
warnings.filterwarnings("ignore", category=DeprecationWarning)

def calculate_new_cases_pred(train_days, num_steps, p, d, q, surge_amt):
    # Calculate new_cases_pred using ARIMA or other methods
    new_cases_pred = arima_dynamic_forecast(train_days, num_steps, p, d, q, surge_amt, var_to_pred="New Cases")
    return new_cases_pred

# Function to set default values based on session state or browser cookies


def set_default_values():
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = "Weekly"

    if 'num_cse' not in st.session_state:
        # Check if the value exists in query parameters
        num_cse_query = st.experimental_get_query_params().get('num_cse')
        st.session_state.num_cse = int(
            num_cse_query[0]) if num_cse_query else default_num_cse

    if 'num_temps' not in st.session_state:
        # Check if the value exists in query parameters
        num_temps_query = st.experimental_get_query_params().get('num_temps')
        st.session_state.num_temps = int(
            num_temps_query[0]) if num_temps_query else default_num_temps

    if 'num_csa' not in st.session_state:
        # Check if the value exists in query parameters
        num_csa_query = st.experimental_get_query_params().get('num_csa')
        st.session_state.num_csa = int(
            num_csa_query[0]) if num_csa_query else default_num_csa

# Function for ARIMA dynamic forecasting


def arima_dynamic_forecast(train_days, num_steps, p, d, q, surge_amt, var_to_pred):
    # Extract the specified column from the training data
    combined_train_data = train_days[var_to_pred]

    # Initialize history with combined training data
    history = [x for x in combined_train_data]

    predictions = []

    # Iterate over the number of time steps to make predictions
    for i in range(num_steps):
        # Fit ARIMA model with seasonal differencing
        model = ARIMA(history, order=(p, d, q))
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


# Define prod_forecast function


def prod_forecast(manpower_days, num_steps, p, d, q, manpower_df, surge_amt):
    # predict csa
    csa_train_data = manpower_days['Avg Case Closed Per CSA']

    # Initialize history with combined training data
    csa_history = [x for x in csa_train_data]

    csa = list()

    csa_train_data = manpower_days['Avg Case Closed Per CSA']
    cse_train_data = manpower_days['Avg Case Closed Per CSE']
    temps_train_data = manpower_days['Avg Case Closed per Temp']

    csa_history = [x for x in csa_train_data]
    cse_history = [x for x in cse_train_data]
    temp_history = [x for x in temps_train_data]

    csa_predictions = []
    cse_predictions = []
    temp_predictions = []

    # Iterate over the number of time steps to make predictions for 2024
    for i in range(num_steps):
        # Fit ARIMA model with seasonal differencing
        model = ARIMA(csa_history, order=(p, d, q))
        model_fit = model.fit()

        # Forecast the next value
        yhat = model_fit.forecast()[0]
        # Append the forecasted value to predictions
        csa.append(yhat)
        # Round down to the nearest whole number
        yhat_rounded = np.floor(yhat)

        # Update the history with the forecasted value
        csa_history.append(yhat_rounded)

    # predict cse
    cse_train_data = manpower_days['Avg Case Closed Per CSE']

    # Initialize history with combined training data
    cse_history = [x for x in cse_train_data]

    cse = list()

    # Iterate over the number of time steps to make predictions for 2024
    for i in range(num_steps):
        # Fit ARIMA model with seasonal differencing
        model = ARIMA(cse_history, order=(p, d, q))
        model_fit = model.fit()

        # Forecast the next value
        yhat = model_fit.forecast()[0]
        # Append the forecasted value to predictions
        cse.append(yhat)
        # Round down to the nearest whole number
        yhat_rounded = np.floor(yhat)
        # Update the history with the forecasted value
        cse_history.append(yhat_rounded)

    # predict temps
    temps_train_data = manpower_days['Avg Case Closed per Temp']

    # Initialize history with combined training data
    temp_history = [x for x in temps_train_data]

    temp = list()

    # Iterate over the number of time steps to make predictions for 2024
    for i in range(num_steps):
        # Fit ARIMA model with seasonal differencing
        model = ARIMA(temp_history, order=(p, d, q))
        model_fit = model.fit()

        # Forecast the next value
        yhat = model_fit.forecast()[0]
        # Append the forecasted value to predictions
        temp.append(yhat)
        # Round down to the nearest whole number
        yhat_rounded = np.floor(yhat)
        # Update the history with the forecasted value
        temp_history.append(yhat_rounded)

    # Multiply forecasted values by number of agents
    manpower_df = pd.read_csv(manpower_file_path)
    num_csa = manpower_df.iloc[-1]['CSA']
    num_cse = manpower_df.iloc[-1]['CSE']
    num_temps = manpower_df.iloc[-1]['Temps']

    # Calculate cases closed for each time step
    cases_closed = []
    for csa_val, cse_val, temp_val in zip(csa, cse, temp):
        total_cases_closed = csa_val * num_csa + \
            cse_val * num_cse + temp_val * num_temps
        cases_closed.append(total_cases_closed)
    # Round down to the nearest whole number
    cases_closed = np.floor(cases_closed)
    return cases_closed


def render(num_steps, p, d, q, start_date, end_date, manpower_days, button_options, train_days):
    st.header("Case Data Simulation")

    set_default_values()  # Ensure default values are set even after page refresh

    with st.container():
        # Generate a unique key for the form using a timestamp
        form_key = f'case_data_form_{uuid.uuid4()}'
        with st.form(key=form_key):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Manpower Allocation")
                st.caption("Number of agents editable")
                # Display the number inputs with default values
                num_cse_input = st.number_input(
                    "Number of CSE", value=st.session_state.num_cse, min_value=0, step=1)
                num_temps_input = st.number_input(
                    "Number of Temps", value=st.session_state.num_temps, min_value=0, step=1)
                num_csa_input = st.number_input(
                    "Number of CSA", value=st.session_state.num_csa, min_value=0, step=1)
            with col2:
                st.subheader("Expected change in Enquiry Volume")
                st.caption("Value of percentage editable")
                surge_amt = st.number_input(
                    "Expected Change in Percentage (e.g. +2 / -1)", step=1)

            # Define the form submit button
            submit_button = st.form_submit_button(label='Update Graphs')

            if submit_button:
                # Store the current inputs in session state and browser cookies
                st.session_state.num_cse = num_cse_input
                st.session_state.num_temps = num_temps_input
                st.session_state.num_csa = num_csa_input
                st.experimental_set_query_params(
                    num_cse=num_cse_input, num_temps=num_temps_input, num_csa=num_csa_input)
                st.session_state.surge_amt = surge_amt
                st.experimental_set_query_params(surge_amt=surge_amt)

                # Update CSV file with new values
                manpower_df.iloc[-1,
                                 manpower_df.columns.get_loc('CSE')] = num_cse_input
                manpower_df.iloc[-1,
                                 manpower_df.columns.get_loc('CSA')] = num_csa_input
                manpower_df.iloc[-1,
                                 manpower_df.columns.get_loc('Temps')] = num_temps_input
                manpower_df.to_csv(manpower_file_path, index=False)
                surge_df.iloc[-1,
                                 surge_df.columns.get_loc('surge_amt')] = surge_amt
                surge_df.to_csv(surge_path.column, index=False)

                st.success("Graphs updated successfully!")

            with st.container(border=True):
                st.subheader("Simulated Open Balance")
                st.write(f"From {start_date} to {end_date}")

                time_index = pd.date_range(start=start_date, end=end_date)

                # Call arima_dynamic_forecast to forecast open balances
                open_bal_predictions = arima_dynamic_forecast(
                    train_days, num_steps, p, d, q, surge_amt, var_to_pred='Open Balances')
                day_0_pred = open_bal_predictions[0]

                sim_open = []
                sim_open.append(day_0_pred)

                # Calculate simulated open balances
                new_cases_pred = arima_dynamic_forecast(
                    train_days, num_steps, p, d, q, surge_amt, var_to_pred="New Cases")

                # Calculate cases_closed using prod_forecast
                cases_closed = prod_forecast(
                    manpower_days, num_steps, p, d, q, manpower_df, surge_amt)

                for i in range(num_steps - 1):
                    # Check for NaN values before performing calculations
                    if np.isnan(new_cases_pred[i]) or np.isnan(cases_closed[i]):
                        # Handle NaN values here (e.g., replace with a default value)
                        pass
                    else:
                        # Calculate simulated open balance for the current time step
                        new_balance = sim_open[i] + \
                            new_cases_pred[i] - cases_closed[i]
                        # Round down to the nearest whole number
                        new_balance_rounded = np.floor(new_balance)
                        # Append the rounded balance to the list of simulated open balances
                        sim_open.append(new_balance_rounded)

                # Add tab selection
                tab_graph, tab_table = st.tabs(["Graph", "Table"])

                with tab_graph:
                    # Plot simulated open balances
                    plt.figure(figsize=(18, 6))
                    plt.plot(time_index[:len(sim_open)], sim_open,
                             label='Simulated Open Balances', color='green')
                    plt.title('')
                    plt.xlabel('Date')
                    plt.xticks(rotation=45)
                    plt.ylabel('Open Balance')
                    plt.grid(True)
                    # Display plot using st.pyplot()
                    st.pyplot(plt.gcf())
                with tab_table:
                    # Calculate the table data
                    open_table_data = pd.DataFrame({
                        'Date': time_index[:len(sim_open)],
                        'Simulated Open Balance': sim_open
                    })
                    # Display the table
                    st.write(open_table_data)

            colnew, colclosed = st.columns([1, 1])
            with colnew:
                with st.container(border=True):
                    st.subheader("Forecasted New Incoming Cases")
                    st.write(f"From {start_date} to {end_date}")

                    time_index = pd.date_range(start=start_date, end=end_date)

                    # Perform ARIMA dynamic forecasting for New Cases
                    original_data = cases_df['New Cases'][-60:]
                    dynamic_predictions = arima_dynamic_forecast(
                        {'New Cases': original_data}, num_steps, p, d, q, surge_amt, var_to_pred="New Cases")

                    # Add tab selection
                    tab_graph, tab_table = st.tabs(["Graph", "Table"])
                    with tab_graph:
                        plt.figure(figsize=(10, 6))
                        plt.plot(time_index[:len(dynamic_predictions)], dynamic_predictions,
                                 label='Predictions 2024', color='red')
                        plt.xlabel('Date')
                        plt.xticks(rotation=45)
                        plt.ylabel('New Cases')
                        plt.grid(True)
                        st.pyplot(plt.gcf())
                    with tab_table:
                        with tab_table:
                            # Calculate the table data
                            new_table_data = pd.DataFrame({
                                'Date': time_index[:len(dynamic_predictions)],
                                'Forecasted New Incoming Cases': dynamic_predictions
                            })
                            # Display the table
                            st.write(new_table_data)

            with colclosed:
                with st.container(border=True):
                    st.subheader("Simulated Total Closed Cases")
                    st.write(f"From {start_date} to {end_date}")

                    time_index = pd.date_range(start=start_date, end=end_date)

                    # Perform ARIMA dynamic forecasting for Total Closed Cases
                    # Placeholder, replace with your data
                    cases_closed = prod_forecast(
                        manpower_days, num_steps, p, d, q, manpower_df, surge_amt)

                    # Add tab selection
                    tab_graph, tab_table = st.tabs(["Graph", "Table"])
                    with tab_graph:
                        plt.figure(figsize=(10, 6))
                        plt.plot(time_index[:len(cases_closed)], cases_closed,
                                 label='Simulated Total Closed Cases', color='blue')
                        plt.xlabel('Date')
                        plt.xticks(rotation=45)
                        plt.ylabel('Total Closed Cases')
                        plt.grid(True)
                        st.pyplot(plt.gcf())
                    with tab_table:
                        # Calculate the table data
                        closed_table_data = pd.DataFrame({
                            'Date': time_index[:len(cases_closed)],
                            'Simulated Total Closed Cases': cases_closed
                        })
                        # Display the table
                        st.write(closed_table_data)

    return new_cases_pred


if __name__ == "__main__":
    render()
