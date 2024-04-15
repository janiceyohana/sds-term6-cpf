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
productivity_file_path = os.path.join(parent_dir, 'data', 'Case_Closure_(Oct22-Mar24).csv')
cases_file_path = os.path.join(parent_dir, 'data', '2022-2024_Stats.csv')
surge_path = os.path.join(parent_dir, 'data', 'Surge_Amount.csv')

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

last_row_surge = surge_df.iloc[-1]
default_surge_amt = last_row_surge['surge_amt']

# Suppress deprecation warnings for st.experimental_get_query_params()
warnings.filterwarnings("ignore", category=DeprecationWarning)

def update_surge_amount(surge_amt):
  # Get the current date
  current_date = datetime.date.today().strftime('%Y-%m-%d')

  # Update the session state
  st.session_state.surge_amt = surge_amt
  st.experimental_set_query_params(surge_amt=surge_amt)

  # Read the surge data
  surge_df = pd.read_csv(surge_path)

  # Create a new row with current date and surge amount
  new_row = {'Date': current_date, 'surge_amt': surge_amt}

  # Append the new row to the DataFrame
  surge_df = surge_df.append(new_row, ignore_index=True)

  # Print the surge_amt and surge_df after appending
  print(f"Surge Amountttt: {surge_amt}")
  print(surge_df)  # This will print the entire DataFrame

  # Save the updated DataFrame to the CSV file
  surge_df.to_csv(surge_path, index=False)

def set_default_values():
    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = "Weekly"

    if 'num_cse' not in st.session_state:
        num_cse_query = st.experimental_get_query_params().get('num_cse')
        st.session_state.num_cse = int(num_cse_query[0]) if num_cse_query else default_num_cse

    if 'num_temps' not in st.session_state:
        num_temps_query = st.experimental_get_query_params().get('num_temps')
        st.session_state.num_temps = int(num_temps_query[0]) if num_temps_query else default_num_temps

    if 'num_csa' not in st.session_state:
        num_csa_query = st.experimental_get_query_params().get('num_csa')
        st.session_state.num_csa = int(num_csa_query[0]) if num_csa_query else default_num_csa

    if 'surge_amt' not in st.session_state:
        surge_amt_query = st.experimental_get_query_params().get('surge_amt')
        st.session_state.surge_amt = int(surge_amt_query[0]) if surge_amt_query else default_surge_amt

def calculate_new_cases_pred(train_days, num_steps, p, d, q, surge_amt):
    # Calculate new_cases_pred using ARIMA or other methods
    new_cases_pred = arima_dynamic_forecast(train_days, num_steps, p, d, q, surge_amt, var_to_pred="New Cases")
    return new_cases_pred

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
        yhat *= (1 + surge_amt / 100)  # <- Use surge_amt directly
        # Round down to the nearest whole number
        yhat_rounded = np.floor(yhat)
        # Append the forecasted value to predictions
        predictions.append(yhat_rounded)

        # Update the history with the forecasted value
        history.append(yhat)
    return predictions

# Define prod_forecast function
def prod_forecast(manpower_days, num_steps, p, d, q, manpower_df):
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

    new_cases_pred = None  # Initialize with a default value

    # Get the directory path of the current script file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the parent directory (one level up)
    parent_dir = os.path.dirname(current_dir)

    # Construct the file paths to the CSV files
    surge_path = os.path.join(parent_dir, 'data', 'Surge_Amount.csv')

    # Read the CSV files
    surge_df = pd.read_csv(surge_path)

    # Construct the file paths to the CSV files
    last_row_surge = surge_df.iloc[-1]
    surge_amt = last_row_surge['surge_amt']

    with st.container():
        with st.form(key='case_data_form'):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Manpower Allocation")
                num_cse_input = st.number_input("Number of CSE", value=st.session_state.num_cse, min_value=0, step=1)
                num_temps_input = st.number_input("Number of Temps", value=st.session_state.num_temps, min_value=0, step=1)
                num_csa_input = st.number_input("Number of CSA", value=st.session_state.num_csa, min_value=0, step=1)
            with col2:
                st.subheader("Expected Change in Enquiry Volume")
                surge_amt_input = st.number_input("Expected Change in Percentage (e.g. +2 / -1)", step=1)

            submit_button = st.form_submit_button(label='Update Graphs')

        if submit_button:
            # Retrieve inputs from the form submission
            num_cse_input = num_cse_input  # The value is already retrieved from the form
            num_temps_input = num_temps_input  # The value is already retrieved from the form
            num_csa_input = num_csa_input  # The value is already retrieved from the form
            surge_amt_input = surge_amt_input  # The value is already retrieved from the form

            # Update the session state and browser cookies
            st.session_state.num_cse = num_cse_input
            st.session_state.num_temps = num_temps_input
            st.session_state.num_csa = num_csa_input
            st.experimental_set_query_params(num_cse=num_cse_input, num_temps=num_temps_input, num_csa=num_csa_input)

            # Update the CSV file with new values for manpower
            manpower_df.iloc[-1, manpower_df.columns.get_loc('CSE')] = num_cse_input
            manpower_df.iloc[-1, manpower_df.columns.get_loc('CSA')] = num_csa_input
            manpower_df.iloc[-1, manpower_df.columns.get_loc('Temps')] = num_temps_input
            manpower_df.to_csv(manpower_file_path, index=False)

            # Update the session state and browser cookies for surge amount
            st.session_state.surge_amt = surge_amt_input
            st.experimental_set_query_params(surge_amt=surge_amt_input)

            # Update the surge amount in surge_df
            surge_df.iloc[-1, surge_df.columns.get_loc('surge_amt')] = surge_amt_input
            surge_df.to_csv(surge_path, index=False)

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
                manpower_days, num_steps, p, d, q, manpower_df)

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
                    manpower_days, num_steps, p, d, q, manpower_df)

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
