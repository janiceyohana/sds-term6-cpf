import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from statsmodels.tsa.arima.model import ARIMA

# Load the manpower_working.csv file
manpower_df = pd.read_csv('./data/manpower_working.csv')

# Define ranges for case_closure_rate targets
case_closure_rate_range = {
    'CSA': (15, 25),
    'CSE': (10, 35),
    'Temps': (10, 25),
}

# Generate all combinations of values
combinations = list(itertools.product(
    *[range(start, end + 1) for start, end in case_closure_rate_range.values()]))


def arima_dynamic_forecast(train_days, num_steps, p, d, q, var_to_pred):
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

        # Append the forecasted value to predictions
        predictions.append(yhat)

        # Update the history with the forecasted value
        history.append(yhat)

    return predictions

# Define total case closed calculation function


def calculate_total_case_closed(targets, counts):
    total = 0
    for target, count in zip(targets, counts):
        total += target * count
    return total


# Define function to check if open_balances differ by at most, the threshold
def within_threshold(open_balances, open_diff):
    for i in range(len(open_balances) - 1):
        if abs(open_balances[i + 1] - open_balances[i]) > open_diff:
            return False
    return True


# Define function to find best combinations
def find_best_combinations(combinations, day_0_pred, num_steps, new_cases_pred, open_diff, range_start, range_end, csa_input, cse_input, temps_input):
    valid_combinations = {}

    # check if combination is within threshold
    for combination in combinations:
        manpower_open_sim = [day_0_pred]

        total_case_closed_perday_target = calculate_total_case_closed(
            combination, [csa_input, cse_input, temps_input])

        for i in range(num_steps - 1):
            manpower_open_sim.append(
                manpower_open_sim[i] + new_cases_pred[i] - total_case_closed_perday_target)

        if within_threshold(manpower_open_sim, open_diff=open_diff):
            valid_combinations[combination] = manpower_open_sim

    valid_combinations = {key: value for key, value in valid_combinations.items(
    ) if range_start <= value[-1] <= range_end}

    # find best combination by using the min combination
    if valid_combinations:
        best_combination = min(
            valid_combinations, key=lambda x: valid_combinations[x][-1])
        return valid_combinations[best_combination], best_combination
    else:
        return []


def analyse_pred(cases_closed, start_date, end_date, var_to_pred):
    # Create time index
    time_index_2024 = pd.date_range(start=start_date, end=end_date, freq='D')

    # Plotting
    plt.figure(figsize=(10, 6))

    # plt.plot(time_index_2024, open_balances_values, label='Correct 2024', color='blue')

    # Plot predicted values for 2024
    plt.plot(time_index_2024, cases_closed,
             label='Simulated '+var_to_pred, color='red')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.ylabel(var_to_pred)
    plt.legend()
    plt.grid(True)
    plt.show()
    # Display plot using st.pyplot()
    st.pyplot(plt.gcf())


def render(p, d, q, num_steps, train_days, start_date, end_date):
    st.header("Manpower Allocation Simulation")

    with st.form(key='manpower_change_form'):
        with st.container():
            st.subheader("Requirements")
            st.caption("Value of rate editable")
            col1, col2 = st.columns([1, 1])
            with col1:
                ob_balance_input = st.number_input(
                    "The difference in open balance between two subsequent days cannot be more than", value=0, min_value=0, step=1)
            with col2:
                final_ob_input = st.slider(
                    "At the last day of the week, the final open balance should be in between", min_value=0, max_value=10000, value=(0, 10000), step=1, key="final_ob_input")

            # Retrieve data for CSA, CSE, and Temps from manpower_df
            csa_input = manpower_df['CSA'].iloc[-1]  # Using the last value
            cse_input = manpower_df['CSE'].iloc[-1]  # Using the last value
            temps_input = manpower_df['Temps'].iloc[-1]  # Using the last value

        submit_button = st.form_submit_button(label='Run Simulation')

        if submit_button:
            st.success("Simulation done running! See below for results")

            # Predict day_0_pred using ARIMA model
            day_0_pred = arima_dynamic_forecast(
                train_days, num_steps, p, d, q, var_to_pred='Open Balances')[0]

            new_cases_pred = arima_dynamic_forecast(
                train_days, num_steps, p, d, q, var_to_pred='New Cases')

            # Call the function with necessary parameters
            opti_combi, manpower_amt = find_best_combinations(
                combinations, day_0_pred, num_steps, new_cases_pred, ob_balance_input, final_ob_input[0], final_ob_input[1], csa_input, cse_input, temps_input)

            best_csa_no = manpower_amt[0]
            best_cse_no = manpower_amt[1]
            best_temps_no = manpower_amt[2]

            # Display the results in a new box
            with st.container(border=True):
                st.subheader("Results")
                colgraph, colgap, colresult = st.columns([4, 1, 3])
                with colgraph:
                    # Create a bar graph for Monday and a line graph for the rest of the week
                    fig, ax = plt.subplots()
                    analyse_pred(opti_combi, start_date, end_date,
                                 var_to_pred='Open Balance')
                with colgap:
                    st.empty()
                with colresult:
                    # # Calculate the total number of cases closed each day based on the optimal combination of productivity rates
                    # total_cases_closed = np.array(
                    #     opti_combi) - np.array(new_cases_pred) + np.array([best_csa_no, best_cse_no, best_temps_no])

                    # # Calculate the open balance for each day
                    # open_balances = [day_0_pred] + \
                    #     list(np.cumsum(total_cases_closed))

                    # Retrieve the open balance on the end date
                    open_balance_end_date = opti_combi[-1]
                    st.write(f"Open balance on the last day: {open_balance_end_date}")
                    st.write(
                        f"Suggested productivity rate of CSE: {best_cse_no}")
                    st.write(
                        f"Suggested productivity rate of CSA: {best_csa_no}")
                    st.write(
                        f"Suggested productivity rate of Temps: {best_temps_no}")


if __name__ == "__main__":
    render()
