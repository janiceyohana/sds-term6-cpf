import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def process_input_data(date_input, open_balance_input, new_cases_input, closed_cases_input, df_combined):
    # Check if all inputs are provided
    if date_input and open_balance_input is not None and new_cases_input is not None and closed_cases_input is not None:
        try:
            # Convert inputs to appropriate types
            open_balance = float(open_balance_input)
            new_cases = float(new_cases_input)
            closed_cases = float(closed_cases_input)
        except ValueError:
            st.error("Error: Please enter valid numbers for balances.")
            return df_combined  # Return original DataFrame if input is invalid

        # Convert date_input to string for comparison
        date_input_str = date_input.strftime('%Y-%m-%d')

        # Check if date_input already exists in DataFrame
        if date_input_str in df_combined['Date'].values:
            # Replace existing entry
            df_combined.loc[df_combined['Date'] ==
                            date_input_str, 'Open Balances'] = open_balance
            df_combined.loc[df_combined['Date'] ==
                            date_input_str, 'Closed'] = closed_cases
            df_combined.loc[df_combined['Date'] ==
                            date_input_str, 'New Cases'] = new_cases
            st.success(f"Data for {date_input_str} updated successfully!")
        else:
            # Append new data to DataFrame
            new_data = {
                'Date': date_input_str,
                'Open Balances': open_balance,
                'Closed': closed_cases,
                'New Cases': new_cases
            }
            df_combined = df_combined.append(new_data, ignore_index=True)
            st.success("New data added!")

        # Sort DataFrame by date
        df_combined['Date'] = pd.to_datetime(df_combined['Date'])
        df_combined.sort_values(by='Date', inplace=True)
        df_combined.reset_index(drop=True, inplace=True)

        return df_combined

    else:
        st.error("Error: Please fill in all input fields.")
        return df_combined  # Return original DataFrame if inputs are missing


def render():
    df_combined = pd.read_csv('data/2022-2024_Stats.csv')
    # Display input form
    st.header("Input Data")
    st.caption("Input Yesterday's Actual Case Data")

    with st.container():
        with st.form(key='input_data_form'):
            col1, col2 = st.columns(2)

            with col1:
                # Get yesterday's date
                yesterday = datetime.now() - timedelta(days=1)
                max_date = yesterday

                # Date input restricted to yesterday's date
                date_input = st.date_input(
                    "Date", min_value=max_date, max_value=max_date, value=yesterday, key=None)

                open_balance_input = st.number_input(
                    "Open Balance", value=0, min_value=0, step=1)

            with col2:
                new_cases_input = st.number_input(
                    "New Incoming Cases", value=0, min_value=0, step=1)
                closed_cases_input = st.number_input(
                    "Total Closed Cases", value=0, min_value=0, step=1)

            # Form submission button for input data
            submit_button = st.form_submit_button(label='Submit')

            if submit_button:
                # Process the input data
                df_combined = process_input_data(
                    date_input, open_balance_input, new_cases_input, closed_cases_input, df_combined)
                # st.write(df_combined)  # Display DataFrame in the app
                # Save DataFrame to CSV file
                df_combined.to_csv('data/2022-2024_Stats.csv', index=False)

if __name__ == "__main__":
    render()
