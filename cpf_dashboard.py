import streamlit.components.v1 as components
import streamlit as st
from datetime import datetime, timedelta
import pytz
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Function to convert current date


def convert_utc_to_sgt(utc_dt):
    sgt_tz = pytz.timezone('Asia/Singapore')
    sgt_dt = utc_dt.astimezone(sgt_tz)
    # Format the date as "Tuesday, 19 March 2024"
    return sgt_dt.strftime('%A, %d %B %Y')

# Function to format date from "YYYY/MM/DD" to "YYYY-MM-DD"


def format_date(date_str):
    # Convert string to datetime object
    date_obj = datetime.strptime(date_str, "%Y/%m/%d")
    # Format datetime object as string in "YYYY-MM-DD" format
    return date_obj.strftime("%Y-%m-%d")


# Placeholder dataset
dataset = pd.DataFrame(
    columns=["Date", "Open Balances", "Closed", "New Cases"])

# Function to submit input data


def submit_input_data(date, open_balance, new_cases, closed_cases):
    global dataset

    # Check if the date already exists in the dataset
    if date in dataset["Date"].values:
        # Prompt user to confirm resubmission
        confirmation = st.confirm(
            f"Data for {date} already exists. Do you want to resubmit?")
        if confirmation:
            # Update the existing data
            dataset.loc[dataset["Date"] == date, [
                "Open Balances", "New Cases", "Closed"]] = open_balance, new_cases, closed_cases
            st.success("Data updated successfully!")
        else:
            st.info("Data not submitted.")
    else:
        # Append new data to the dataset
        dataset = dataset.append({"Date": date, "Open Balances": open_balance,
                                 "New Cases": new_cases, "Closed": closed_cases}, ignore_index=True)
        st.success("New data submitted successfully!")

# Function to simulate the data processing


def simulate_data_processing():
    # Placeholder for simulation logic
    result_data = {
        "No. of open balance by the end of the week": 10,
        "Suggested productivity rate of CSE": 20,
        "Suggested productivity rate of CSA": 30,
        "Suggested productivity rate of Temp": 40,
    }
    return result_data


# Load your existing dataset (replace this with your actual dataset)
case_data = pd.read_csv("data/2022-2024_Stats.csv")

# Function to check if the provided date already exists in the dataset


def date_exists(date):
    return date in case_data["Date"].values


# Set the page configuration and background color
st.set_page_config(page_title="Central Provident Fund Board", layout="wide")

col1, col2 = st.columns([3, 1])
with col1:
    st.html(
        """
        <div style="display: flex; align-items: center;">
            <picture>
                <img src="cpf_logo.png" width="54" style="margin-right: 10px;"/>
            </picture>
            <h1 style="text-align: left; font-size: 50px;">Central Provident Fund Board</h1>
        </div>
        """
    )

with col2:
    utc_now = datetime.utcnow()
    sgt_now = convert_utc_to_sgt(utc_now)
    st.markdown(
        f"<h1 style='text-align: right; height: 72px; font-size: 16px; align-content: center;'>{sgt_now}</h1>", unsafe_allow_html=True)
    components.html(
        """
        <style>
            body {
              display: flex;
              align-items: center;
              justify-content: flex-end;
              height: 35px;
              margin: 0;
            }

            .toggleContainer {
              position: relative;
              display: grid;
              grid-template-columns: repeat(2, 1fr);
              width: 200px;
              height: fit-content;
              font-family: "Source Sans Pro", sans-serif;
              border: 3px solid #BAD6D6;
              border-radius: 10px;
              background: #BAD6D6;
              font-weight: medium;
              color: #BAD6D6;
              cursor: pointer;
              margin: 0; /* Remove margin */
              padding: 0; /* Remove padding */
            }
            .toggleContainer::before {
              content: '';
              position: absolute;
              width: 50%;
              height: 100%;
              left: 0%;
              border-radius:9px;
              background: white;
              transition: all 0.3s;
            }
            .toggleCheckbox:checked + .toggleContainer::before {
                left: 50%;
            }
            .toggleContainer div {
              padding: 6px;
              text-align: center;
              z-index: 1;
            }
            .toggleCheckbox {
              display: none;
            }
            .toggleCheckbox:checked + .toggleContainer div:first-child{
              color: #0C6464;
              transition: color 0.3s;
            }
            .toggleCheckbox:checked + .toggleContainer div:last-child{
              color: #343434;
              transition: color 0.3s;
            }
            .toggleCheckbox + .toggleContainer div:first-child{
              color: #343434;
              transition: color 0.3s;
            }
            .toggleCheckbox + .toggleContainer div:last-child{
              color: #0C6464;
              transition: color 0.3s;
            }
        </style>
        <div style="height: 35px;">
            <input type="checkbox" id="toggle" class="toggleCheckbox" />
            <label for="toggle" class='toggleContainer'>
              <div>Weekly</div>
              <div>Monthly</div>
            </label>
        </div>
        """
    )

st.header("Input Data")
st.caption("Input Actual Case Data")

with st.container():
    with st.form(key='input_data_form'):
        col1, col2 = st.columns(2)

        with col1:
            date_input = st.date_input(
                "Date", min_value=None, max_value=None, value=None, key=None)
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
            st.success("Input data submitted successfully!")
            # Format the date
            formatted_date = format_date(date_input.strftime("%Y/%m/%d"))

            # Extract input data
            date = formatted_date
            open_balance = open_balance_input
            new_cases = new_cases_input
            closed_cases = closed_cases_input

            # Submit input data
            submit_input_data(date, open_balance, new_cases, closed_cases)

st.header("Case Data Simulation")

# Add a box to contain the Manpower Allocation and Expected change sections
with st.container():
    with st.form(key='case_data_form'):
        col1, col2 = st.columns([1, 1])
        with col1:
            with st.container():
                st.subheader("Manpower Allocation")
                st.caption("Number of agents editable")
                num_cse_input = st.number_input(
                    "Number of CSE", value=0, min_value=0, step=1)
                num_temps_input = st.number_input(
                    "Number of Temps", value=0, min_value=0, step=1)
                num_csa_input = st.number_input(
                    "Number of CSA", value=0, min_value=0, step=1)
        with col2:
            with st.container():
                st.subheader("Expected change in Enquiry Volume")
                st.caption("Value of percentage editable")
                change_input = st.number_input(
                    "Expected Change in Percentage (e.g. +2 / -1)", step=1)

        submit_button = st.form_submit_button(label='Update Graphs')

        if submit_button:
            st.success("Graphs updated successfully!")

st.header("Manpower Allocation Simulation")

# Add a box to contain the simulation sections
with st.container():
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

        submit_button = st.form_submit_button(label='Run Simulation')

        if submit_button:
            st.success("Simulation done running! See below for results")

            # Simulate data processing
            result_data = simulate_data_processing()

            # Display the results in a new box
            with st.expander('', expanded=True):
                st.subheader("Results")
                colgraph, colgap, colresult = st.columns([4, 1, 3])
                with colgraph:
                    # st.empty()
                    # Generate some sample data
                    days = ['Mon', 'Tue', 'Wed',
                            'Thu', 'Fri', 'Sat', 'Sun']
                    counts = np.random.randint(10, 100, size=7)

                    # Create a bar graph for Monday and a line graph for the rest of the week
                    fig, ax = plt.subplots()
                    ax.bar(days[:1], counts[:1], color='blue', label='Monday')
                    ax.plot(days[1:], counts[1:], marker='o',
                            color='red', label='simulated')

                    ax.set_xlabel('Day of the Week')
                    ax.set_ylabel('Open Balance')
                    ax.legend()
                    st.pyplot(fig)
                with colgap:
                    st.empty()
                with colresult:
                    for key, value in result_data.items():
                        st.write(f"{key}: {value}")
