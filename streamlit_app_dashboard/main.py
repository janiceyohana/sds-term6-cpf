import streamlit as st
from datetime import datetime, timedelta
import pytz
import pandas as pd

# Load other necessary components
from components import input_form, manpower_allocation_simulation, case_data


def main():
    st.set_page_config(
        page_title="Central Provident Fund Board", layout="wide")

    df_combined = pd.read_csv('data/2022-2024_Stats.csv')

    def convert_utc_to_sgt(utc_dt):
        sgt_tz = pytz.timezone('Asia/Singapore')
        sgt_dt = utc_dt.astimezone(sgt_tz)
        # Format the date as "Tuesday, 19 March 2024"
        return sgt_dt.strftime('%A, %d %B %Y')

    def format_date(date_str):
        # Convert string to datetime object
        date_obj = datetime.strptime(date_str, "%Y/%m/%d")
        # Format datetime object as string in "YYYY-MM-DD" format
        return date_obj.strftime("%Y-%m-%d")

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

    # Use the current date as the start date
    start_date = format_date(utc_now.strftime("%Y/%m/%d"))

    df_productivity = pd.read_csv('data/Case_Closure_(Oct22-Mar24).csv')

    # Define button options and their corresponding parameters
    button_options = {
        "Weekly": {"train_days": df_combined[-90:-30], "num_steps": 7, "p": 7, "d": 1, "q": 2,
                   "start_date": start_date, "td_days": 6, "manpower_days": df_productivity[-60:]},
        "Monthly": {"train_days": df_combined[-130:-30], "num_steps": 30, "p": 7, "d": 1, "q": 1,
                    "start_date": start_date, "td_days": 29, "manpower_days": df_productivity[-100:]}
    }

    # Display buttons using checkboxes
    selected_option = st.radio(
        "Select an option:", list(button_options.keys()))

    # Define a function to handle checkbox selection
    def on_checkbox_selected(selected_option):
        global train_days, num_steps, p, d, q, start_date, end_date
        print("Option selected:", selected_option)

        # Update parameters based on the selected option
        parameters = button_options.get(selected_option)
        if parameters:
            train_days = parameters["train_days"]
            num_steps = parameters["num_steps"]
            p, d, q = parameters["p"], parameters["d"], parameters["q"]
            start_date = parameters["start_date"]
            td_days = parameters["td_days"]
            end_date = datetime.strptime(
                start_date, "%Y-%m-%d") + timedelta(days=td_days)
            end_date = end_date.strftime("%Y-%m-%d")
            print(len(train_days), num_steps, p, d,
                  q, start_date, end_date, td_days)

    # Pass start_date and end_date to the Streamlit app
    input_form.render()

    # Handle checkbox selection
    on_checkbox_selected(selected_option)

    # Call case_data.render() after defining num_steps
    case_data.render(num_steps, p, d, q, start_date,
                     end_date, df_productivity, button_options, train_days)

    manpower_allocation_simulation.render(
        p, d, q, num_steps, train_days, start_date, end_date)


if __name__ == "__main__":
    main()
