import requests
import streamlit as st
import json

st.text('COVID Similarities Among Different Contries (based on data from Sept. 2023)')

# Create a text input field
text_variable = st.text_input("Enter the country you look similar countries for:", value="e.g. Taiwan")

# Define the URL of the web service
api_url = "http://localhost:5000/country/" + text_variable
error_response = 'Country not in the list.'

# Button to trigger the request
if st.button("Search"):
    response = requests.get(api_url)
    if response.status_code == 200:
        # Parse the JSON response
        response_data = json.loads(response.text)
        countries = response_data["similar_countries"]
        if countries != 'Country not in the list.':
            countries = json.loads(countries)

            table_data = []
            for key, value in countries.items():
                table_data.append([key, value])

            # Display the table 
            st.dataframe(table_data, height=750, hide_index=True, column_config={
                "0": "Country",
                "1": "Similarity Score"
            })
        else:
            st.write(error_response)
    else:
        st.error(f"API request failed: {response.text}")