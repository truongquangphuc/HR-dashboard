import pandas as pd
import streamlit as st
import plotly.express as px

# Assuming the CSV file is uploaded using Streamlit file uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Debugging output to ensure file is read correctly
        st.write("File loaded successfully!")
        st.write(df)

        # Handle potential errors during CSV reading
        if df.empty:
            st.error("The uploaded file is empty or invalid. Please try again.")
        elif 'Gender' not in df.columns:
            st.error("The 'Gender' column is missing in the uploaded file.")
        else:
            # Prepare data for the pie chart
            male_count = df[df['Gender'] == 'Male'].shape[0]
            female_count = df[df['Gender'] == 'Female'].shape[0]
            gender_counts = [male_count, female_count]
            gender_labels = ['Male', 'Female']

            # Debugging output to check gender counts
            st.write("Male count:", male_count)
            st.write("Female count:", female_count)

            # Create the pie chart with Plotly Express
            fig = px.pie(values=gender_counts, names=gender_labels,
                         title='Gender Distribution',
                         hole=0.4,  # Adjust hole size for better visibility
                         labels={'values': 'Count', 'names': 'Gender'})

            # Add informative hovertext
            fig.update_traces(textposition='inside', textinfo='label+percent')

            # Display the pie chart using Streamlit
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error reading the file: {e}")

else:
    st.info("Please upload a CSV file to display the gender distribution.")
