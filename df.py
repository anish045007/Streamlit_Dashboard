import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.express as px

# Initialize Streamlit app
st.set_page_config(layout="wide")

# Provide the URL to your CSV file on GitHub
data_url = "https://raw.githubusercontent.com/anish045007/Streamlit_Dashboard/main/train.csv"

# Load your dataset
try:
    df = pd.read_csv(data_url)

    # Create the genre distribution plot before using it
    fig_genre_distribution = px.bar(
        x=df['track_genre'].value_counts().index,
        y=df['track_genre'].value_counts().values,
        labels={'x': 'Genre', 'y': 'Count'},
        title='Distribution of Genres'
    )

    # Add slicers for interactivity
    genre_slicer = st.sidebar.multiselect('Select Genre(s)', df['track_genre'].unique())
    artist_slicer = st.sidebar.multiselect('Select Artist(s)', df['artists'].unique())

    # Filter the DataFrame based on slicer selections
    filtered_df = df[df['track_genre'].isin(genre_slicer) & df['artists'].isin(artist_slicer)]

    # Display filtered data
    st.write('Filtered Data:')
    st.write(filtered_df)

    # Query 1: Average Popularity by Explicitness
    st.subheader("Query 1: Average Popularity by Genre")
    average_popularity_by_explicitness = df.groupby('track_genre')['popularity'].mean()
    st.bar_chart(average_popularity_by_explicitness)

    # Query 2: Multiple Linear Regression
    st.subheader("Query 2: Multiple Linear Regression")
    X = df[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    X = sm.add_constant(X)  # Add a constant for the intercept
    y = df['popularity']
    model = sm.OLS(y, X).fit()

    # Visualize coefficients using Plotly bar chart
    fig_coefficients = px.bar(
        x=model.params.drop('const').index,
        y=model.params.drop('const').values,
        labels={'x': 'Variables', 'y': 'Coefficient Value'},
        title='Multiple Linear Regression Coefficients'
    )
    st.plotly_chart(fig_coefficients)

    # Query 3: Relationship between Energy and Valence
    st.header('Query 3: Relationship between Energy and Valence')
    # Aggregated summary plot
    fig_energy_valence = plt.figure()
    aggregated_df = df.groupby('energy')['valence'].mean().reset_index()
    sns.lineplot(x='energy', y='valence', data=aggregated_df)
    plt.xlabel('Energy')
    plt.ylabel('Mean Valence')
    st.pyplot(fig_energy_valence)

    # Multiple Variable Analysis
    st.header("Multiple Variable Analysis")

    # Query 4: Violin plot for 'popularity' vs. 'explicit'
    st.subheader("Query 4: Violin Plot for 'popularity' vs. 'explicit'")
    fig, ax = plt.subplots()
    sns.violinplot(x='explicit', y='popularity', data=df, ax=ax)
    st.pyplot(fig)

    # Multiple Variable Analysis
    st.header("Multiple Variable Analysis")

    # Query 5: Box plot for 'energy' across different 'time_signature'
    st.subheader("Query 5: Box Plot for 'energy' across different 'time_signature'")
    fig, ax = plt.subplots()
    sns.boxplot(x='time_signature', y='energy', data=df, ax=ax)
    st.pyplot(fig)

    # Multiple Variable Analysis
    st.header("Multiple Variable Analysis")

    # Query 6: Explore factors contributing to popularity
    st.subheader("Query 6: Relationship between Popularity and Features")
    # Sample a subset of the data
    sampled_df = df.sample(frac=0.2, random_state=10)
    # Hexbin plot using matplotlib
    fig_popularity_features = plt.figure(figsize=(12, 6))
    plt.hexbin(x=sampled_df['danceability'], y=sampled_df['popularity'], C=sampled_df['energy'], gridsize=25, cmap='viridis', alpha=0.7)
    plt.xlabel('Danceability')
    plt.ylabel('Popularity')
    plt.colorbar(label='Energy')
    plt.title('Hexbin Plot: Danceability vs Popularity with Energy Color Coding')
    st.pyplot(fig_popularity_features)

    # Query 7: Compare characteristics of explicit and non-explicit tracks
    st.subheader("Query 7: Characteristics of Explicit vs. Non-Explicit Tracks")
    fig_explicit_comparison = plt.figure(figsize=(12, 6))
    sns.boxplot(x='explicit', y='popularity', data=df)
    plt.xlabel('Explicit')
    plt.ylabel('Popularity')
    st.pyplot(fig_explicit_comparison)

    # Query 8: Create a correlation matrix
    st.subheader("Query 8: Correlation Matrix")

    # Select only numeric columns for the correlation matrix
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numeric_columns].corr()

    # Plot the correlation matrix
    fig_corr_matrix, ax_corr_matrix = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax_corr_matrix)

    # Display the plot
    st.pyplot(fig_corr_matrix)

except pd.errors.EmptyDataError:
    st.error("Error: Empty dataset.")
except Exception as e:
    st.error(f"Error: {e}")

# Run the Streamlit app
st.sidebar.text('Note: Use the slicers to filter data')
st.sidebar.text('Data Source: Spotify Dataset')
st.sidebar.text('Dashboard created using Streamlit and Seaborn')
st.sidebar.text('By Anish Rawat')
