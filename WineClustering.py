import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load the dataset
df = pd.read_csv("Wine.csv")


# Remove the Unwanted columns
X = df.drop(columns=['Magnesium', 'Proline', 'Customer_Segment'])

# Preprocess the data: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA for dimensionality reduction (optional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Determine the number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Sidebar
st.sidebar.title("Cluster Number Selection")

# Main content
st.title("Clustering Of Wine Types Based on Their Chemical Componants")



# User input for selecting the number of clusters
num_clusters = st.sidebar.slider("Select the Number of Clusters (3 is the Appropriate Value)", 2, 10, 3)


# Apply K-Means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters


# Visualize the clusters using PCA

st.subheader("Visualization of Wine Types Clustering")
fig, ax = plt.subplots(figsize=(10, 8))
# plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
ax.set_title('Clustering of Wine Types Based on Their Chemical Componants')
ax.set_xlabel('Chemical Types 1')
ax.set_ylabel('Chemical Types 2')
st.pyplot(fig)
st.set_option('deprecation.showPyplotGlobalUse', False) # Disable Warning


# Display the table with Clusters 
st.write("### Data Table With Cluster Assignments")
st.write(df)  

