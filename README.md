# Amazon-Music-Clustering
This project applies unsupervised learning, specifically clustering algorithms such as K-Means, to identify natural groupings in music data.
Through feature scaling, dimensionality reduction (PCA), and visualization techniques, it explores the underlying structure of the dataset to discover patterns that relate to genre or mood similarities.
________________________________________
 Objectives
•	Understand and preprocess audio feature data.
•	Apply K-Means and/or Hierarchical clustering to group similar songs.
•	Determine the optimal number of clusters using the Elbow method and Silhouette score.
•	Visualize clusters in reduced dimensions using PCA.
•	Analyze cluster profiles and interpret results.
________________________________________
 Technologies Used
•	Python 3.10+
•	Libraries:
o	pandas – data manipulation
o	numpy – numerical operations
o	scikit-learn – clustering (KMeans, PCA, scaling)
o	matplotlib, seaborn – visualizations
o	streamlit – interactive web interface
________________________________________
 Methodology
1.	Data Loading & Cleaning
o	Handle missing or irrelevant features.
o	Standardize feature scales.
2.	Feature Scaling
o	Applied StandardScaler to normalize features.
3.	Clustering
o	Used K-Means to form song clusters.
o	Experimented with different cluster counts.
4.	Model Evaluation
o	Applied Elbow Method and Silhouette Analysis.
5.	Dimensionality Reduction
o	Used PCA (2D) for cluster visualization.
6.	Cluster Profiling
o	Computed average feature values per cluster.
o	Visualized results using heatmaps and bar plots.
7.	Streamlit Dashboard
o	Built an interactive web app to explore clusters and visualizations.
________________________________________
 Results
•	Songs were successfully grouped into clusters based on audio similarity.
•	Each cluster tends to represent a distinct musical style or mood (e.g., energetic dance tracks vs. calm acoustic songs).
•	PCA and heatmap visualizations provided clear insights into feature groupings.
________________________________________
 Conclusion
This project demonstrates how unsupervised learning can effectively categorize music tracks without labeled data.
By leveraging clustering and visualization techniques, we can gain insights into hidden patterns within musical datasets — a step toward automated music organization and recommendation systems.

