This repository holds a PySpark-based solution for similarity search on a large dataset of textual and numerical data, like product descriptions, user ratings, or social media text. The project performs several key tasks:
1. Preprocessing: Text data goes through preprocessing - tokenization, stop-word removal, and TF-IDF transformation. For numerical data, normalization ensures that there will be no significant variation between feature values.
2. Similarity Metrics: There are implementations for multiple similarity search algorithms within this repository
- Cosine Similarity
  - Jaccard Similarity
  - Laplacian Similarity
  - Euclidean Distance
  - Manhattan Distance
  - Hamming Distance

3. Optimization: The PySpark jobs are optimized using caching, partitioning, and broadcast variables to improve performance and scalability of similarity calculations.

4. Analysis: This repository provides a complete performance comparison of these similarity algorithms. The analysis includes accuracy along with execution time. It also visualizes how efficient and effective the implementation is.

This project aims to illustrate the different ways to implement and optimize similarity metrics using PySpark, distributing computationally expensive operations for larger datasets.

Key Features

* Implementation using PySpark
* Preprocessing of text and numbers
- Multicriteria similarity metrics and comparisons
- Optimized performance on large datasets
- Visualizations of results

This project is useful for anybody interested in working with big datasets and performing similarity-based searches in a distributed computing environment.
![Screenshot (118)](https://github.com/user-attachments/assets/48e87d04-79e3-4cb9-8b1b-bf0fcbd3b579)
![Screenshot (119)](https://github.com/user-attachments/assets/9676d62d-0ea1-4209-985c-15dd3a6a7d7d)
![Screenshot (120)](https://github.com/user-attachments/assets/b4ba1830-b48d-4e7e-89ea-f064a96fd1bd)
![Screenshot (117)](https://github.com/user-attachments/assets/619182dc-26c5-44ba-8b71-416bf62e88a6)
