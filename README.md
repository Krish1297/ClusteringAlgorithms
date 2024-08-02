# Custom Clustering Algorithms

This project features custom implementations of K-Means and DBSCAN clustering algorithms, applied to image segmentation. The implementation aims to showcase practical understanding and application of clustering techniques in a real-world scenario.

## Project Structure

- **`clustering_algorithms.py`**: Contains the implementations of `CustomKMeans` and `CustomDBSCAN`.
- **`clustering_task.py`**: Demonstrates the application of the implemented algorithms on an image (`giraffe.png`) to segment it based on color clusters.

## Installation and Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.7 or higher
- Necessary Python packages listed in `requirements.txt`

### Steps

1. **Clone the Repository**:
   git clone https://github.com/yourusername/clustering-algorithms.git
   cd clustering-algorithms

2. **Install Dependencies**:

Install the required packages using pip:

  pip install -r requirements.txt



### Detailed Description
## CustomKMeans
A custom implementation of the K-Means algorithm, characterized by:

# Parameters:

n_clusters: Number of clusters (k).
max_iter: Maximum iterations allowed.
random_state: Seed for reproducibility.
Key Methods:

fit(X): Trains the model using data X.
fit_predict(X): Trains the model and returns the cluster labels.

## CustomDBSCAN
A custom implementation of the DBSCAN algorithm, featuring:

# Parameters:

eps: Radius for neighborhood points.
min_samples: Minimum points required to form a dense region.
metric: Distance metric, default is 'euclidean'.
Key Methods:

fit(X): Fits the model on data X.
fit_predict(X): Fits the model and returns cluster labels.

### Results

The output includes:

Original Image: The unmodified input image.
K-Means Output: Image segmented using K-Means clustering.
DBSCAN Output: Image segmented using DBSCAN clustering.
These outputs demonstrate the application of clustering algorithms to practical image processing tasks.
