from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def fit_on_data(embedding_tensor):
    # Convert to NumPy array
    numpy_array = embedding_tensor.numpy()

    # Clustering
    n_clusters = 150
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(numpy_array)

    # Get Cluster Labels
    return kmeans

#def get_centers(embedding_tensor, images, kmeans:KMeans):
    


if __name__ == "__main__":
    print('hello')