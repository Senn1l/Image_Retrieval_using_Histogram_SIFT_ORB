# Building Image Retrieval using Traditional and Global Features
This project implements a **Content-Based Image Retrieval (CBIR)** system for building facade images using traditional local feature descriptors (SIFT, ORB) and global color histograms.

## Methodology
### Global Features
- Each image is represented using a color histogram with 768 bins (e.g., 256 bins per channel in RGB).
- For a given query image:
  - A histogram is computed.
  - It is compared with all stored histograms using the Histogram Intersection method:

    ![Formula](https://latex.codecogs.com/png.image?\dpi{110}&space;Similarity=\sum_{i=1}^{768}\min(hist1[i],hist2[i]))

  - A higher score indicates greater similarity.
- The top-k images with the highest similarity scores are returned.

### Local Features
- For each image, SIFT or ORB descriptors are extracted and stored.
- At query time:
  - Descriptors are extracted from the query image.
  - Each stored image is matched against the query using Brute-Force Matching:
    - SIFT uses Euclidean distance (L2).
    - ORB uses Hamming distance.
  - The total matching distance is used as a similarity score — lower means more similar.
- The top-k images with the smallest distances are returned.

## Dataset
- [The Timisoara Building Dataset](https://github.com/CipiOrhei/TMBuD)

## Evaluation
- Retrieval tested with top-k values (3, 5, 11, 21).
- Evaluation metric: Mean Average Precision (MAP) is used to measure retrieval quality.
- Procedure:
  - For each query image, the top-k most similar images are retrieved.
  - Precision is computed at each correct retrieval position.
  - Average Precision (AP) is calculated for each query.
  - The final MAP is the mean of all APs across the dataset.
- Results:
  - See the `report/` folder.
 
## How to Run
1. Follow `A_Practical_Guide_to_OpenCV.pdf` to set up OpenCV in a Visual Studio environment (or watch a tutorial on YouTube).
2. Refer to `_USERGUIDE_USAGE.txt` in the `data/` folder for instructions.
