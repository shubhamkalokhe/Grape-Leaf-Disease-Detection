# Grape Leaf Disease Detection

## Overview

The Grape Leaf Disease Detection project aims to develop a method for the supervised classification of grape fields, particularly focusing on the detection of diseases in grape leaves. The project utilizes satellite imagery data, specifically Synthetic Aperture Radar (SAR) and Multi-Spectral Imagery (MSI), to classify grape fields and identify unhealthy crops based on their spectral characteristics.

## Methodology

The methodology employed in this project involves several key steps:

1. **Data Acquisition**: Satellite imagery data from the Sentinel-1 C-band SAR and Sentinel-2 MSI collections are used for the classification task. The data is filtered and preprocessed to select relevant bands and time periods.

2. **Feature Extraction**: Spectral indices such as Normalized Difference Vegetation Index (NDVI), Green Normalized Difference Vegetation Index (GNDVI), and Red Edge Hydrologic Break Index (REHBI) are computed from the satellite imagery to capture vegetation health and disease-related information.

3. **Supervised Classification**: The Random Forest algorithm is employed for supervised classification using SAR data, MSI data, and their combination. The classifier is trained on labeled data to differentiate between healthy and unhealthy grape crops.

4. **Unsupervised Classification**: Additionally, unsupervised classification techniques such as K-means clustering are applied to the extracted spectral indices to identify distinct clusters representing different levels of crop health.

5. **Evaluation and Export**: The accuracy of the classification models is evaluated using confusion matrices and resubstitution accuracy metrics. The resulting classified images are exported for further analysis and visualization.

## Usage

To replicate or extend the project:

1. Ensure access to relevant satellite imagery datasets, such as Sentinel-1 SAR and Sentinel-2 MSI data.
2. Set up the necessary environment for geospatial data processing, including tools like Google Earth Engine.
3. Implement the provided JavaScript code for data preprocessing, feature extraction, and classification.
4. Modify parameters and algorithms as needed to improve classification accuracy or adapt to different study areas.
5. Evaluate the performance of the classification models using appropriate metrics and visualizations.
6. Export the classified images for further analysis or integration into other applications.

## Acknowledgments
This project is based on the research and development efforts of "Shubham Kalokhe". 
Special thanks to the contributors to the Google Earth Engine platform and the providers of Sentinel satellite data for making this project possible.

