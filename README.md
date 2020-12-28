# Bird classification problem

One day you notice a beautiful bird, and want to know what's the bird?
If you live in the North America, this app aims to resolve it for you.

## How to use the app
This app has two ways of usages:
### 1. Use it as an API service.
Submit the POST request and get top five candidates for the bird names with their probabilities in JSON format.
### 2. (Work in progress) Use our frontend interface

## Data
Data is a courtesy of Kaggle datasets

https://www.kaggle.com/gpiosenka/100-bird-species

Thanks for providing this dataset

## Methodology
The bird classification is done using the convolutional
neural network (CNN) using the transfer learning from ResNet101V2.
The training was performed on the Google Colab.
Please check ... for the CNN architecture.

## Future directions
1. Currently, the CNN works well on square images. However, modern phones
and cameras primarily take non-square images. I plan to add the tool for cropping
the image.

2. Insufficient data for many species. The dataset description says that 80% of samples
are males, and, therefore, it is possible that female birds are not classified correctly.
It will be fixed using the web scraping.

3. Add description to each bird name and sample photos.