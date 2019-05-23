# DEVELOPMENT OF A FOOD IMAGE CLASSIFICATION SYSTEM BASED ON TRANSFER LEARNING WITH CONVOLUTIONAL NEURAL NETWORKS
by Alberto Sánchez López

----

## Links

#### · [Transfer Learning Model for Image Recognition](https://nbviewer.jupyter.org/github/sanxlop/tfm_etsit/blob/master/transfer_learning_model_for_image_recognition.ipynb)
#### · [Web Scraping System for Data Extraction](https://nbviewer.jupyter.org/github/sanxlop/tfm_etsit/blob/master/web_scraping_system_for_data_extraction.ipynb)
#### · [Foodiefy for Android](https://play.google.com/store/apps/details?id=com.phonegap.foodiefy)
#### · [Foodiefy Website](https://foodiefy.herokuapp.com)

----

## Description

The main goal of this project is developing an image classification system based on transfer learning with convolutional neural networks, focused on object recognition. Are employed several pre-trained architectures accessible from [Keras](https://keras.io/), a high-level API developed on [TensorFlow](https://www.tensorflow.org/) highlighted for deep learning research. System evaluation is carried out by a popular dataset of labeled food [ETHZ Food-101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) which measures the quality of each of the previous architectures.

The practical case of use carried out with this system is the prediction of the class and labeling of thousands of food images of restaurants mined from [TripAdvisor](https://www.tripadvisor.es/) with [Selenium WebDriver](https://www.seleniumhq.org/projects/webdriver/). From this new set of data, a restaurant search engine based on food images is implemented, that means, the search of restaurants is done by food photographs appearance. Everything mentioned above is developed in a web application to show all the objectives achieved.

----

## Results

Following table shows for each of the architecures the metrics of the model:

| Model | Loss | Accuracy | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: | :---: | :---: |
| MobileNetV2 | 0.749 | 0.817 | 0.880 | 0.880 | 0.880 |
| ResNet50 | 0.755 | 0.827 | 0.830 | 0.830 | 0.830 |
| InceptionResNetV2 | 0.527 | 0.887 | 0.890 | 0.890 | 0.890 |
| Xception | 0.551 | 0.885 | 0.880 | 0.880 | 0.880 |

----

## Files & Folders

1. __`transfer_learning_model_for_image_recognition.ipynb`__:
      > Firstly, model building process and configurations are explained step by step, specifically applied to the ETHZ Food-101 image dataset, although it could be extrapolated to any other image dataset. Secondly, the architecture and features of the pre-trained models used to transfer knowledge are presented to our classifier. Thirdly, is carried out with the experimental phase the analysis of the pre-trained models used for transfer learning.
Finally, the results obtained are analyzed and the optimal model for the project is chosen.
      
2. __`web_scraping_system_for_data_extraction.ipynb`__:
      > This notebook shows how to build a database by extracting data from web pages using web scraping techniques and applying a deep learning model for classifying. Based on the nature of the project, TripAdvisor web is selected. The main idea is to use TripAdvisor restaurants information and food images from reviews to model a dataset for our study case.
      
3. __`model-best-MobileNetV2-224x224-b64-e25.h5`__:
      > Light Keras model generated for mobile applications. It can be used directly to predict food images category with Keras.

4. __`/results`__:
      > Folder containing all training reports and graphics, and the data extracted from web scraping.
      
       
