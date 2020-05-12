# Udacity Machine Learning with Tensforflow Nanodegree



![img](images/ML%20Nanodegree.png)



# Project 1: Finding Donors

`RandomForest`, `AdaBoost`, `Gradient Boosting`, `Support Vector Machine`, `Gaussian Naive Bayes`

## Description

CharityML is a fictitious charity organization located in the heart of Silicon Valley that was established to provide financial support for people eager to learn machine learning. After nearly 32,000 letters were sent to people in the community, CharityML determined that every donation they received came from someone that was making more than $50,000 annually. To expand their potential donor base, CharityML has decided to send letters to residents of California, but to only those most likely to donate to the charity. With nearly *15 million* working Californians, CharityML has brought you on board to help build an algorithm to best identify potential donors and reduce overhead cost of sending mail. Your goal will be evaluate and optimize several different supervised learners to determine which algorithm will provide the highest donation yield while also reducing the total number of letters being sent. 

## Software and Libraries

This project uses the following software and Python libraries:

- [Python](https://www.python.org/downloads/release/python-364/)
- [NumPy](http://www.numpy.org/)
- [pandas](http://pandas.pydata.org/)
- [scikit-learn](http://scikit-learn.org/0.17/install.html) (v0.17)
- [Matplotlib](http://matplotlib.org/)



# Project 2 Image Classifier

`Data Preprocessing`, `Deep Learning`, `Neural Networks`, `Tensforflow`, `Keras`, `Model Deployment`

## Part 1: Developing an Image Classifier with Deep Learning

In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with TensorFlow



## Part 2 - Building the Command Line Application

Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. Your application should be a Python script that run from the command line. For testing, you should use the saved Keras model you saved in the first part.

### Specifications

The project submission must include a `predict.py` file that uses a trained network to predict the class for an input image. Feel free to create as many other files as you need. Our suggestion is to create a module just for utility functions like preprocessing images. **Make sure to include all files necessary to run the `predict.py` file in your submission.**

The `predict.py` module should predict the top flower names from an image along with their corresponding probabilities. 

**Basic usage:**

```python
$ python predict.py /path/to/image saved_model
```

**Options:**

- `--top_k` : Return the top K*K* most likely classes:

```python
$ python predict.py /path/to/image saved_model --top_k KK
```

- `--category_names` : Path to a JSON file mapping labels to flower names:

```python
$ python predict.py /path/to/image saved_model --category_names map.json
```

The best way to get the command line input into the scripts is with the [argparse module](https://docs.python.org/3/library/argparse.html) in the standard library. You can also find [a nice tutorial for argparse here](https://pymotw.com/3/argparse/). 

### Examples

For the following examples, we assume we have a file called `orchid.jpg` in a folder named`/test_images/` that contains the image of a flower. We also assume that we have a Keras model saved in a file named `my_model.h5`.

**Basic usage:**

```python
$ python predict.py ./test_images/orchid.jpg my_model.h5
```

**Options:**

- Return the top 3 most likely classes:

```python
$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3
```

- Use a `label_map.json` file to map labels to flower names:

```python
$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json
```



# Project 3: Identify Customer Segments

`Data preprocessing`, `Feature Selection`, `Feature Engineering`, `Principle Component Analysis (PCA)`, `KMeans Clustering`

In this project, our Bertelsmann partners AZ Direct and Arvato Financial Solutions have provided two datasets one with demographic information about the people of Germany, and one with that same information for customers of a mail-order sales company. You’ll look at relationships between demographics features, organize the population into clusters, and see how prevalent customers are in each of the segments obtained.

# Project Overview

In this project, you will work with real-life data provided to us by our Bertelsmann partners AZ Direct and Arvato Finance Solution. The data here concerns a company that performs mail-order sales in Germany. Their main question of interest is to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign. Your job as a data scientist will be to use unsupervised learning techniques to organize the general population into clusters, then use those clusters to see which of them comprise the main user base for the company. Prior to applying the machine learning methods, you will also need to assess and clean the data in order to convert the data into a usable form.

## What will I need to install?

It is highly recommended that you use the project workspace to complete and submit this project. Should you decide to download the data, you will need to remove it from your computer via the agreement with Arvato Bertlesmann. If you choose to use your machine to complete this project, below are a list of the requirements.

This project uses Python 3 and is designed to be completed through the Jupyter Notebooks IDE. It is highly recommended that you use the Anaconda distribution to install Python, since the distribution includes all necessary Python libraries as well as Jupyter Notebooks. The following libraries are expected to be used in this project:

- **NumPy**
- **pandas**
- **Sklearn** / scikit-learn
- **Matplotlib** (for data visualization)
- **Seaborn** (for data visualization)

## Why this project?

The unsupervised learning branch of machine learning is key in the organization of large and complex datasets. While unsupervised learning lies in contrast to supervised learning in the fact that unsupervised learning lacks objective output classes or values, it can still be important in converting the data into a form that can be used in a supervised learning task. Dimensionality reduction techniques can help surface the main signals and associations in your data, providing supervised learning techniques a more focused set of features upon which to apply their work. Clustering techniques are useful for understanding how the data points themselves are organized. These clusters might themselves be a useful feature in a directed supervised learning task. This project will give you hands-on experience with a real-life task that makes use of these techniques, focusing on the unsupervised work that goes into understanding a dataset.

In addition, the dataset presented in this project requires a number of assessment and cleaning steps before you can apply your machine learning methods. In workplace contexts, you will frequently need to work with data that is untidy or needs preprocessing before standard algorithms and models can be applied. The ability to perform data wrangling and the ability to make decisions on data that you work with are both valuable skills that you will practice in this project.

## Project Details

Below, you will find an outline of the steps that you will take in this project. It is recommended that you read through all of these instructions before starting on the project so you can build a high-level understanding of the tasks ahead!

### Step 0: The Data

There are a number of files used for this project available in the workspace on the next page. **It is strongly recommended that you complete and turn in this project from the workspace.** As you will see from the agreement on the next page, if you choose to download a .zip file you will need to **REMOVE** the files from your computer via the agreement with Arvato Bartlesmann after completing the project, as the data are proprietary. The files available for the project are:

- `Udacity_AZDIAS_Subset.csv`: Demographic data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
- `Udacity_CUSTOMERS_Subset.csv`: Demographic data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
- `Data_Dictionary.md`: Information file about the features in the provided datasets.
- `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographic data.
- `Identify_Customer_Segments.ipynb`: Jupyter Notebook divided into sections and guidelines for completing the project. The notebook provides more details and tips than the outline given here.

After downloading the data and unzipping it to a working folder of your choice, use the Terminal or Command Prompt to navigate to your selected folder and execute the command `jupyter notebook Identify_Customer_Segments.ipynb` to open up a browser window or tab to work on the project notebook.

### Step 1: Preprocessing

When you start an analysis, you must first explore and understand the data that you are working with. In this (and the next) step of the project, you’ll be working with the general demographics data. As part of your investigation of dataset properties, you must attend to a few key points:

- How are missing or unknown values encoded in the data? Are there certain features (columns) that should be removed from the analysis because of missing data? Are there certain data points (rows) that should be treated separately from the rest?
- Consider the level of measurement for each feature in the dataset (e.g. categorical, ordinal, numeric). What assumptions must be made in order to use each feature in the final analysis? Are there features that need to be re-encoded before they can be used? Are there additional features that can be dropped at this stage?

You will create a cleaning procedure that you will apply first to the general demographic data, then later to the customers data.

### Step 2: Feature Transformation

Now that your data is clean, you will use dimensionality reduction techniques to identify relationships between variables in the dataset, resulting in the creation of a new set of variables that account for those correlations. In this stage of the project, you will attend to the following points:

- The first technique that you should perform on your data is feature scaling. What might happen if we don’t perform feature scaling before applying later techniques you’ll be using?
- Once you’ve scaled your features, you can then apply principal component analysis (PCA) to find the vectors of maximal variability. How much variability in the data does each principal component capture? Can you interpret associations between original features in your dataset based on the weights given on the strongest components? How many components will you keep as part of the dimensionality reduction process?

You will use the **sklearn** library to create objects that implement your feature scaling and PCA dimensionality reduction decisions.

### Step 3: Clustering

Finally, on your transformed data, you will apply clustering techniques to identify groups in the general demographic data. You will then apply the same clustering model to the customers dataset to see how market segments differ between the general population and the mail-order sales company. You will tackle the following points in this stage:

- Use the k-means method to cluster the demographic data into groups. How should you make a decision on how many clusters to use?
- Apply the techniques and models that you fit on the demographic data to the customers data: data cleaning, feature scaling, PCA, and k-means clustering. Compare the distribution of people by cluster for the customer data to that of the general population. Can you say anything about which types of people are likely consumers for the mail-order sales company?

**sklearn** will continue to be used in this part of the project, to perform your k-means clustering. In the end, you will export the completed notebook with your work as an HTML file, which will serve as a report documenting your approach and findings.

### Step 4: Review and Submit the Project

Before you submit your project for review, you should first review the [project rubric](http://review.udacity.com/#!/projects/f5062c28-1399-403e-af18-0e5b0e4587fd/rubric). Your project will be evaluated by an Udacity reviewer using this criteria, so you should make a self-evaluation as well. In order for your project to pass review, it must meet all specifications.

Once you are ready for project submission, export an HTML version of your notebook and add to your workspace submission, then submit your entire project from the workspace using in the button in the bottom right.

