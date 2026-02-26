# BRANCHING STRAT TO AVOID MERGE CONFLICTS

-Put anything you find here.....

# Coursework Brief: House Price Prediction Using Neural Networks

## Coursework Type

Group Project

## Weighting

e.g. 30% of the module mark

## Group Size

5--6 students per group

## Submission Format

Source code (Python) and a written report (PDF, 3--5 pages)

## 1. Introduction

House prices are influenced by both the characteristics of individual
houses and their surrounding neighbourhoods. This coursework introduces
a supervised learning regression problem using artificial neural
networks. Students will also explore how neighbourhood relationships,
represented in a simple graph structure, can be used to improve
prediction performance. In this coursework, you will use the California Housing Prices
dataset, which is a publicly available dataset provided via scikit-learn.

## 2. Learning Objectives

-   Implement a feedforward neural network for regression
-   Use non-linear activation functions
-   Apply neural networks to a supervised learning problem
-   Understand how graph-based relationships can be used to extract useful
    input features
-   Evaluate and analyse model performance

## 3. Problem Description

You are given a dataset. Each data point represents a housing area
(census block group) in California and contains aggregated housing and
population statistics, including median income, average number of rooms,
average number of bedrooms, population-related statistics, and
geographical coordinates (latitude and longitude). Your task is to
predict housing prices for each housing area using a neural network.

## 4. Neighbourhood Relationships (Graph Representation)

Each housing area is treated as a node. An edge exists between two
housing areas if they are geographically close to each other. The graph
is used only to identify neighbouring housing areas. Neighbourhood
relationships may be defined using either a distance threshold (e.g.
within r km) or k-nearest neighbours. You are not required to implement
graph algorithms or graph neural networks; the graph is used solely to
define neighbourhood relationships for feature construction.

**Important:** You are NOT required to implement Graph Neural Networks
(GNNs). The graph is used only to define neighbourhood relationships.
You should compute simple numerical statistics from neighbouring nodes (e.g. average
neighbour price) and include them as additional input features to a standard feedforward neural network.

## 5. Feature Construction

You may compute neighbourhood-based features such as average neighbour
price (computed using training data only) or number of neighbours and
include them as inputs to your neural network. To avoid data leakage,
neighbourhood-based features must be constructed using training data
only. When computing statistics such as average neighbour price, you
must ensure that test labels are not used.

## 6. Task Requirements

-   Regression task using mean squared error loss
-   Feedforward neural network
-   At least one non-linear activation function (e.g. ReLU)
-   Train/test split

### Recommended Workflow

1.  Load and explore the dataset
2.  Preprocess data (normalisation and train/test split)
3.  Train a baseline neural network using original features
4.  Construct neighbourhood-based features
5.  Train a second model including neighbourhood features
6.  Compare test performance and analyse results

## 7. Report Requirements

Your report (3--5 pages) should include:

-   **Problem understanding:** Description of the task and dataset
-   **Methodology:** Feature selection and preprocessing; neural network
    architecture; training procedure
-   **Results:** Test set performance; compare the performance of a
    baseline model (without neighbourhood features) and an enhanced
    model (with neighbourhood features)
-   **Discussion:** Interpretation of results; limitations of the
    approach; possible improvements

## 8. Assessment Criteria

-   Data & Problem Understanding (20%)
-   Neural Network Implementation (30%)
-   Graph-based Features (20%)
-   Results & Analysis (20%)
-   Code & Report Quality (10%)

## Assessment Rubric

### 1. Data & Problem Understanding (20%)

-   Clear definition of the regression task (5%)
-   Clear description of input features and output variable (5%)
-   Appropriate data preprocessing, e.g. normalisation (5%)
-   Proper train/test split (5%)

### 2. Neural Network Implementation (30%)

-   Correct use of a feedforward neural network (10%)
-   Use of non-linear activation functions (10%)
-   Appropriate loss function for regression (10%)

### 3. Graph-based Feature Usage (20%)

-   Clear definition of neighbourhood relationships (10%)
-   Use of at least one neighbourhood-based feature (10%) (e.g. number
    of neighbours, mean distance to neighbours, or neighbourhood
    statistics used as input features)

### 4. Results & Analysis (20%)

-   Reporting test set performance (5%)
-   Interpretation of results (5%)
-   Discussion of limitations (5%)
-   Use of simple visualisations where appropriate (5%)

### 5. Code Quality & Report Clarity (10%)

-   Code is well-structured and readable (5%)
-   Report is clear, well-organised, and concise (5%)
