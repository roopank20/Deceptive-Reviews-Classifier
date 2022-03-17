# Deceptive-Reviews-Classifier

Here, we were provided with the following :

* train.txt - a training dataset containing user generated reviews of multiple hotels present in the city of Chicago. 
It contains both fake and legitimate reviews and its purpose is to train our classifier.

* test.txt - the testing dataset containing user generated reviews of multiple hotels present in the city of Chicago.
It contains both fake and legitimate reviews and it is used to test/evaluate the genuineness of a hotel review. Each 
review is fed to the classifier and it determines the legitimacy of the review.

* SeekTruth.py - it contained the skeleton code where we are supposed to publish our solution.

## Problem Statement

To write a program that estimates the Naive Bayes parameters from training data (where the correct label is given), and 
then uses these parameters to classify the reviews in the testing data. The task is to classify reviews into faked or 
legitimate, for 20 hotels in Chicago.

## Solution

Naive Bayes is a family of probabilistic algorithms that take advantage of probability theory and Bayes’ Theorem to 
predict the tag of a text (like a customer review). They are probabilistic, which means that they calculate the 
probability of each tag for a given text, and then output the tag with the highest one. The way they get these 
probabilities is by using Bayes’ Theorem, which describes the probability of a feature, based on prior knowledge of 
conditions that might be related to that feature. We've tried to implement Multinomial Naive Bayes here. The multinomial
naive Bayes model is typically used for discrete counts. Below we have tried to explain the implementation of our 
solution in a hierarchical form.

### Data Cleaning

The raw data provided to us contains some anomalies and redundancies that may hamper the computation of our classifier.
The raw data, a sequence of symbols (i.e. strings) cannot be fed directly to the algorithms themselves as most of them 
do not contribute towards the classification of reviews. In order to clean/filter our data, certain operations were 
applied over the data so that it produces optimal result. These operations include removing stopwords, converting every
single word to lowercase, removing punctuation marks, removing non ascii characters, and applying regex to surpass the 
meaningful words. This process was applied over both training and testing dataset.

###  Frequency Calculation

As per the Naive Bayes theorem, our efforts should go to calculating the frequency of every word rather then looking at
the individual sentences. This means that the frequency of a word is calculated in both fake and legitimate part of the 
dataset, and then based on the frequency, we calculate the probability of that word in that category.

### Applying Conditional Probability using Naive Bayes 

The final step is just to calculate every probability and see which one turns out to be larger. Calculating a 
probability is just counting in our training data. First, we calculate the a priori probability of each word: 
for a given sentence in our training data, Then, calculating how many times the word categorical texts divided by the 
total number of words in that category.

### Using Laplace Smoothening

At times, we might encounter one or more than one word in our testing text, which does not appear in our training data. 
This can cause a problem as it will make our probability equal to zero. This problem can be handled using Laplace 
Smoothening. It is nothing but a technique where a small-sample correction, or pseudo-count, will be incorporated in 
every probability estimate. 

## Design Decisions

A number of data structures including list, set, dictionary etc. and a variety of inbuilt functions like 
math.log, str.maketrans were used in this solution. Along with that, regular expressions were used for filtering
out the unnecessary words. The main approach to decide the designing decisions for this problem was to implement 
the solution in minimum time complexity. Log of probabilities was taken and added in place of multiplying so that 
heavy computation is avoided and to ensure that very small probabilistic values are not ignored while calculating
the final conditional probability. The output here displays the accuracy attained by our classifier on this specific
dataset which comes out to be around 80-85%. 

## Assumpltions

It was assumed that every word in a sentence is independent of the other ones. This is done so that we can concentrate 
on individual words rather than sentences.

## Challenges Faced

Problems were faced for cleaning the data as a number of irrelevant and grammatically incorrect words were present in 
the database. Other than that, calculating probability of those words, which were not there in training data, was another
challenge, but it was appropriately handled using Laplace Smoothening. Lastly, computing probability of those words 
which had a small frequency was a problem, it was solved by taking the log of probabilities and adding them.
