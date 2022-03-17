# SeekTruth.py : Classify text objects into two categories
#
# Code by: Sri Varsha Chellapilla (srchell), Roopank Kohli (rookohli) and Akash Bhapkar (abhapkar)
#
# Based on skeleton code by D. Crandall, October 2021

# We have taken the logic to solve this problem from below given URL.
# https://monkeylearn.com/blog/practical-explanation-naive-bayes-classifier/

# Help was taken from the following resources -

# To understand the practical usage and importance of Laplace Smoothening -
# https://www.analyticsvidhya.com/blog/2021/04/improve-naive-bayes-text-classifier-using-laplace-smoothing/

# To understand the utility of Stop Words
# https://www.opinosis-analytics.com/knowledge-base/stop-words-explained/#.YYXCImBKhPY

# Understanding Naive Bayer Threorem
# https://towardsdatascience.com/na%C3%AFve-bayes-spam-filter-from-scratch-12970ad3dae7

# To formulate the regex
# https://docs.python.org/3/library/re.html

import sys
import re
import string
import math

# English Stopword list referred from
# https://github.com/igorbrigadir/stopwords


stopword = ['off', 'herself', 'most', 're', 'how', 'himself', 'i', 'those', 'not', 'all', 'then', "couldn't", 'into',
            'nor', 'after', 'is', 'when', 'so', 'if', 'shouldn', 'has', 'because', 'of', 'their', 'aren', 'before',
            'doesn', 'my', 'mustn', 'than', 'do', 'as', 'are', "wouldn't", 'under', 'o', 'ourselves', 'what', 'up',
            'hasn', 'its', 'does', 'other', 'didn', 'hadn', "you'll", 'again', "weren't", 'wouldn', 't', 'against',
            'any', "hadn't", 'through', 'below', 'mightn', 'his', 'more', 'own', "won't", 'have', 's', 'doing', 'a',
            'theirs', 'can', 'needn', 'ours', 'll', 'above', "it's", 'at', 'once', 'why', 'with', 'they', 'same', 'by',
            've', 'too', 'been', 'from', 'only', "haven't", 'during', 'this', "she's", 'and', 'yours', "needn't", 'was',
            'wasn', 'won', 'should', 'which', 'that', 'both', 'who', 'be', 'm', 'yourselves', 'whom', 'he', 'some',
            'haven', "isn't", 'out', 'yourself', "you've", "shouldn't", 'an', 'in', 'about', 'just', 'very', "didn't",
            'shan', 'we', 'ain', "hasn't", 'isn', "shan't", 'to', 'you', 'me', 'each', 'couldn', 'it', 'them', "you're",
            'myself', 'these', 'were', 'him', 'on', 'now', 'y', "doesn't", "that'll", 'weren', "mustn't", 'where',
            'the', 'until', 'her', 'over', 'itself', 'don', 'there', 'further', 'she', 'being', "don't", 'between',
            'having', 'down', 'our', 'such', 'here', "wasn't", 'but', "should've", 'did', 'd', 'hers', 'your', 'am',
            'few', 'will', 'ma', 'no', 'themselves', 'or', 'had', "you'd", 'while', "aren't", "mightn't", 'for']


# Method to load or read the given text file (test and train dataset). It organises the dataset into a dictionary/map.

def load_file(filename):
    objects = []
    labels = []
    with open(filename, "r") as f:
        for line in f:
            parsed = line.strip().split(' ', 1)
            labels.append(parsed[0] if len(parsed) > 0 else "")
            objects.append(parsed[1] if len(parsed) > 1 else "")

    return {"objects": objects, "labels": labels, "classes": list(set(labels))}


# Method to clean, and filter the training dataset. Firstly, it removes the non ascii characters from the dataset, if
# any. After that it converts the words into lowercase form, so that similar words are grouped together. Stop Words are
# used to remove the non useful and redundant small words which do not contribute enough towards the probability
# calculated by the classifier. Furthermore, all the punctuation marks are deleted from the respective words in the
# reviews.

def cleanTrainingData(Data):
    fileContent = Data["objects"]
    deceptiveList = []
    truthfulList = []
    wordDict = {}
    counter = 0
    allWordList = []
    for line in fileContent:
        contents = line.encode("ascii", "ignore").decode().split(" ")
        key = Data["labels"][counter]
        counter += 1
        for word in contents:
            word = word.lower()

            if word not in stopword:
                temp = (word[-2:])

                if len(word) > 2 and (temp == "'s" or temp == "'S"):
                    word = word[0:-2]

                translator = str.maketrans('', '', string.punctuation)
                word = word.translate(translator)

                if word in [re.sub(r'[^a-z]', r'', str(word))] and word not in "\n":
                    allWordList.append(word)
                    if key in "deceptive":
                        deceptiveList.append(word)
                    else:
                        truthfulList.append(word)

        wordDict["deceptive"] = deceptiveList
        wordDict["truthful"] = truthfulList

    deceptiveSet = set(deceptiveList)
    truthfulSet = set(truthfulList)

    return wordDict, allWordList, deceptiveSet, truthfulSet


# Method to clean, and filter the test dataset. Firstly, it removes the non ascii characters from the dataset, if any.
# After that it converts the words into lowercase form, so that similar words are grouped together. Stop Words are used
# to remove the non useful and redundant small words which do not contribute enough towards the probability calculated
# by the classifier. Furthermore, all the punctuation marks are deleted from the respective words in the reviews.

def cleanTestData(Data):
    dataList = []
    fileContent = Data["objects"]

    for line in fileContent:
        contents = line.encode("ascii", "ignore").decode().split(" ")
        sen = ""
        wordList = []
        for word in contents:

            word = word.lower()

            if word not in stopword:

                temp = (word[-2:])

                if len(word) > 2 and (temp == "'s" or temp == "'S"):
                    word = word[0:-2]

                translator = str.maketrans('', '', string.punctuation)

                word = word.translate(translator)

                if word in [re.sub(r'[^a-z]', r'', str(word))] and word not in stopword and word not in "\n":
                    if word not in wordList:
                        sen += word + " "
                        wordList.append(word)

        dataList.append(sen)

    return dataList


# After cleaning the dataset, frequency of each and every word present in our dataset is computed in order to calculate
# the condition probability ( using Multinomial Naive Bayes Theorem ) of every word. This is done to  classify the
# category of each hotel review.

def calculateWordFreq(Data):
    DfreqDict = {}
    TfreqDict = {}
    for key, value in Data.items():
        if key == "deceptive":
            for word in value:

                if word in DfreqDict:
                    val = DfreqDict[word] + 1
                    DfreqDict[word] = val
                else:
                    DfreqDict[word] = 1
        else:
            for word in value:
                if word in TfreqDict:
                    val = TfreqDict[word] + 1
                    TfreqDict[word] = val
                else:
                    TfreqDict[word] = 1

    return DfreqDict, TfreqDict


# Method to determine the probability of every single word. It returns a dictionary where each word of our clean dataset
# acts as the key and its overall probability in that category ( Deceptive / Truthful ) is stored as the value.

def calculateWordProb(FreqDict, total_length):
    ProbDict = {}
    for key, value in FreqDict.items():
        ProbDict[key] = ((value / total_length))

    return ProbDict


# Method to check if laplace smoothening is to be done or not for a particular hotel review. if the value of dflag or
# tflag is 1, then laplace smoothening is applied to that string of words / review .

def checkLaplace(revList, dSet, tSet):
    dflag = 0
    tflag = 0

    for word in revList:
        if word not in dSet:
            dflag = 1
            break

    for word in revList:
        if word not in tSet:
            tflag = 1
            break

    return dflag, tflag


# Method to compute the total conditional Probability using Multinomial Naive Bayes Theorem. It first evaluates if
# laplace smoothening is required or not in order to compute the probability.

def computeProbability(test_data_clean, DeceptiveProbDict, TruthfulProbDict, count, dSet, tSet, DeceptiveFreqDict,
                       TruthfulFreqDict, dl, tl):
    resultList = []
    c = 0
    alpha = 0.4
    for review in test_data_clean:
        finalProduct_D = 1
        finalProduct_T = 1
        revList = review.split(" ")
        c += 1

        Dflag, Tflag = checkLaplace(revList, dSet, tSet)

        for word in revList:

            if word in DeceptiveProbDict and Dflag == 0:
                val = DeceptiveProbDict[word]
                finalProduct_D += val

            else:
                num = 0
                if word in DeceptiveFreqDict:
                    num = DeceptiveFreqDict[word]

                val = math.log((alpha + num) / (dl + (alpha * count)))
                finalProduct_D += val

            if word in TruthfulProbDict and Tflag == 0:
                val = TruthfulProbDict[word]
                finalProduct_T += val

            else:

                num = 0
                if word in TruthfulFreqDict:
                    num = TruthfulFreqDict[word]

                val = math.log((alpha + num) / (tl + (alpha * count)))
                finalProduct_T += val

        finalProduct_T *= 0.5
        finalProduct_D *= 0.5

        if finalProduct_D <= finalProduct_T:
            resultList.append("truthful")
        else:
            resultList.append("deceptive")

    return resultList


# classifier : Train and apply a bayes net classifier
#
# This function should take a train_data dictionary that has three entries:
#        train_data["objects"] is a list of strings corresponding to reviews
#        train_data["labels"] is a list of strings corresponding to ground truth labels for each review
#        train_data["classes"] is the list of possible class names (always two)
#
# and a test_data dictionary that has objects and classes entries in the same format as above. It
# should return a list of the same length as test_data["objects"], where the i-th element of the result
# list is the estimated classlabel for test_data["objects"][i]
#
# Do not change the return type or parameters of this function!

def classifier(train_data, test_data):
    # This is just dummy code -- put yours here!

    train_data_clean, allWordSet, deceptiveUniqueWordSet, truthfulUniqueWordSet = cleanTrainingData(train_data)

    DeceptiveFrequencyDict, TruthfulFrequencyDict = calculateWordFreq(train_data_clean)

    DLength = len(train_data_clean["deceptive"])
    TLength = len(train_data_clean["truthful"])

    total_word_count = len(allWordSet)

    DeceptiveProbDict = calculateWordProb(DeceptiveFrequencyDict, DLength)
    TruthfulProbDict = calculateWordProb(TruthfulFrequencyDict, TLength)

    test_data_clean = cleanTestData(test_data)

    results = computeProbability(test_data_clean, DeceptiveProbDict, TruthfulProbDict, total_word_count,
                                 deceptiveUniqueWordSet, truthfulUniqueWordSet, DeceptiveFrequencyDict,
                                 TruthfulFrequencyDict, DLength, TLength)

    # return [test_data["classes"][1]] * len(test_data["objects"])
    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Usage: classify.py train_file.txt test_file.txt")

    (_, train_file, test_file) = sys.argv
    # Load in the training and test datasets. The file format is simple: one object
    # per line, the first word one the line is the label.
    train_data = load_file(train_file)
    test_data = load_file(test_file)
    if (sorted(train_data["classes"]) != sorted(test_data["classes"]) or len(test_data["classes"]) != 2):
        raise Exception("Number of classes should be 2, and must be the same in test and training data")

    # make a copy of the test data without the correct labels, so the classifier can't cheat!
    test_data_sanitized = {"objects": test_data["objects"], "classes": test_data["classes"]}

    results = classifier(train_data, test_data_sanitized)

    correct_ct = sum([(results[i] == test_data["labels"][i]) for i in range(0, len(test_data["labels"]))])
    print("Classification accuracy = %5.2f%%" % (100.0 * correct_ct / len(test_data["labels"])))
