#!/usr/bin/env python
# coding: utf-8

import re
import math

# # Section 1: Word Association Mining

# ## Basic Statistics of the Corpus
# First step was to just get a basic feel for the dataset. 
# Needed to count the total reviews and get a sense of the class balance.

review_filepath = 'amazon_reviews.txt' # NOTE: Using a relative path for portability

num_reviews = 0
num_positive_reviews = 0
num_negative_reviews = 0
with open(review_filepath) as f:
    f.readline() # Skipping the header line.
    for line in f:
        num_reviews += 1
        # The label is the first token in each line.
        if line.strip().split('\t')[0] == '1':
            num_positive_reviews += 1
        else:
            num_negative_reviews += 1
print('total number of reviews:', num_reviews)
print('total number of positive reviews:', num_positive_reviews)
print('total number of negative reviews:', num_negative_reviews)


# ## Count Frequency of Single Words
# Before any complex analysis, I needed a simple text processing function. 
# This strips punctuation and standardizes to lowercase. Pretty standard stuff.

def process_text(text):
    for punctuations in [',', '.', '"', '!', '?', ':', ';', '-', '(', ')', '[', ']']:
        text = text.replace(punctuations, ' ')
    text = re.sub('\s+', ' ', text)
    text = text.lower().strip()
    return text

# Now, to get the unigram frequencies. Just iterating through each review,
# processing the text, and then updating counts in a dictionary.

def get_single_word_frequency(filepath):
    word_freq = {}
    with open(filepath) as f:
        f.readline() 
        for line in f:
            review_text = process_text(line.split('\t')[1])
            for word in review_text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
    return word_freq

word_freq = get_single_word_frequency(review_filepath)
# Just printing the top 10 to see if they make sense.
for word, freq in sorted(word_freq.items(), key = lambda x: x[1], reverse = True)[:10]:
    print(word, freq)

total_num_words = sum(word_freq.values())
print('number of unique words:', len(word_freq))
print('total number of word occurrences:', total_num_words)


# ## Count Frequency of Co-occurring Word Pairs
# Here's where it gets more interesting. I needed to find pairs of words that appear
# together within a certain window. This is the first step for PMI.

def get_ordered_word_pair_frequency(filepath, window_size):
    pair_freq = {}
    with open(filepath) as f:
        f.readline()
        for line in f:
            word_list = process_text(line.split('\t')[1]).split()
            # This nested loop is a bit clunky, but it gets the job done.
            # It creates every possible ordered pair within the window.
            for i in range(len(word_list)):
                for j in range(i + 1, min(i + window_size, len(word_list))):
                    order_word_pair = (word_list[i], word_list[j])
                    pair_freq[order_word_pair] = pair_freq.get(order_word_pair, 0) + 1
    return pair_freq

TEXT_WINDOW_SIZE = 5
pair_freq = get_ordered_word_pair_frequency(review_filepath, TEXT_WINDOW_SIZE)
# Let's check the most common pairs. As expected, they're mostly stop words.
for pair, freq in sorted(pair_freq.items(), key = lambda x: x[1], reverse = True)[:10]:
    print(pair, freq)


# ## Calculate Pointwise Mutual Information (PMI)
# Okay, now for the actual PMI calculation. The goal was to find pairs that are
# genuinely associated, not just frequently occurring.

WORD_PAIR_FREQUENCY_THRESHOLD = 50
pmi_per_pair = {}
for pair, freq in pair_freq.items():
    if freq < WORD_PAIR_FREQUENCY_THRESHOLD: # Unwantedly infrequent pairs are filtered out.
        continue
    
    # Calculating the individual and joint probabilities was the next logical step.
    prob_word_pair = freq / total_num_words
    prob_word_0 = word_freq.get(pair[0], 0) / total_num_words
    prob_word_1 = word_freq.get(pair[1], 0) / total_num_words
    
    # The core of the PMI formula. Using logs to measure the ratio of joint vs. independent probabilities.
    if prob_word_0 > 0 and prob_word_1 > 0:
        pmi_per_pair[pair] = math.log(prob_word_pair / (prob_word_0 * prob_word_1))

# Now, sorting by PMI should give us much more interesting, meaningful associations.
top_100_pmi = sorted(pmi_per_pair.items(), key=lambda x: x[1], reverse=True)[:100]
for pair, pmi in top_100_pmi:
    print(f"Word-Pair: {pair}, PMI: {pmi}")


# # Section 2: Feature Selection using Chi-Square Statistic

# ## Count Document Frequencies per Sentiment
# For Chi-Square, the first thing is to build the components of the contingency table.
# So, for each word, I needed to know how many positive docs it appears in, and how many negative.

def get_single_word_doc_frequency_per_label(filepath, label):
    word_freq_per_label = {}
    with open(filepath) as f:
        f.readline()
        for line in f:
            line_parts = line.strip().split('\t')
            sentiment_label = line_parts[0]
            if sentiment_label == label:
                # Using set() here to only count presence once per review.
                review_words = set(process_text(line_parts[1]).split())
                for word in review_words:
                    word_freq_per_label[word] = word_freq_per_label.get(word, 0) + 1
    return word_freq_per_label

positive_word_freq = get_single_word_doc_frequency_per_label(review_filepath, '1')
print("Top words in positive reviews:")
for word, freq in sorted(positive_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(word, freq)

negative_word_freq = get_single_word_doc_frequency_per_label(review_filepath, '0')
print("\nTop words in negative reviews:")
for word, freq in sorted(negative_word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(word, freq)


# ## Calculate Chi-Square Statistic
# With all the counts ready, I can now build the contingency table for each word and calculate the statistic.
# The table values are: a = present&positive, b = present&negative, c = absent&positive, d = absent&negative.

chi2_per_word = {}
for word, freq in word_freq.items():
    if freq < 10: # Filtering out very rare words again.
        continue

    # Observed Frequencies
    a = positive_word_freq.get(word, 0)
    b = negative_word_freq.get(word, 0)
    c = num_positive_reviews - a
    d = num_negative_reviews - b
    
    # Expected Frequencies, figuring this out took a moment.
    # It's based on what we'd expect if the word's presence and sentiment were totally independent.
    E11 = (a + b) * (a + c) / num_reviews
    E12 = (a + b) * (b + d) / num_reviews
    E21 = (c + d) * (a + c) / num_reviews
    E22 = (c + d) * (b + d) / num_reviews
    
    # The Chi-Square formula: sum of (Observed - Expected)^2 / Expected
    chi_square = 0
    if E11 > 0: chi_square += ((a - E11)**2 / E11)
    if E12 > 0: chi_square += ((b - E12)**2 / E12)
    if E21 > 0: chi_square += ((c - E21)**2 / E21)
    if E22 > 0: chi_square += ((d - E22)**2 / E22)
    
    chi2_per_word[word] = chi_square

# Sorting by Chi-Square value reveals the words most statistically tied to sentiment.
top_100_chi2 = sorted(chi2_per_word.items(), key=lambda x: x[1], reverse=True)[:100]
for word, chi2 in top_100_chi2:
    print(f"Word: {word}, Chi-Square: {chi2}")
