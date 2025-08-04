#!/usr/bin/env python
# coding: utf-8

import editdistance

# # Section 3: Spell Correction using Letter N-grams
# The third part of the project was to build a spell corrector. 
# The main idea was to compare a fast method (Jaccard Similarity on n-grams) 
# with a slow but accurate one (Edit Distance).

# First, I need to load the dictionary of correctly spelled words.
a_list_filepath = 'enwiktionary.a.list' # NOTE: Using a relative path
a_list = []
with open(a_list_filepath) as f:
    for line in f:
        a_list.append(line.strip())

print('number of words/phrases in the dictionary:', len(a_list))

# This helper function breaks a word into a set of its n-grams.
# Pretty straightforward, just a loop that creates substrings of length n.
def chunk_word_into_letter_ngrams(word, n):
    ngrams = []
    for i in range(len(word) - n + 1):
        ngrams.append(word[i:i+n])
    return set(ngrams)

# I needed functions for both similarity metrics to compare them.
# Jaccard is just the size of the intersection over the size of the union. Simple.
def jaccard_similarity(set1, set2):
    if not isinstance(set1, set) or not isinstance(set2, set):
        return 0.0
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# For Edit Distance, using an external library was much easier than implementing it myself.
def calculate_edit_distance(word1, word2):
    return editdistance.eval(word1, word2)

# This function finds the top 10 dictionary words based on Jaccard similarity.
# It iterates through the whole dictionary, which can be slow, but it works.
def get_top_jaccard_similarity(input_str, dictionary, n):
    input_ngrams = chunk_word_into_letter_ngrams(input_str, n)
    similarity = []
    for word in dictionary:
        word_ngrams = chunk_word_into_letter_ngrams(word, n)
        sim = jaccard_similarity(input_ngrams, word_ngrams)
        similarity.append((word, sim))
    return sorted(similarity, key=lambda x: x[1], reverse=True)[:10]

# And this one does the same for edit distance, but sorts by smallest distance.
def get_top_edit_distance(input_str, dictionary):
    distance = []
    for word in dictionary:
        dist = calculate_edit_distance(input_str, word)
        distance.append((word, dist))
    return sorted(distance, key=lambda x: x[1])[:10]

# Here are the test strings I had to evaluate.
given_strings = ['abreviation', 'abstrictiveness', 'accanthopterigious', 'artifitial inteligwnse', 'agglumetation']

# Now, running the actual experiment.
# The goal is to see which n-gram length gives results closest to the Edit Distance baseline.
for a_string in given_strings:
    print(f"\n--- Results for '{a_string}' ---")
    
    print("\n**Jaccard Similarity Results:**")
    for n in [2, 3, 4, 5]:
        print(f"\nTop 10 for n-gram size={n}:")
        top_jaccard = get_top_jaccard_similarity(a_string, a_list, n)
        for word, similarity in top_jaccard:
            print(f"  {word}: {similarity:.4f}")
            
    print("\n**Edit Distance Baseline (for comparison):**")
    top_edit = get_top_edit_distance(a_string, a_list)
    print("Top 10:")
    for word, dist in top_edit:
        print(f"  {word}: (distance: {dist})")
