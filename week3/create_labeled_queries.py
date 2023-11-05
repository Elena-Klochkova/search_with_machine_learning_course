import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'
queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

min_queries = -1
if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns=['category', 'parent'])
parents_df.to_csv('/workspace/datasets/parents.csv')

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]


# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
def normalize_query(query):
    new_query = ''
    for char in query:
        if char.isalnum():
            new_query += char.lower()
        else:
            new_query += ' '

    new_query = (new_query.split())
    res = [stemmer.stem(word) for word in new_query]
    
    return ' '.join(res)


queries_df['query'] = queries_df['query'].apply(normalize_query)
pd.options.display.width = 0

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
def rollup_to_parent(queries_df, parents_df, count_df):
    res = pd.merge(queries_df, parents_df, how="left", on=["category"])
    res = pd.merge(res, count_df, how="left", on=["category"])
    res['cat'] = res.apply(categorize, axis=1)
    res = res[['query', 'cat']]
    res.rename({'cat': 'category'}, axis=1, inplace=True)

    return res


def categorize(row):
    if row['cnt_queries'] < min_queries:
        return row['parent']
    
    return row['category']


def calc_count(queries_df):
    count_df = queries_df\
        .groupby('category', as_index=False)\
        .count()
    count_df.rename({'query': 'cnt_queries'}, axis=1, inplace=True)

    return count_df


if min_queries > 0:
    count_df = calc_count(queries_df)
    while True:
        if count_df[count_df['cnt_queries'] < min_queries].shape[0] > 0:
            queries_df = rollup_to_parent(queries_df, parents_df, count_df)
            count_df = calc_count(queries_df)
            print(count_df.sort_values(['cnt_queries', 'category']).head(10))
        else:
            break

print('TEST category:')
print(count_df[count_df['category'] == 'abcat0701001'])

print(f"Number of unique categories: {queries_df['category'].nunique()}")

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
