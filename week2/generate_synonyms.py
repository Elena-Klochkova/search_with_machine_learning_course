import fasttext
import csv


model = fasttext.load_model('/workspace/datasets/fasttext/title_model_200.bin')

with open('/workspace/datasets/fasttext/top_words.txt', 'r') as source:
    with open('/workspace/datasets/fasttext/synonyms.csv', 'w') as target:
        for word in source.readlines():
            word = word.replace('\n', '')
            predict = model.get_nearest_neighbors(word)
            synonyms = [rec[1] for rec in predict if rec[0] >= 0.8]
            if synonyms:
                res = [word] + synonyms
                writer = csv.writer(target)
                writer.writerow(res)
