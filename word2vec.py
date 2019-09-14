import gzip
import gensim
import logging
import os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read_input(input_file):
    with gzip.open(input_file, 'rb') as f:
        for line in f:
            yield gensim.utils.simple_preprocess(line)

if __name__ == '__main__':

    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "../reviews_data.txt.gz")

    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    model = gensim.models.Word2Vec(documents,size=150,window=10,min_count=2,workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)

    model.wv.save(os.path.join(abspath, "../vectors/default"))

    w1 = "dirty"
    print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1))

    # look up top 6 words similar to 'polite'
    w1 = ["polite"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'france'
    w1 = ["france"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'shocked'
    w1 = ["shocked"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'shocked'
    w1 = ["beautiful"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # get everything related to stuff on the bed
    w1 = ["bed", 'sheet', 'pillow']
    w2 = ['couch']
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            negative=w2,
            topn=10))

    # similarity between two different words
    print("Similarity between 'dirty' and 'smelly'",
          model.wv.similarity(w1="dirty", w2="smelly"))

    # similarity between two identical words
    print("Similarity between 'dirty' and 'dirty'",
          model.wv.similarity(w1="dirty", w2="dirty"))

    # similarity between two unrelated words
    print("Similarity between 'dirty' and 'clean'",
          model.wv.similarity(w1="dirty", w2="clean"))