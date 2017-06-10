import os
import re
import json
import jieba
import jieba.analyse
from gensim.models import word2vec
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

stopwords = set()
books_dict = {}
corpus = []
count_vectorizer = CountVectorizer()
tfidf_transformer = TfidfTransformer()

def set_jieba():
    traditional_chinese_dict_path = '../data/dict.txt.big'
    jieba.set_dictionary(traditional_chinese_dict_path)

def set_stopwords():
    stopwords_path = '../data/stop_words.txt'

    with open(stopwords_path, 'r', encoding='utf-8') as f:
        for line in f:
            stopwords.add(line.strip('\n'))

    stopwords.add('\n')
    stopwords.add(' ')
    return stopwords

def load_books():
    library_books_path = '../data/books_list.json'
    eslite_books_path = '../data/誠品書籍資料.json'
    
    with open(library_books_path) as f:
        library_books = json.load(f)

    with open(eslite_books_path) as f:
        eslite_books = json.load(f)

    return library_books, eslite_books

def build_books_dict():
    library_books, eslite_books = load_books()

    for book in library_books:
        title = book['book name']
        num = book['book number']

        books_dict[title] = {}
        books_dict[title]['name'] = title
        books_dict[title]['num'] = num
        books_dict[title]['abstract'] = ''
        books_dict[title]['publish'] = ''
        books_dict[title]['author'] = ''
        books_dict[title]['author_intro'] = ''
        books_dict[title]['catelog'] = ''
        books_dict[title]['category'] = ''

    for index, book in enumerate(eslite_books):
        title = book['bookname']
        abstract = book['abstract']
        catelog = book['catelog']
        category = book['category']

        if title in books_dict.keys():
            books_dict[title]['abstract'] += ' ' + abstract
            books_dict[title]['catelog'] += ' ' + catelog
            books_dict[title]['category'] += ' ' + category

    print('Loaded books:', len(books_dict.keys()))

def segment_text(text):
    words = list(jieba.cut(text, cut_all=False))
    for word in words:
        if word in stopwords or word.isdigit():
            words.remove(word)

    joined = [' '.join(words)]
    return joined

def extract_keywords(text, k_ratio=0.25, k_num_max=10):
    keywords = []
    sentences = re.split('。|\n|，', text)

    for sentence in sentences:
        k_num = round(len(sentence) * k_ratio)
        sentence_keywords = jieba.analyse.extract_tags(sentence, topK=min(k_num, k_num_max)) # get keywords
        keywords += sentence_keywords
    return keywords

def build_corpus(mode='tfidf'):
    print('Building corpus...')
    global corpus

    length = len(books_dict.keys())
    for count, title in enumerate(books_dict.keys()):
        abstract = books_dict[title]['abstract']
        catelog = books_dict[title]['catelog']
        category = books_dict[title]['category']
        # segment all the info of each book
        title_segmented = segment_text(title)
        abstract_segmented = segment_text(abstract)
        catelog_segmented = segment_text(catelog)
        category_segmented = segment_text(category)
        info_segmented = title_segmented + abstract_segmented + catelog_segmented + category_segmented
        info_concatenated = '\n'.join(list(filter(None, info_segmented)))
        # extract keywords
        keywords = extract_keywords(info_concatenated)
        # add to corpus
        if mode == 'tfidf':
            corpus.append(' '.join(keywords))
        elif mode == 'word2vec':
            corpus.append(keywords)
        # print progress
        print('%.2f' % float(count / length * 100), '%', end = '\r')

    print('Corpus builded.')

def train_tfidf():
    print('Training TF-IDF...')
    X_train_counts = count_vectorizer.fit_transform(corpus)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print('TF-IDF trained.')

def initialize_text_tools():
    set_jieba()
    set_stopwords()
    build_books_dict()
    build_corpus(mode='word2vec')
    print('Text-tools initialized.')

def test_tfidf(text_segmented):
    X_new_counts = count_vectorizer.transform(text_segmented)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    return X_new_tfidf

def display_tfidf(tfidf):
    word = count_vectorizer.get_feature_names()
    weight = tfidf.toarray()
    for i in range(len(weight)):
        print("-------這裡輸出第", i, "類文本的詞語 tf-idf 權重------")
        for j in range(len(word)):
            print(word[j], weight[i][j])

def train_word2vec_model():
    dimension = 250
    model = word2vec.Word2Vec(corpus, size=dimension)

    # save our model
    model_path = '../data/250.model.bin'
    model.save(model_path)

def load_word2vec_model():
    model_path = '../data/250.model.bin'
    model = word2vec.Word2Vec.load(model_path)
    return model

def sentence2vec(sentence_segmented, word2vec_model):
    sentence_vec = []

    for word in sentence_segmented:
        if word in word2vec_model.wv.vocab:
            word_vec = word2vec_model[word]
            print(word, word_vec)
            if len(sentence_vec) > 0:
                sentence_vec += word_vec
            else:
                sentence_vec = word_vec

    return sentence_vec

def demo(model):
    print("提供 3 種測試模式\n")
    print("輸入一個詞，則去尋找前二十個該詞的相似詞")
    print("輸入兩個詞，則去計算兩個詞的餘弦相似度")
    print("輸入三個詞，進行類比推理")

    while True:
        try:
            query = input()
            q_list = query.split()

            if len(q_list) == 1:
                print("相似詞前 20 排序")
                res = model.most_similar(q_list[0],topn = 20)
                for item in res:
                    print(item[0]+","+str(item[1]))

            elif len(q_list) == 2:
                print("計算 Cosine 相似度")
                res = model.similarity(q_list[0],q_list[1])
                print(res)
            else:
                print("%s之於%s，如%s之於" % (q_list[0],q_list[2],q_list[1]))
                res = model.most_similar([q_list[0],q_list[1]], [q_list[2]], topn= 20)
                for item in res:
                    print(item[0]+","+str(item[1]))
            print("----------------------------")
        except Exception as e:
            print(repr(e))

if __name__ == '__main__':
    # initialize all needed for text-tools
    # initialize_text_tools()

    """
    # Train IDF
    train_tfidf()
    # Test TF-IDF
    test_text_segmented = ['曹操 三國 演義']
    tfidf = test_tfidf(test_text_segmented)
    print(tfidf)
    print('tfidf vector length: ', len(tfidf.toarray()[0]))
    """
    # train word2vec model
    # word2vec_model = train_word2vec_model()

    # load word2vec model
    word2vec_model = load_word2vec_model()

    # demo word2vec model
    test_word_1 = '孔子'
    test_word_2 = '論語'
    similarity = word2vec_model.similarity(test_word_1, test_word_2)

    test_sentence_segmented = [test_word_1, test_word_2]
    sentence_vec = sentence2vec(test_sentence_segmented, word2vec_model)
    print('sentence_vec', sentence_vec)

    # demo(word2vec_model)
