from wordcloud import WordCloud
from pprint import pprint
import gensim
import gensim.corpora as corpora
#WORDCLOUD_FONT_PATH = r'./data/Inkfree.ttf'
import simplemma
import string
import re
import pandas as pd
import numpy as np

def generate_wordcloud(docs, collocations: bool = False):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    wordcloud = WordCloud( width=700, height=600, background_color='white', collocations=collocations).generate(wordcloud_text)
    return wordcloud

def remove_stop_words(s):
  filtered_words = [i for i in s.split(" ") if i not in stop_words_list]
  clean_text = ' '.join(filtered_words)
  return clean_text

def transform_lemma(s):
  lang_data = simplemma.load_data("en")
  transformed_sent = []
  for tmp_word in s.split():
    cleaned = simplemma.lemmatize(tmp_word,lang_data)
    transformed_sent.append(cleaned)

  return " ".join(transformed_sent)

def clean_comments(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """
    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = s.replace("."," ")
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    s = s.replace("&#39;","'")

    s = s.translate(str.maketrans('', '', string.punctuation))
    

    s = s.replace("kaynak","").replace("\xa0"," ")\
    .replace("®","").replace("soure","")
    s = s.replace("toyota","").replace("toyota","")\
    .replace("hybrid","").replace("vehicle","").replace("corolla","")
    s= remove_stop_words(s)
    s= transform_lemma(s)

    return s.strip()

def prepare_training_data(data_df):
    data_words = data_df.comments_cleaned.apply(lambda x: x.split()).tolist()
    id2word = corpora.Dictionary(data_words)
    corpus = [id2word.doc2bow(text) for text in data_words]
    return id2word,corpus,data_words

def train_model(id2word,corpus, num_topics: int = 3, per_word_topics: bool = True):

    model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics, per_word_topics=per_word_topics)
    return model


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def find_lda_topic(x,id2word,corpus,lda_model,num_topics):
    
    bow = id2word.doc2bow(x.split())
    transform_document = lda_model.get_document_topics(bow)
    prediccted_topic = {i[0]:i[1] for i in transform_document}
    return max(prediccted_topic, key=prediccted_topic.get)