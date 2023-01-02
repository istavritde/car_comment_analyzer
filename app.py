import streamlit as st
import numpy as np
import pandas as pd
from utils import generate_wordcloud, train_model,format_topics_sentences,prepare_training_data,find_lda_topic
import random
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime

today = datetime.date.today()
one_year_ago = today - datetime.timedelta(days=365)


st.title("Best Selling Cars in Turkiye")

car_options = ["toyota corolla","toyota corolla hybrid","fiat egea","renault clio hb"]

st.text("Select a car brand to view the statistics")

car_selected = st.selectbox("Best Selling Cars",options=car_options)

car_selected= car_selected.replace(" ","-")


all_cars =pd.read_pickle("files/all_cars_en_cleaned_v3.pickle")
all_cars['date'] = pd.to_datetime(all_cars['date'])

st.text("Please select date range for the data")
start_date = st.date_input('Start date', one_year_ago)
end_date = st.date_input('End date', today)
if start_date < end_date:
    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date.')

# filter data by selection of brand and date range
data_all = all_cars[all_cars.source==car_selected]
data_all = data_all[(data_all['date'] >= str(start_date)) & (data_all['date'] <= str(end_date))]

# number of topics
num_topics = 3
# Build LDA model
id2word, corpus,data_words = prepare_training_data(data_all)
lda_model =train_model(id2word,corpus,num_topics=num_topics) 

# assign topics to raw data
data_all['lda_topic']=data_all['comments_cleaned'].apply(lambda x:find_lda_topic(x,id2word,corpus,lda_model,num_topics) )

fig = plt.figure()

gs = gridspec.GridSpec(2,2)

ax1=fig.add_subplot(gs[0,0])
ax2=fig.add_subplot(gs[0,1])
ax3=fig.add_subplot(gs[1,:])
fig.tight_layout()
#[["comment_lenght", "word_count", "sentiment"]].hist(bins=20, figsize=(15, 10))
data_all["word_count"].plot(kind='hist',ax = ax1,title='Word Count',color='green')
data_all["comment_lenght"].plot(kind='hist',ax = ax2,title='Comment Length',color='blue')
data_all["sentiment"].plot(kind='hist',ax = ax3,title='Comment Sentiment',color='red')
st.pyplot(fig)

st.subheader('Top N Topic Keywords Wordclouds')
topics = lda_model.show_topics(formatted=False, num_topics=num_topics)
cols = st.beta_columns(3)
COLORS= ["red",'yellow','blue','cyan','green','purple','orange','brown','gray']
colors = random.sample(COLORS, k=len(topics))
for index, topic in enumerate(topics):
    wc = WordCloud( width=700, height=600, background_color='white', prefer_horizontal=1.0, color_func=lambda *args, **kwargs: colors[index])
    with cols[index % 3]:
        wc.generate_from_frequencies(dict(topic[1]))
        st.image(wc.to_image(), caption=f'Topic #{index}', use_column_width=True)

df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=data_words)

# Display setting to show more characters in column
pd.options.display.max_colwidth = 100

sent_topics_sorteddf_mallet = pd.DataFrame()
sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

# Show
st.dataframe(sent_topics_sorteddf_mallet[['Topic_Num','Representative Text']].head(10))
