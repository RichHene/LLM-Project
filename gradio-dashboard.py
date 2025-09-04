import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr #package that allows to build dashboard to showcase ML models


load_dotenv()

books = pd.read_csv('books_with_emotions.csv')
#books dataset provides a dataset which links to google books
#we want the largest possible size available for better resolution

books['large_thumbnail'] = books['thumbnail'] + '&fife=w800'
#a number of books do not have covers
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.jpg', #repalce empty thumbnails with cover not found jpg
    books['large_thumbnail']
)

#build a vector database
raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load() #read tagged descriptions into TextLoader
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=0.1, chunk_overlap=0) #instantiating a character text splitter
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings()) #convert into document embeddings using OpenAiEmbeddings, stored in Chroma database

#a function that retrieves semantic recommendations from the books dataset
#and apply filtering based on category and sorting based on emotional tone

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50, #retrieve 50 recommendations
        final_top_k: int = 16, #filtered to 16, looks good on the dashboard
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k = initial_top_k)#get recommendations from db_books
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != 'All':
        books_recs = books_recs[books_recs['simple_categories'] == category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == 'Happy':
        books_recs.sort_values(by = 'joy', ascending = False, inplace = True)
    elif tone == 'Surprising':
        books_recs.sort_values(by = 'surprise', ascending = False, inplace = True)
    elif tone == 'Angry':
        books_recs.sort_values(by = 'Anger', ascending = False, inplace = True)
    elif tone == 'Suspenseful':
        books_recs.sort_values(by = 'fear', ascending = False, inplace = True)
    elif tone == 'Sad':
        books_recs.sort_values(by = 'sadness', ascending = False, inplace = True)

    return books_recs

#a function that specifies what we want to display on the gradio dashboard
def recommend_books(
        query: str,
        category: str,
        tone: str,
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []
    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_desc_split = ' '.join(truncated_desc_split[:30]) + '...' #if description has more than 30 words cut ir off, and make itr continuous
        authors_split =  row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f'{authors_split[0]} {authors_split[1]}'
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row['authors']

        caption = f"{row['title']} by {authors_str}: {truncated_desc_split}"
        results.append([row['large_thumbnail'], caption])
    return results


categories = ['All'] + sorted(books['simple_categories'].unique())
tones = ['All'] + ['Happy', 'Surprising', 'Sadness', 'Angry', 'Suspenseful']

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown('# Semantic Book Recommender')

    with gr.Row():
        user_query = gr.Textbox(label = 'Please enter a description of a book:', placeholder = 'e.g A story about forgiveness')

        category_dropdown = gr.Dropdown(categories, label = 'Select a category', value = 'All')
        tone_dropdown = gr.Dropdown(tones, label = 'Select an emotional tone', value = 'All')

        submit_button = gr.Button('Find Recommendations')

        gr.Markdown('## Results')

        output = gr.Gallery(label = 'Recommended books', columns = 8, rows = 2)
        submit_button.click(fn = recommend_books,
                            inputs = [user_query, category_dropdown, tone_dropdown],
                            outputs = output)




if __name__ == '__main__':
    dashboard.launch()