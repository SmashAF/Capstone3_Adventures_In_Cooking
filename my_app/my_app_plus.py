from flask import Flask, request, render_template
import pickle as pickle
import pandas as pd
import numpy as np
import spacy
# import en_core_web_sm
import string
from sklearn.feature_extraction.text import TfidfVectorizer
# from vectorizer_clean import TextClassifer
from sklearn.metrics.pairwise import cosine_similarity
from pre_processing_clean import nlp_list, remove_pos, remove_pos_list, combine_words, combine_words_list, remove_extra_quotes, stop_words_singular
from Recomatic import recomatic
import _pickle as cPickle
import gc

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('css_template.html')

@app.route('/rating_recommendation')
def get_rating_recommendation():
    return render_template('rating_recommend_template.html')

@app.route('/get_recommendation')
def get_recommendation():
    return render_template('recommend_template.html')

@app.route('/predict', methods=['POST'])
def predict():
    # nlp = spacy.load("en_core_web_sm")
    ingredients = [str(request.form['user_input'])]
    ingredients = ingredients[0].replace(', ', ',')
    ingredients=[ingredients]
    ingredients = ingredients[0].split(',')

    # list_nlp = nlp_list(ingredients, nlp)
    # remove = remove_pos_list(list_nlp)
    # combined = [combine_words(i) for i in remove]
    # clean = remove_extra_quotes(combined)
    # clean_raw = [clean]
    gc.disable()

    with open('data/pandas_data.pkl', 'rb') as f:
        df = pickle.load(f)
    # with open('data/Features.pkl', 'rb') as f:
    #     model = pickle.load(f)

    with open('data/Features.pkl', 'rb') as f:
        Tfidf = pickle.load(f)
        
    with open('data/TFIDF_model.pkl', 'rb') as f:
        model = pickle.load(f)


    gc.enable()


    # Tfidf_ingred = model.transform(clean_raw)
    # new_Tfidf = np.append(Tfidf_ingred.toarray(), Tfidf.toarray())
    # new_Tfidf = np.append(Tfidf_ingred, Tfidf)
    # new_Tfidf = new_Tfidf.reshape(637, 1075)

    # cosine_similarities = cosine_similarity(Tfidf, Tfidf_ingred)
    # cosine_similarities = map(lambda x: cosine_similarity(Tfidf_ingred, x), Tfidf)

    # similar_indices = np.argsort(cosine_similarities, axis=0)#[-5:-1]
    # similar_items = [df.values[i] for i in similar_indices]
    recs = recomatic(ingredients)

    t1 = recs.loc[0][0]
    ingred1 = recs.loc[0][1]
    instructions1 = recs.loc[0][2]

    
    t2 = recs.loc[1][0]
    ingred2 = recs.loc[1][1]
    instructions2 = recs.loc[1][2]
 

    t3 = recs.loc[2][0]
    ingred3 = recs.loc[2][1]
    instructions3 = recs.loc[2][2]
 

    grocery_bag= ingred1, ingred2, ingred3

    # return render_template('predict_template.html', data =(title, stars, description, ingredients, link, title_2, stars_2, description_2, ingredients_2, link_2, title_3, stars_3, description_3, ingredients_3, link_3), ingredients=ingredients_for_groceries, input_ingredients=clean)
    return render_template('predict_template.html', data =(t1, ingred1, instructions1, t2, ingred2, instructions2, t3, ingred3, instructions3) , ingredients = grocery_bag, input_ingredients=ingredients)

@app.route('/get_groceries', methods=['GET', 'POST'])
def get_groceries():
    input_ingredients = request.args.get('input_ingredients')
    choice = request.args.get('ingredients')
    choice_update = choice.split(' ')
    input_ingredients_update = input_ingredients
    grocery_list = [val for val in choice_update if val not in input_ingredients_update]
    grocery_update = [grocery_list[i].replace('_', ' ') for i in range(len(grocery_list))]
    return render_template('grocery_template.html', data=grocery_update)
@app.route('/get_info')
def get_info():
    return render_template('info.html')

if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm")
    app.run(host='0.0.0.0', port=5000, debug=True)
