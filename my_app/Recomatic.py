import pandas as pd
import numpy as np
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from collections import Counter
import nltk
vocabulary = nltk.FreqDist()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
import pickle
import numpy as np
import pandas as pd
import _pickle as cPickle
import gc

recipes = pd.read_pickle('/Users/scottsturtz/DSI/School_Work/Smash_Capstones/Capstone3/my_app/data/pandas_data.pkl')


def reco_maker(top_n, scores):
    print(scores[:3])
    top = sorted(range(len(scores)), key= lambda i: scores[i], reverse=True)[:top_n]

    recos = pd.DataFrame(columns = ["recipe", "ingredients",'instructions', "score"])

    count= 0
    for i in top:
        recos.at[count, "recipe"] = format_title(recipes["item_name"][i])
        recos.at[count, "ingredients"] = recipes["combined"][i]
        recos.at[count, "instructions"] = recipes['summary'][i]
        # recos.at[count, "score"] = "{:.3f}".format(float(scores[i]))
        
        count += 1
    return recos

def format_title(title):
    title = unidecode(title)
    return title

# def pretty_grocery(ingredient):
    
#     if isinstance(ingredient, list):
#         ingredients=ingredient
#     # else:
#     #     ingredients = ast.literal_eval(ingredient)   

#     ingredients = ",".join(ingredients)
#     ingredients = unidecode(ingredients)
#     return ingredients

def recomatic(ingredients, top_n=3):

    gc.disable()
    with open('/Users/scottsturtz/DSI/School_Work/Smash_Capstones/Capstone3/my_app/data/TFIDF_Model.pkl','rb') as f:
        tfidf = pickle.load(f)

    with open('/Users/scottsturtz/DSI/School_Work/Smash_Capstones/Capstone3/my_app/data/Features.pkl', 'rb') as f:
        feats = pickle.load(f)

    gc.enable()

    parsed = " ".join(ingredients)


    ingredients_tfidf = tfidf.transform([parsed])
    #cosine similarity  
    similarity= map(lambda x: cosine_similarity(ingredients_tfidf, x), feats)
    scores = list(similarity)

    #filter reccomendations
    reccomendations = reco_maker(top_n, scores)

    return reccomendations


# if __name__ == "__main__":
    # test ingredients
    # test_ingredients = ['ground beef', 'pasta', 'diced tomatoes']
    # recs = recomatic(test_ingredients)
    