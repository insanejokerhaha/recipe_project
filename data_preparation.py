import pandas as pd
import numpy as np
import random
import os
import json
from Doc2Vec import *
from preprocessing_utils import *
from sklearn.model_selection import train_test_split
import numpy as np
from gensim.models import Word2Vec

import regex as re
from nltk import download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
#download('wordnet')
#download('stopwords')
#download('punkt')
lemmatizer = WordNetLemmatizer()

def cleanse_name(name):
  # remove words in brackets
  reg = r'\([\s\S]*\)'
  result = re.sub(reg, '', name)
  result = result.strip().lower()

  # tokenise
  tokens = word_tokenize(result)
  tokens_without_sw = [word for word in tokens if not word in stopwords.words('english')]

  result = ''
  for token in tokens_without_sw:
    result = result + lemmatizer.lemmatize(token) + ' '
  result = result.strip()
  return result

model = Word2Vec.load("word2vec.model")


def get_vector_representation(foodname):
  result = None
  foodname = cleanse_name(foodname)
  if ' ' in foodname:
    ngram = foodname.lower().replace(' ', '_')
    if ngram in model.wv:
      result = model.wv[ngram]
      # print('BRANCH 1:' + str(result.shape))
      return result

  vector_list = []
  for word in foodname.split(' '):
    if word in model.wv:
      vector_list.append(model.wv[word])
  if len(vector_list) < 1:
    result = None
  else:
    result = np.mean(vector_list, axis=0)
  # print('BRANCH 2:' + str(result.shape))
  return result

directory = 'data/Ecare_embedding'
cuisines = {0: 'Afro-Caribbean', 1: 'European', 2: 'American', 3: 'Mediterranean', 4: 'Asian', 5: 'Latin American', 6: 'Indian', 7: 'Italian'}
diets = {0: 'paleolithic', 1: 'gluten free', 2: 'primal', 3: 'fodmap friendly', 4: 'lacto ovo vegetarian', 5: 'dairy free', 6: 'pescatarian', 7: 'vegan'}
vectorizer = Doc2VecVectorizer(f'{directory}/doc2vec_ingredients_and_instructions.model')
cdcsv = pd.read_csv(f'{directory}/jsonfiles_with_instructions.csv', dtype={'jsonfile':str}, index_col='jsonfile')
cddict = cdcsv.to_dict(orient='index')

"""glove_embeddings_index = dict()
f = open(f'{directory}/glove.6B.300d.txt','r',encoding='utf8')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove_embeddings_index[word] = coefs

f.close()
print(f'Loaded {len(glove_embeddings_index)} word vectors. {coefs.shape}')"""

def categorical_to_num(itemlist,categories):
  vec = np.zeros(8,dtype=np.float32)
  for k,v in categories.items():
    if v in itemlist:
      vec[k] = 1
    else:
      vec[k] = 0
  return vec

def get_cuisine_and_diet_from_csv(recipe_id):
  if recipe_id in cddict.keys():
    cuisine_vec = categorical_to_num(cddict[recipe_id]['cuisine'],cuisines)
    diet_vec = categorical_to_num(cddict[recipe_id]['diet'],diets)
    return (cuisine_vec, diet_vec)
  else:
    print('Recipe ID not in the look-up list')
    return (np.zeros(8), np.zeros(8))


foodex_csv = pd.read_csv(f'{directory}/DB_foodon_to_foodex.csv',usecols=['Food','FOODON ID','FoodEx_IRI'],index_col=['Food'])
#print(foodex_csv[foodex_csv.duplicated('Food')])
foodex_dict = foodex_csv.to_dict(orient='index')
foodex_reverse_dict = {foodex_dict[k]['FoodEx_IRI']: k for k in foodex_dict.keys()}

with open(f'{directory}/poincare_foodex2_500D.txt','r') as foodex_embedding_file:
  lines = foodex_embedding_file.readlines()
  foodex_embedding_dict = {line.split()[0]:np.array(line.split()[1:],dtype=np.float) for line in lines}

def get_foodex_embedding(food_name):
  def iri_exist(food_name_iri):
    foodex_iri = foodex_dict[food_name_iri]['FoodEx_IRI']
    if foodex_iri in foodex_embedding_dict.keys():
      return foodex_embedding_dict[foodex_iri]
    else:
      return None
  if food_name in foodex_dict.keys():
    return iri_exist(food_name)
  elif food_name.split()[0] in foodex_dict.keys():
    return iri_exist(food_name.split()[0])
  else:
    return None


def filter_food_name(ins):
  ins = ins.replace(':', ' ')
  ins = ins.replace('_', ' ')
  return ins

def get_neg_embedding(recipe_id, masterDB_name):
  poslist = [filter_food_name(temp_name) for temp_name in query_table.loc[(query_table['recipe_id'] == recipe_id) & (query_table['MasterDB match for subsitutable ingredient'] == masterDB_name)]['DB Food'].tolist()]
  addlist = list()
  for i in poslist:
    if i.split()[0] not in poslist:
      addlist.append(i.split()[0])
  poslist = poslist + addlist
  randindex = random.randint(0,len(foodex_dict.keys())-1)
  keylist = list(foodex_dict.keys())
  while keylist[randindex] in poslist:
    randindex = random.randint(0,len(foodex_dict.keys())-1)
  foodex_iri = foodex_dict[keylist[randindex]]['FoodEx_IRI']
  return foodex_iri

def get_combine_embedding(recipe_id, ingredient_name, masterDB_name, sub_DB_name):
  try:
    with open(f'{directory}/jsonfiles_with_instructions/{recipe_id}.json','r') as origin_jsonfile:
      origin_jsonobj = json.load(origin_jsonfile)
      origin_recipes = [{'ingredients':[ing['ingredient_name'] for ing in origin_jsonobj['ingredients']], 'instructions':[ins['instruction'] for ins in origin_jsonobj['instructions']]}]
    with open(f'{directory}/jsonfiles_with_instructions_rep_DBfood/{recipe_id}.json','r') as rep_jsonfile:
      jsonobj = json.load(rep_jsonfile)
      rep_recipes = [{'ingredients':[ing['ingredient_name'] for ing in jsonobj['ingredients']], 'instructions':[ins['instruction'] for ins in jsonobj['instructions']]}]
  except FileNotFoundError as e:
    print('The recipe does not exist, please choose another one.')
    return None
  else:
    doc2vec_vec = list(vectorizer.transform([list(to_recipe_string_list(origin_recipes,True))]))[0] #By default set with_instrcuctions = True
    rep_doc2vec_vec = list(vectorizer.transform([list(to_recipe_string_list(rep_recipes,True))]))[0]
    (cuisine_vec, diet_vec) = get_cuisine_and_diet_from_csv(recipe_id)
    ref_foodex_vec = get_foodex_embedding(filter_food_name(masterDB_name))

    pos_foodex_vec = get_foodex_embedding(filter_food_name(sub_DB_name))
    if (ref_foodex_vec is not None) and (pos_foodex_vec is not None):
      """if filter_food_name(masterDB_name).split()[0].lower() in glove_embeddings_index:
        ref_glove_vec = glove_embeddings_index[filter_food_name(masterDB_name).split()[0].lower()]
      else:
        ref_glove_vec = np.zeros(300)
      if filter_food_name(sub_DB_name).split()[0].lower() in glove_embeddings_index:
        pos_glove_vec = glove_embeddings_index[filter_food_name(sub_DB_name).split()[0].lower()]
      else:
        pos_glove_vec = np.zeros(300)"""

      """ref_w2v_vec = get_vector_representation(filter_food_name(masterDB_name).split()[0].lower())
      if ref_w2v_vec is None:
        ref_w2v_vec = np.zeros(100)
      pos_w2v_vec = get_vector_representation(filter_food_name(sub_DB_name).split()[0].lower())
      if pos_w2v_vec is None:
        pos_w2v_vec = np.zeros(100)"""

      neg_foodex_iri = get_neg_embedding(recipe_id, masterDB_name)
      if neg_foodex_iri in foodex_embedding_dict.keys():
        neg_foodex_vec = foodex_embedding_dict[neg_foodex_iri]
      else:
        neg_foodex_vec = np.zeros(500)
      """if filter_food_name(foodex_reverse_dict[neg_foodex_iri]).split()[0].lower() in glove_embeddings_index:
        neg_glove_vec = glove_embeddings_index[filter_food_name(foodex_reverse_dict[neg_foodex_iri]).split()[0].lower()]
      else:
        neg_glove_vec = np.zeros(300)"""
      """neg_w2v_vec = get_vector_representation(filter_food_name(foodex_reverse_dict[neg_foodex_iri]).split()[0].lower())
      if neg_w2v_vec is None:
        neg_w2v_vec = np.zeros(100)"""
      with open(f'{directory}/jsonfiles_with_instructions/{recipe_id}.json', 'r') as rep:
          content = rep.read().lower().replace(f'{ingredient_name.lower()}', filter_ingredient(filter_food_name(foodex_reverse_dict[neg_foodex_iri])))
          neg_jsonobj = json.loads(content)
          neg_recipes = [{'ingredients': [ing['ingredient_name'] for ing in neg_jsonobj['ingredients']],
                          'instructions': [ins['instruction'] for ins in neg_jsonobj['instructions']]}]
          neg_doc2vec_vec = list(vectorizer.transform([list(to_recipe_string_list(neg_recipes,True))]))[0]
      #return (np.concatenate((doc2vec_vec,cuisine_vec,diet_vec,ref_foodex_vec)), np.concatenate((doc2vec_vec,cuisine_vec,diet_vec,pos_foodex_vec)), np.concatenate((doc2vec_vec,cuisine_vec,diet_vec,neg_foodex_vec)))
      #return (np.concatenate((doc2vec_vec,cuisine_vec,diet_vec,ref_foodex_vec,pos_foodex_vec)), np.concatenate((doc2vec_vec, cuisine_vec, diet_vec, ref_foodex_vec, neg_foodex_vec)))
      #return (np.concatenate((doc2vec_vec,ref_foodex_vec,ref_glove_vec)), np.concatenate((rep_doc2vec_vec,pos_foodex_vec,pos_glove_vec)), np.concatenate((neg_doc2vec_vec,neg_foodex_vec,neg_glove_vec)))
      return (np.concatenate((doc2vec_vec, ref_foodex_vec, pos_foodex_vec)), np.concatenate((doc2vec_vec, ref_foodex_vec, neg_foodex_vec)))
      #return (np.concatenate((doc2vec_vec, ref_w2v_vec, pos_w2v_vec)), np.concatenate((doc2vec_vec, ref_w2v_vec, neg_w2v_vec)))

    else:
      return None

query_table = pd.read_csv(f'{directory}/recipe_substitution_10Sept_7Oct_new.csv')

def get_w2vonly_embedding(recipe_id, ingredient_name, masterDB_name, sub_DB_name):
  try:
    with open(f'{directory}/jsonfiles_with_instructions/{recipe_id}.json','r') as origin_jsonfile:
      origin_jsonobj = json.load(origin_jsonfile)
      origin_recipes = [{'ingredients':[ing['ingredient_name'] for ing in origin_jsonobj['ingredients']], 'instructions':[ins['instruction'] for ins in origin_jsonobj['instructions']]}]
    with open(f'{directory}/jsonfiles_with_instructions_rep_DBfood/{recipe_id}.json','r') as rep_jsonfile:
      jsonobj = json.load(rep_jsonfile)
      rep_recipes = [{'ingredients':[ing['ingredient_name'] for ing in jsonobj['ingredients']], 'instructions':[ins['instruction'] for ins in jsonobj['instructions']]}]
  except FileNotFoundError as e:
    print('The recipe does not exist, please choose another one.')
    return None
  else:
    doc2vec_vec = list(vectorizer.transform([list(to_recipe_string_list(origin_recipes,True))]))[0] #By default set with_instrcuctions = True
    rep_doc2vec_vec = list(vectorizer.transform([list(to_recipe_string_list(rep_recipes,True))]))[0]
    (cuisine_vec, diet_vec) = get_cuisine_and_diet_from_csv(recipe_id)
    ref_w2v_vec = get_vector_representation(filter_food_name(masterDB_name).split()[0].lower())
    if ref_w2v_vec is None:
      ref_w2v_vec = np.zeros(100)
    pos_w2v_vec = get_vector_representation(filter_food_name(sub_DB_name).split()[0].lower())
    if pos_w2v_vec is None:
      pos_w2v_vec = np.zeros(100)
    neg_foodex_iri = get_neg_embedding(recipe_id, masterDB_name)
    if neg_foodex_iri in foodex_embedding_dict.keys():
      neg_foodex_vec = foodex_embedding_dict[neg_foodex_iri]
    else:
      neg_foodex_vec = np.zeros(500)
    neg_w2v_vec = get_vector_representation(filter_food_name(foodex_reverse_dict[neg_foodex_iri]).split()[0].lower())
    if neg_w2v_vec is None:
      neg_w2v_vec = np.zeros(100)
    with open(f'{directory}/jsonfiles_with_instructions/{recipe_id}.json', 'r') as rep:
        content = rep.read().lower().replace(f'{ingredient_name.lower()}', filter_ingredient(filter_food_name(foodex_reverse_dict[neg_foodex_iri])))
        neg_jsonobj = json.loads(content)
        neg_recipes = [{'ingredients': [ing['ingredient_name'] for ing in neg_jsonobj['ingredients']],
                        'instructions': [ins['instruction'] for ins in neg_jsonobj['instructions']]}]
        neg_doc2vec_vec = list(vectorizer.transform([list(to_recipe_string_list(neg_recipes,True))]))[0]
    return (np.concatenate((doc2vec_vec, ref_w2v_vec, pos_w2v_vec)), np.concatenate((rep_doc2vec_vec, ref_w2v_vec, neg_w2v_vec)))


def prepare_data():
  X = list()
  Y = list()
  for index in range(0,len(query_table)):
    query = query_table.iloc[index]
    if not query['DB Food'].endswith('SAME'):
      result = get_combine_embedding(int(query['recipe_id']), query['ingredient_name'], query['MasterDB match for subsitutable ingredient'],query['DB Food'])
      if result is not None:
        (pos_vec, neg_vec) = result
        #X.append(anchor_vec)
        #Y.append(1)
        X.append(pos_vec)
        Y.append(1)
        X.append(neg_vec)
        Y.append(0)
  return X, Y

if __name__ == '__main__':

  X, y = prepare_data()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  x_train = np.array(X_train)
  x_test = np.array(X_test)
  y_train = np.array(y_train)
  y_test = np.array(y_test)

  directory = 'data/Ecare_embedding/docrep_improve'
  if not os.path.exists(directory):
    os.mkdir(directory)

  np.save(f'{directory}/X_train.npy', x_train)
  np.save(f'{directory}/y_train.npy', y_train)
  np.save(f'{directory}/X_test.npy', x_test)
  np.save(f'{directory}/y_test.npy', y_test)
