# Howdy! *E66Pa4P8nR7CnD

import json
import os
from flask import Flask, request
import tensorflow as tf
import numpy as np
import pandas as pd
from Doc2Vec import *
from preprocessing_utils import *

model_path = 'saved_model'
data_directory = 'data/Ecare_embedding'
model = tf.keras.models.load_model(f'{model_path}/binary_rep_improve')
vectorizer = Doc2VecVectorizer(f'{data_directory}/doc2vec_ingredients_and_instructions.model')
substitution_list = ["Remove_or_complex", "Quorn", "Mushroom", "Plant-based sausage", "Tofu", "Vegetable oil", "Seed", "Coconut oil", "Pumpkin", "Sweet potato", "Olive oil", "Plant-based cheese", "Lentil", "Margarine", "Double cream", "Cauliflower", "Chicken", "Pepper", "Bean", "Courgette", "Pork", "Turkey", "Mozzarella", "Long-grain rice", "Basmati rice", "Soya milk", "Almond milk", "Tomato", "Gram flour", "Chickpea", "Aubergine", "Carrot", "Squash", "Yam", "Soya mince", "Paneer", "Borlotti bean", "Haricot bean", "Lamb", "Bacon", "Beef", "Prawn", "White rice", "Jasmine rice", "Halloumi", "Vegetable stock cube", "Chicken stock", "Beef stock", "Potato", "Oil", "Plant-based yogurt", "Brown rice", "Kidney bean", "Vegetable stock", "Oat milk", "Coconut milk", "Rice milk", "Sunflower oil", "Single cream", "Cr\u00e8me fraiche", "Goats cheese", "Cheddar", "Natural Yogurt", "Bass", "Cashew cream", "Mascarpone", "Milk", "Coconut cream", "Soured cream", "Cashew nut", "Gouda", "Emmental", "Red Leicester", "Honey", "Agave", "Butter", "Rapeseed oil", "Butternut squash", "Chorizo", "Cod", "Pollock", "Halibut", "Haddock", "Turbot", "Plant-based mince", "White fish", "Egg", "Sardine", "Anchovy", "Jackfruit", "Butter bean", "Quinoa", "Bulgur wheat", "Onion", "Broad bean", "Pea", "Mushroom sauce", "Seaweed", "Plant-based bacon", "Broccoli", "Spinach", "Green bean", "Nut", "Monkfish", "Couscous", "Sausage", "Salmon", "Mayonnaise", "Evaporated milk", "Pasta", "Mackerel", "Parmesan", "Double Gloucester", "Feta", "Rice", "Wheat noodle", "Plant-based meatball", "Baking powder", "Granulated sugar", "Pearl barley", "Nutritional yeast", "Venison", "Cream cheese", "Pancetta", "Cannellini bean", "Plant-based burger", "Yogurt", "Cheese", "Tomato ketchup", "Bread", "Plant-based ham", "Tuna", "Gourd", "Tomato puree", "Quark", "Brie", "Veal", "Spring onion", "Kale", "Pak choi", "Bamboo shoots", "Peanut", "Cream", "Wild rice", "Edamame bean", "Aduki bean", "Ricotta", "Cottage cheese", "Pinto bean", "Wheat flour", "Rye flour", "Asparagus", "Salsa", "Barbeque sauce", "Celeriac", "Parsnip", "Almond", "Pine nut", "Leek", "Pesto", "Camembert", "Brussels Sprouts", "Lettuce", "Rice noodle", "Quorn sausage", "Spring green", "Cabbage", "Ham", "Blue Cheese", "Plant-based cream", "Rice flour", "Buckwheat flour"]
subembeddings = np.load('subembeddings_new.npy')
foodex_csv = pd.read_csv(f'{data_directory}/DB_foodon_to_foodex.csv',usecols=['Food','FOODON ID','FoodEx_IRI'],index_col=['Food'])
#print(foodex_csv[foodex_csv.duplicated('Food')])
foodex_dict = foodex_csv.to_dict(orient='index')
with open(f'{data_directory}/poincare_foodex2_500D.txt','r') as foodex_embedding_file:
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

def find_top_k_and_index(k, inputList):
    count = 0
    output = dict()
    tempList = list(inputList).copy()
    while count < k:
        maxi = max(tempList)
        for iterindex, item in enumerate(list(inputList)):
            if item == maxi:
                output[iterindex] = item
                tempList[iterindex] = min(list(inputList))
        count += 1
    return output


# Flask application
application = Flask(__name__)

@application.route('/', methods=['GET'])
def index():
    return 'Welcome!'


@application.route('/get_substitute', methods=['POST'])
def get_substitute():
    request_data = request.get_json()
    origin_recipes = [{'ingredients':[ing['ingredient_name'] for ing in request_data['ingredients']], 'instructions':[ins['instruction'] for ins in request_data['instructions']]}]
    doc2vec_vec = list(vectorizer.transform([list(to_recipe_string_list(origin_recipes,True))]))[0] #By default set with_instrcuctions = True
    #cuisine_vec = categorical_to_num(get_cuisine(origin_jsonobj), cuisines)
    #diet_vec = categorical_to_num(get_diet(origin_jsonobj), diets)
    high_carbon_ings = {ing['ingredient_number']:ing['matched_ingredient'] for ing in request_data['ingredients'] if ing['high_carbon']==True}
    recommend_list = list()
    for num, ing in high_carbon_ings.items():
        ref_foodex_vec = get_foodex_embedding(filter_food_name(ing.lower().title()))
        if ref_foodex_vec is not None:
            randvecs = np.array([np.concatenate((doc2vec_vec, ref_foodex_vec, subemb)) for subemb in subembeddings])    
            re = model.predict(randvecs)
            topNum = 5
            toplist = find_top_k_and_index(topNum, re)
            recommend_list.append({'ingredient_number':num, 'matched_ingredient':ing, 'substitution_list':[{'substitute':substitution_list[index],'probability':float(prob)} for index, prob in toplist.items()]})
        else:
            recommend_list.append({'ingredient_number':num, 'matched_ingredient':ing, 'substitution_list':[]})
    return json.dumps(recommend_list)

# run the app.
if __name__ == "__main__":
  application.debug = True
  application.run()
