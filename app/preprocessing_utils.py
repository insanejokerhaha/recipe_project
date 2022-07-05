import json
import string
import re
from nltk import download
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
download('wordnet')
download('stopwords')
download('omw-1.4')


lemmatizer = WordNetLemmatizer()

directory = 'data/Ecare_embedding'  # specify the directory that contains all the files needed
with open(f'{directory}/synonyms.json', 'r') as f:
    synonyms = json.load(f)
with open(f'{directory}/food_names.json', 'r') as f:
    food_names = json.load(f)

all_names = set(food_names).union(synonyms.keys())


def get_name(ing):
    if ing in food_names:
        return ing
    return synonyms[ing]


def to_recipe_string_list(recipes, with_instructions=False):
    """
    Generator that takes in Recipe1M format recipes and returns the normalised
    string representation
    """
    for recipe in recipes:
        recipe_ings = []
        for ing in recipe['ingredients']:
            filtered_ing = filter_ingredient(ing)
            if filtered_ing:
                recipe_ings.append(filtered_ing)
        ing_string = " ".join(recipe_ings)
        if with_instructions:
            recipe_insts = []
            for inst in recipe['instructions']:
                recipe_insts.append(filter_instruction(inst))
                instructions_string = " || ".join(recipe_insts)
                yield ing_string + " @@ " + instructions_string
        else:
            yield ing_string


def filter_instruction(ins):
    """
    Takes in a string and normalises the ingredients without removing other words
    """
    ins = ins.lower()

    ins = ins.replace('-', ' ')
    ins = ins.replace(',', ' ')
    ins = ins.replace('/', ' ')

    # remove punctuation
    ins = ins.translate(str.maketrans('', '', string.punctuation))

    # remove digits
    ins = re.sub(r'\d', "", ins)

    # lemmatize words and remove stopwords
    sw = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in ins.split() if word not in sw]

    ins = ''
    i = 0
    while i < len(words) - 2:
        if f'{words[i]}_{words[i + 1]}_{words[i + 2]}' in all_names:
            ins += get_name(f'{words[i]}_{words[i + 1]}_{words[i + 2]}') + ' '
            i += 2
        elif f'{words[i]}_{words[i + 1]}' in all_names:
            ins += get_name(f'{words[i]}_{words[i + 1]}') + ' '
            i += 1
        elif f'{words[i + 1]}_{words[i]}' in all_names:
            ins += get_name(f'{words[i + 1]}_{words[i]}') + ' '
            i += 1
        elif words[i] in all_names:
            ins += get_name(words[i]) + " "
        else:
            ins += words[i] + " "
        i += 1

    # if there are 2 remaining words
    if i == len(words) - 2:
        if f'{words[i]}_{words[i + 1]}' in all_names:
            ins += get_name(f'{words[i]}_{words[i + 1]}')
        elif f'{words[i + 1]}_{words[i]}' in all_names:
            ins += get_name(f'{words[i + 1]}_{words[i]}')
        elif f'{words[i + 1]}_{words[i]}' in all_names:
            ins += get_name(f'{words[i + 1]}_{words[i]}')
        else:
            ins += " ".join(words[-2:])

    # if there's a remaining word
    if i == len(words) - 1:
        if words[i] in all_names:
            ins += get_name(words[i])

    return " ".join(ins.split())


def filter_ingredient(ing):
    """
    Takes in a string and removes words that aren't defined ingredients
    """
    ing = ing.lower()

    ing = ing.replace('-', ' ')
    ing = ing.replace(',', ' ')
    ing = ing.replace('/', ' ')

    # remove punctuation except parentheses and dashes
    ing = ing.translate(str.maketrans('', '', string.punctuation.replace('()', "")))

    # remove parenthesised items
    ing = re.sub(r'\(.*\)', "", ing)

    # remove fractions
    ing = re.sub(r'\d/\d', "", ing)

    # remove digits
    ing = re.sub(r'\d', "", ing)

    # lemmatize words
    words = [lemmatizer.lemmatize(word) for word in ing.split()]

    # the following loop ensures multi-word ingredient names
    # are included without including the subwords
    ing = ''
    i = 0
    while i < len(words) - 2:
        if f'{words[i]}_{words[i + 1]}_{words[i + 2]}' in all_names:
            ing += get_name(f'{words[i]}_{words[i + 1]}_{words[i + 2]}') + ' '
            i += 2
        elif f'{words[i]}_{words[i + 1]}' in all_names:
            ing += get_name(f'{words[i]}_{words[i + 1]}') + ' '
            i += 1
        elif f'{words[i + 1]}_{words[i]}' in all_names:
            ing += get_name(f'{words[i + 1]}_{words[i]}') + ' '
            i += 1
        elif words[i] in all_names:
            ing += get_name(words[i]) + " "
        i += 1
    # if there are 2 remaining words
    if i == len(words) - 2:
        if f'{words[i]}_{words[i + 1]}' in all_names:
            ing += get_name(f'{words[i]}_{words[i + 1]}')
        elif f'{words[i + 1]}_{words[i]}' in all_names:
            ing += get_name(f'{words[i + 1]}_{words[i]}')
        else:
            if words[i] in all_names:
                ing += get_name(words[i]) + ' '
            if words[i + 1] in all_names:
                ing += get_name(words[i + 1])

    # if there's 1 remaining word
    if i == len(words) - 1:
        if words[i] in all_names:
            ing += get_name(words[i])
    return " ".join(ing.split())