#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unitconvert import massunits
from unitconvert import volumeunits
from nltk.stem import WordNetLemmatizer
import pandas as pd

stemmer = WordNetLemmatizer()
volume_unit_dictionary = {'dessertspoon':'dessertspoon','milliliter':'ml', 'ml':'ml', 'millilitre':'ml','liter': 'l', 'litre':'l', 'l':'l', 'teaspoon':'tsp', 'tsp':'tsp', 'tablespoon':'tbsp', 'tbsp':'tbsp', 'cup':'cup', 'quart':'qt', 'floz':'floz', 'pint':'pt', 'pt':'pt', 'qt':'qt', 'gallon':'gal', 'gal':'gal', 'lcup':'lcup', 'in3':'in3', 'ft3':'ft3'}
mass_unit_dictionary = {'oz':'oz','lb':'lb','g':'g','mg':'mg','kg':'kg','ounce':'oz','pound':'lb','gram':'g','kilogram':'kg','milligram':'mg'}

food_mass_tb = pd.read_csv(r'/home/mbaxshzf/Downloads/Food Masses.csv',dtype='str')
for col in food_mass_tb.columns: #['Unnamed: 0', 'Unnamed: 1', 'Mass (g)', 'Mass in kg', 'Synonym']
	food_mass_tb[col] = food_mass_tb[col].apply(lambda x: str(x).lower().strip())

carbon_density_tb = pd.read_csv(r'/home/mbaxshzf/Downloads/Master data - ECare - v1.6.xlsx - Master DB.csv',usecols=['Food','Alternate name','GHG (Mean)','Density in g/ml (including mass and bulk density)'],dtype='str')
for col in carbon_density_tb.columns: 
	carbon_density_tb[col] = carbon_density_tb[col].apply(lambda x: str(x).lower().strip())
cleaned_carbon_density_tb = carbon_density_tb.drop_duplicates()


"""def is_wet_ingredient(item: str) -> bool: # determine if the ingredient is wet or solid
	return False"""

def preprocess_unit(unit: str) -> list:
	return [stemmer.lemmatize(sp,'n') for sp in unit.strip().lower().split()]

def preprocess_item(item: str) -> str:
	return ' '.join([stemmer.lemmatize(token,'n') for token in item.strip().lower().split()])

def unit_expressed_in_massunit(unit: list):
	for u in unit:
		if u in mass_unit_dictionary:
			return mass_unit_dictionary[u]
	return False

def unit_expressed_in_volumeunit(unit: list):
	for u in unit:
		if u in volume_unit_dictionary:
			return volume_unit_dictionary[u]
	return False


def unit_expressed_in_spoons(unit: str) -> bool:
	if unit in ['tsp','tbsp','dessertspoon']:
		return True
	else:
		return False

def unit_expressed_in_scale(unit: list):
	for u in unit:
		if 'small' in u:
			return 'small'
		elif'medium' in u:
			return 'medium'
		elif 'large' in u:
			return 'large'
		else:
			return False

def convert_spoon_to_ml(amount: float, unit: str) -> float:
	if unit == 'tsp':
		return 5.0*amount
	elif unit == 'dessertspoon':
		return 10.0*amount
	elif unit == 'tbsp':
		return 15.0*amount
	else:
		raise Exception('unit did not expressed in spoon look up list')

def get_density(item: str) -> float: #need link to density database
	match_result = cleaned_carbon_density_tb.loc[(cleaned_carbon_density_tb['Food'] == item) | (cleaned_carbon_density_tb['Alternate name'] == item)]
	if len(match_result) > 0:
		return float(match_result['Density in g/ml (including mass and bulk density)'])
	else:
		print(f'{item} is not in the density CSV, return 0.0 g/ml')
		return 0.0

def convert_ml_to_kg(ml: float, item: str) -> float:
	density = get_density(item)
	return ml*density*0.001

def convert_mass_to_kg(amount: float, unit_in_mass: str) -> float:
	return massunits.MassUnit(amount, unit_in_mass, 'kg').doconvert()

def convert_volume_to_ml(amount: float, unit_in_vol: str) -> float:
	return volumeunits.VolumeUnit(amount, unit_in_vol, 'ml').doconvert()

def convert_scale_to_kg(amount: float, name: str, scale: str) -> float: #link to database
	name_match = food_mass_tb.loc[(food_mass_tb[food_mass_tb.columns[0]]==name) | (food_mass_tb['Synonym']==name)]
	if len(name_match) > 1:
		match = name_match.loc[name_match['Unnamed: 1']==scale]
		assert len(match) == 1
		return float(match['Mass in kg'])*amount
	elif len(name_match) == 1:
		return float(name_match['Mass in kg'])*amount
	else:
		print(f'{name} is not in the CSV, return 0.0')
		return 0.0

def convert_nounit_to_kg(amount: float, name: str) -> float: #link to database
	name_match = food_mass_tb.loc[(food_mass_tb[food_mass_tb.columns[0]]==name) | (food_mass_tb['Synonym']==name)]
	if len(name_match) > 1:
		match = name_match.loc[name_match['Unnamed: 1']=='medium']
		assert len(match) == 1
		return float(match['Mass in kg'])*amount
	elif len(name_match) == 1:
		return float(name_match['Mass in kg'])*amount
	else:
		print(f'{name} is not in the CSV, return 0.0')
		return 0.0

def kg_conversion(amount: str, unit: str, item: str) -> dict:
	try:
		num_amount = float(amount.strip())
		assert num_amount >= 0.0
	except Exception as e:
		print(f'Unable to process a non-numerical amount or negative amount:{amount}')
	else:
		if len(unit.strip()) > 0:
			result = {'amount':num_amount,'unit':preprocess_unit(unit),'item':preprocess_item(item)}
			if unit_expressed_in_massunit(result['unit']):
				return {'amount':convert_mass_to_kg(result['amount'],unit_expressed_in_massunit(result['unit'])),'unit':'kg','item':result['item']}
			elif unit_expressed_in_volumeunit(result['unit']):
				v = unit_expressed_in_volumeunit(result['unit'])
				if unit_expressed_in_spoons(v):
					return {'amount':convert_ml_to_kg(convert_spoon_to_ml(result['amount'],v),result['item']),'unit':'kg','item':result['item']}
				else:
					num_ml = convert_volume_to_ml(result['amount'],v)
					return {'amount':convert_ml_to_kg(num_ml,result['item']),'unit':'kg','item':result['item']}
			elif unit_expressed_in_scale(result['unit']):
				return {'amount':convert_scale_to_kg(num_amount,result['item'],unit_expressed_in_scale(result['unit'])),'unit':'kg','item':result['item']}
			elif 'qty' in result['unit']:
				return {'amount':convert_nounit_to_kg(num_amount,result['item']),'unit':'kg','item':result['item']}
			else:
				print(f'Unknown unit {unit} could not convert to kg, return 0.0')
				return {'amount':0.0,'unit':'kg','item':result['item']}
		else:
			return {'amount':convert_nounit_to_kg(num_amount,item.strip().lower()),'unit':'kg','item':preprocess_item(item)}

def get_carbon_per_kg(item: str) -> float: # link to GHG database
	match_result = cleaned_carbon_density_tb.loc[(cleaned_carbon_density_tb['Food'] == item) | (cleaned_carbon_density_tb['Alternate name'] == item)]
	if len(match_result) > 0:
		return float(match_result['GHG (Mean)'])
	else:
		print(f'{item} is not in the carbon CSV, return 0.0 GHG (Mean)')
		return 0.0

def calculate_carbon(result: dict) -> float:
	carbon_per_kg = get_carbon_per_kg(result['item'])
	CO2_vol = carbon_per_kg * result['amount']
	return CO2_vol

if __name__ == '__main__':
	print(kg_conversion('10','g pack','carrot'))