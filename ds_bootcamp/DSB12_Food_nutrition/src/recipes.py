import pandas as pd
import numpy as np
import joblib
import re

INGREDIENT_ALIASES = {
    'milk':          'milk/cream',
    'cream':         'milk/cream',
    'jam':           'jam or jelly',
    'jelly':         'jam or jelly',
    'scallion':      'green onion/scallion',
    'green onion':   'green onion/scallion',
    'sweet potato':  'sweet potato/yam',
    'yam':           'sweet potato/yam',
    'cornmeal':      'hominy/cornmeal/masa',
    'masa':          'hominy/cornmeal/masa',
    'puff pastry':   'phyllo/puff pastry dough',
    'phyllo':        'phyllo/puff pastry dough',
    'cognac':        'cognac/armagnac',
    'armagnac':      'cognac/armagnac',
}

def resolve(ingredient):
    return INGREDIENT_ALIASES.get(ingredient.lower().strip(), ingredient.lower().strip())

def tokenize(text: str) -> set[str]:
    """Process input text in set of words"""
    return {t for t in re.split(r"[\/\s]+", text) if t}

def token_search(ingredient: str, ingredients_list: list) -> str:
    ingredient = resolve(ingredient)
    if ingredient in ingredients_list:
        return ingredient

    ing_tokens = tokenize(ingredient)
    if not ing_tokens:
        return None

    for ing in ingredients_list:
        from_list_tokens = tokenize(ing)
        if ing_tokens.issubset(from_list_tokens):
            return ing

class Predictor:
    def __init__(self, model_path='data/best_model_voting.pkl'):
        self.model = joblib.load(model_path)
        try:
            self.all_features = self.model.named_steps['model'].feature_names_in_
        except (AttributeError, KeyError):
            try:
                self.all_features = self.model.feature_names_in_
            except AttributeError:
                self.all_features = []

    def predict(self, input_ingredients):
        if not len(self.all_features):
            return "unknown"
        vector = pd.DataFrame(0, index=[0], columns=self.all_features)
        for ing in input_ingredients:
            found = token_search(ing, self.all_features)
            if found in vector.columns and found != None:
                vector[found] = 1
        prediction = self.model.predict(vector)[0]
        return str(prediction)


class NutritionFacts:
    def __init__(self, csv_path='data/ingredients_nutrition.csv'):
        self.df = pd.read_csv(csv_path)
        self.df['ingredient'] = self.df['ingredient'].str.lower()
        self.all_ingredients = self.df['ingredient'].tolist()

    def get_facts(self, ingredients):
        result = {}
        missing = []
        for ing in ingredients:
            found = token_search(ing, self.all_ingredients)
            row = self.df[self.df['ingredient'] == found]
            if not row.empty:
                facts = row.iloc[0].drop('ingredient').to_dict()
                result[found.capitalize()] = facts
            else:
                missing.append(ing)
        return result, missing


class RecipeSearcher:
    def __init__(self, recipes_path='data/recipes_links.csv', ings_path='data/ingredients_nutrition.csv'):
        self.df = pd.read_csv(recipes_path)
        self._dv_cols = [c for c in self.df.columns if '_DV%' in c]
        ings_df = pd.read_csv(ings_path)
        self.all_ingredients = ings_df['ingredient'].tolist()
        self.all_ingredients = list(map(lambda x: x.lower(), self.all_ingredients))

    def find_similar(self, input_ingredients, n=3):
        found_set = {token_search(i, self.all_ingredients) for i in input_ingredients}
        n_ing = len(found_set)

        if n_ing <= 4:
            min_matches = 1
        else:
            min_matches = max(2, n_ing // 3)

        def calculate_score(row_ing_str):
            if pd.isna(row_ing_str): return 0
            row_set = {x.strip().lower() for x in str(row_ing_str).split(',')}
            return len(found_set.intersection(row_set))

        self.df['match_count'] = self.df['ingredient_list'].apply(calculate_score)
        top = self.df[self.df['match_count'] >= min_matches].sort_values(
            by=['match_count', 'rating'], ascending=False
        ).head(n)
        return top

    def generate_daily_menu(self):
        reasonable = self.df[
            self.df[self._dv_cols].max(axis=1) <= 200
        ].copy()

        meals = ['breakfast', 'lunch', 'dinner']

        candidates = {}
        for meal in meals:
            candidates[meal] = reasonable[
                (reasonable[meal] == 1.0) &
                (reasonable['ingredient_list'].notna()) &
                (reasonable['ingredient_list'].str.strip() != '')
            ].nlargest(20, 'rating')

        def find_valid_combos(threshold):
            valid = []
            for _, b in candidates['breakfast'].iterrows():
                for _, l in candidates['lunch'].iterrows():
                    for _, d in candidates['dinner'].iterrows():
                        total = b[self._dv_cols] + l[self._dv_cols] + d[self._dv_cols]
                        if total.max() <= threshold:
                            score = b['rating'] + l['rating'] + d['rating']
                            valid.append((score, b, l, d))
            return valid

        valid_combos = find_valid_combos(100)

        if not valid_combos:
            valid_combos = find_valid_combos(150)

        if valid_combos:
            max_score = max(s for s, *_ in valid_combos)
            top_combos = [(s, b, l, d) for s, b, l, d in valid_combos
                          if s >= max_score - 0.5]
            _, b, l, d = top_combos[np.random.randint(len(top_combos))]
            return {'BREAKFAST': b, 'LUNCH': l, 'DINNER': d}

        return {meal.upper(): candidates[meal].iloc[0]
                for meal in meals if not candidates[meal].empty}