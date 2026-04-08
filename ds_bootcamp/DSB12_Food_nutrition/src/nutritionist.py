#!/usr/bin/env python
import sys
import argparse
from recipes import Predictor, NutritionFacts, RecipeSearcher

def main():
    parser = argparse.ArgumentParser(description="AI Nutritionist & Recipe Advisor")
    parser.add_argument('ingredients', nargs='*', help="List of ingredients separated by spaces or commas")
    parser.add_argument('--menu', action='store_true', help="Generate a random healthy daily menu")
    args = parser.parse_args()

    if args.ingredients:
        input_str = " ".join(args.ingredients)
        if ',' in input_str:
            ingredients = [i.strip().lower() for i in input_str.split(',') if i.strip()]
        else:
            ingredients = [i.strip().lower() for i in input_str.split() if i.strip()]
    else:
        ingredients = []

    predictor = Predictor()
    nutrition = NutritionFacts()
    searcher = RecipeSearcher()

    # --- BONUS: DAILY MENU ---
    if args.menu:
        menu = searcher.generate_daily_menu()
        for meal, data in menu.items():
            print(f"\n{meal}")
            print("-" * 21)
            print(f"{data['title']} (rating: {data['rating']})")
            print("Ingredients:")
            ing_list = data['ingredient_list']
            if pd.notna(ing_list) and str(ing_list).strip() not in ('', 'nan'):
                for ing in str(ing_list).split(', '):
                    print(f"- {ing}")
            else:
                print("- (no ingredient data available)")
            print("Nutrients:")
            dv_cols = [c for c in data.index if '_DV%' in str(c)]
            for col in dv_cols:
                val = data[col]
                if pd.notna(val) and val > 0:
                    print(f"- {col.replace('_DV%', '').lower()}: {val:.0f}%")
            print(f"URL: {data['url']}")
        return

    if not ingredients:
        print("Please provide ingredients or use --menu")
        return

    _, missing = nutrition.get_facts(ingredients)

    if len(missing) > len(ingredients) / 2:
        print(f"The following ingredients are missing in our database: {', '.join(missing)}")
        return

    # --- I. OUR FORECAST ---
    forecast = predictor.predict(ingredients)
    print("\nI. OUR FORECAST")
    if forecast == "bad":
        print("You might find it tasty, but in our opinion, it is a bad idea to have a dish with that list of ingredients.")
    elif forecast == "so-so":
        print("This combination is okay, but we've seen better. Use with caution!")
    else:
        print("Great choice! These ingredients will make a fantastic dish.")

    # --- II. NUTRITION FACTS ---
    facts, missing = nutrition.get_facts(ingredients)
    print("\nII. NUTRITION FACTS")
    if missing:
        print(f"(Note: The following ingredients are missing in our database: {', '.join(missing)})")
    for ing, values in facts.items():
        print(f"\n{ing}")
        for name, val in values.items():
            print(f"{name.replace('_DV%', '')} - {val:.0f}% of Daily Value")

    # --- III. TOP-3 SIMILAR RECIPES ---
    similar = searcher.find_similar(ingredients)
    print("\nIII. TOP-3 SIMILAR RECIPES:")
    if similar.empty:
        print("There are no similar recipes in our database.")
    else:
        for _, row in similar.iterrows():
            print(f"- {row['title'].rstrip()}, rating: {row['rating']}, URL: {row['url']}")

import pandas as pd

if __name__ == "__main__":
    main()