"""Build the cleaned ~5,000-recipe database from the raw HUMMUS file.

Usage:  python -m src.data_prep
Input:  data/pp_recipes/pp_recipes.csv   (raw HUMMUS, ~507k recipes)
Output: data/processed/recipes.jsonl     (one recipe per line)
        data/processed/prep_report.json  (counts from every filtering step)

Key decisions (see docs/data_notes.md for the evidence):
  * Nutrition values are PER SERVING, `duration` is total minutes.
  * Ingredients come from the raw `ingredients` column. `ingredient_food_kg_names` is
    misaligned with the recipes in the raw file (it belongs to other rows) and is NOT used.
  * Dietary flags are derived from ingredient text; the HUMMUS tags are kept for reference.
"""
import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "pp_recipes" / "pp_recipes.csv"
OUT_DIR = ROOT / "data" / "processed"
N_RECIPES = 5000
MAIN_SHARE = 0.6  # share of main dishes, since most scenarios ask for meals
SEED = 42

NUTRITION = {  # raw column -> clean name (all per serving)
    "calories [cal]": "calories_kcal",
    "totalFat [g]": "fat_g",
    "saturatedFat [g]": "saturated_fat_g",
    "cholesterol [mg]": "cholesterol_mg",
    "sodium [mg]": "sodium_mg",
    "totalCarbohydrate [g]": "carbs_g",
    "dietaryFiber [g]": "fiber_g",
    "sugars [g]": "sugar_g",
    "protein [g]": "protein_g",
}
USECOLS = [
    "recipe_id", "title", "description", "duration", "directions", "ingredients",
    "servingsPerRecipe", "servingSize [g]", "tags", "average_rating", "number_of_ratings",
    "normalization_comment", "recipe_url",
    "who_score", "fsa_score", "nutri_score", *NUTRITION,
]

# ---------------------------------------------------------------- ingredients
DESCRIPTORS = {
    "fresh", "freshly", "chopped", "diced", "sliced", "minced", "large", "small", "medium",
    "dried", "ground", "whole", "frozen", "canned", "cooked", "raw", "boneless", "skinless",
    "shredded", "grated", "crushed", "finely", "thinly", "lean", "extra", "virgin", "light",
    "low-fat", "fat-free", "low-sodium", "reduced", "unsalted", "salted", "sweetened",
    "unsweetened", "packed", "softened", "melted", "cold", "warm", "hot", "baby", "organic",
}
UNIT_WORDS = {
    "head", "heads", "clove", "cloves", "stalk", "stalks", "package", "packages", "pkg", "can",
    "cans", "cup", "cups", "pinch", "dash", "slice", "slices", "sprig", "sprigs", "bunch",
    "piece", "pieces", "jar", "bottle", "box", "bag", "envelope", "spray", "tablespoon",
    "tablespoons", "teaspoon", "teaspoons", "lb", "lbs", "ounce", "ounces", "oz", "of",
}
ALIASES = {  # normalized form -> canonical name
    "all - purpose flour": "flour", "all-purpose flour": "flour", "plain flour": "flour",
    "garlic clove": "garlic", "garlic cloves": "garlic", "garlic powder": "garlic powder",
    "egg": "egg", "eggs": "egg", "egg white": "egg white", "egg whites": "egg white",
    "onions": "onion", "scallion": "green onion", "scallions": "green onion",
    "green onions": "green onion", "spring onions": "green onion",
    "tomatoes": "tomato", "potatoes": "potato", "carrots": "carrot",
    "mushrooms": "mushroom", "black pepper": "pepper", "salt and pepper": "salt and pepper",
    "vanilla extract": "vanilla", "confectioners' sugar": "powdered sugar",
    "icing sugar": "powdered sugar", "cilantro": "coriander", "zucchini": "courgette",
    "garbanzo beans": "chickpeas", "chick peas": "chickpeas",
}


def strip_notes(text: str) -> str:
    """Drop parenthetical notes like '(optional)' or '(or use beef broth)' and template junk."""
    s = re.sub(r"\(.*?\)", " ", str(text).lower())
    s = re.sub(r"\$template\d+\$|[()]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_ingredient(name: str) -> str:
    """'large tomatoes, chopped' -> 'tomato'; keeps the part before the first comma or 'or'."""
    s = strip_notes(name).split(",")[0]
    s = re.split(r"or", s)[0]
    s = re.sub(r"\s+", " ", s.replace(" - ", "-")).strip(" ,.;:")
    if s in ALIASES:
        return ALIASES[s]
    words = [w for w in s.split() if w not in DESCRIPTORS and w not in UNIT_WORDS
             and not re.fullmatch(r"[\d/.-]+", w)]
    s = " ".join(words) or s
    return ALIASES.get(s, s)


# Keyword categories used for diet flags and for "avoid X" style constraints.
# Matching is on word boundaries; EXCEPTIONS remove known false positives.
CATEGORIES = {
    "meat": r"beef|steak|pork|bacon|ham|sausages?|lamb|veal|venison|prosciutto|salami|pepperoni|"
            r"chorizo|pancetta|meatballs?|hot dogs?|bratwurst|ground chuck|sirloin|ribs?|brisket|"
            r"lard|gelatin|beef broth|beef stock",
    "poultry": r"chicken|turkey|duck|goose|cornish hens?|chicken broth|chicken stock",
    "fish": r"fish|salmon|tuna|cod|tilapia|halibut|trout|anchov(y|ies)|sardines?|haddock|"
            r"mackerel|catfish|snapper|swordfish|worcestershire sauce|fish sauce",
    "shellfish": r"shrimps?|prawns?|crab|crabmeat|lobster|scallops?|clams?|mussels?|oysters?|squid",
    "dairy": r"milk|butter|cheese|cheddar|parmesan|mozzarella|cream|yogurt|yoghurt|ghee|"
             r"buttermilk|ricotta|feta|whey|half-and-half|sour cream",
    "egg": r"eggs?|egg whites?|egg yolks?|mayonnaise",
    "gluten": r"flour|wheat|bread|breadcrumbs|pasta|spaghetti|macaroni|noodles|barley|rye|"
              r"couscous|soy sauce|beer|crackers?|tortillas?|bisquick|cake mix|biscuits?|"
              r"croutons|panko|semolina|bulgur|pie crust|puff pastry",
    "nuts": r"almonds?|walnuts?|pecans?|cashews?|pistachios?|hazelnuts?|macadamia|peanuts?|"
            r"peanut butter|pine nuts|nuts",
    "honey": r"honey",
    "mushroom": r"mushrooms?|porcini|shiitake|portabell?as?|cremini|chanterelles?",
}
_MEAT_SUBSTITUTE = (r"(vegetarian|veggie|vegan|meatless|soy|tofu|plant-based) (\w+ )?"
                    r"(bacon|sausages?|burgers?|chicken|turkey|ground beef|crumbles|chorizo|"
                    r"hot dogs?|pepperoni|ham)")
EXCEPTIONS = {
    "dairy": r"coconut milk|coconut cream|almond milk|soy milk|soymilk|rice milk|oat milk|"
             r"peanut butter|almond butter|cashew butter|apple butter|butternut|cream of tartar|"
             r"cocoa butter|nut butter|butter beans|ice cream cone",
    "egg": r"eggplants?",
    "gluten": r"rice flour|almond flour|coconut flour|corn flour|cornflour|corn tortillas?|"
              r"rice noodles|gluten-free|tamari|buckwheat|chickpea flour|potato flour|tapioca flour|"
              r"breadfruit",
    "meat": rf"celery ribs?|hot dog buns?|hamburger buns?|{_MEAT_SUBSTITUTE}",
    "poultry": _MEAT_SUBSTITUTE,
    "nuts": r"nutmeg|butternut|coconut|doughnuts?|donuts?|water chestnuts?",
    "fish": r"fish sauce substitute",
}
_CAT_RE = {k: re.compile(rf"\b({v})\b") for k, v in CATEGORIES.items()}
_EXC_RE = {k: re.compile(rf"\b({v})\b") for k, v in EXCEPTIONS.items()}


def ingredient_categories(name: str) -> set[str]:
    s = strip_notes(name)
    cats = set()
    for cat, rx in _CAT_RE.items():
        stripped = _EXC_RE[cat].sub(" ", s) if cat in _EXC_RE else s
        if rx.search(stripped):
            cats.add(cat)
    return cats


def diet_flags(categories: set[str]) -> dict[str, bool]:
    animal = {"meat", "poultry", "fish", "shellfish"}
    vegetarian = not (categories & animal)
    return {
        "vegetarian": vegetarian,
        "pescatarian": not (categories & {"meat", "poultry"}),
        "vegan": vegetarian and not (categories & {"dairy", "egg", "honey"}),
        "dairy_free": "dairy" not in categories,
        "gluten_free": "gluten" not in categories,
        "nut_free": "nuts" not in categories,
        "egg_free": "egg" not in categories,
    }


# ---------------------------------------------------------------- parsing helpers
def safe_list(s):
    """Parse a stringified Python list; the raw file sometimes contains bare `nan`."""
    if not isinstance(s, str):
        return []
    try:
        val = ast.literal_eval(s.replace("[nan, ", "[").replace(", nan", ""))
    except (ValueError, SyntaxError):
        return []
    return [x for x in val if isinstance(x, str) and x.strip()]


def parse_ingredients(s):
    """Raw `ingredients` column: {'section': [(text, 'qty time(s) unit'), ...]} -> [(text, qty)]."""
    try:
        sections = ast.literal_eval(s.replace("nan", "''"))
    except (ValueError, SyntaxError, TypeError, AttributeError):
        return []
    out = []
    for items in sections.values():
        for text, qty in items:
            if not isinstance(text, str) or not isinstance(qty, str):
                continue
            text = re.sub(r"\s+", " ", text.replace("$template2$", "")).strip()
            qty = re.sub(r"\s+", " ", qty.replace("time(s)", "")).strip()
            if text:
                out.append((text, qty))
    return out


COURSE_TAGS = ["main-dish", "side-dishes", "desserts", "breakfast", "lunch", "appetizers",
               "salads", "soups-stews", "breads", "beverages", "snacks", "brunch"]


# ---------------------------------------------------------------- pipeline
def load_candidates(report):
    chunks = []
    total = 0
    for chunk in pd.read_csv(RAW, usecols=USECOLS, chunksize=50_000, low_memory=False):
        total += len(chunk)
        macros = chunk["totalFat [g]"] + chunk["totalCarbohydrate [g]"] + chunk["protein [g]"]
        keep = (
            chunk["duration"].between(1, 600)
            & chunk["ingredients"].notna()
            & chunk["tags"].notna()
            & chunk["normalization_comment"].isna()       # flagged as uncertain by HUMMUS
            & chunk["servingSize [g]"].between(20, 1500)
            & chunk["calories [cal]"].between(10, 2500)
            & (macros <= chunk["servingSize [g]"])          # physically possible per serving
            & (chunk["number_of_ratings"] >= 3)             # some community validation
        )
        chunks.append(chunk[keep])
    df = pd.concat(chunks, ignore_index=True)
    report["raw_rows"] = total
    report["after_quality_filters"] = len(df)
    return df


def enrich(df):
    df = df.copy()
    df["tags"] = df["tags"].map(safe_list)
    parsed = df["ingredients_raw"].map(parse_ingredients)
    df["ingredient_texts"] = parsed.map(lambda xs: [text for text, _ in xs])
    df["ingredients_with_quantities"] = parsed.map(lambda xs: [f"{q} {t}".strip() for t, q in xs])
    df = df[df["ingredient_texts"].map(len) >= 3]
    df["ingredients"] = df["ingredient_texts"].map(
        lambda xs: sorted({normalize_ingredient(x) for x in xs} - {"", "or", "and", "each"}))
    df["ingredient_categories"] = df["ingredient_texts"].map(
        lambda xs: sorted(set().union(*(ingredient_categories(x) for x in xs))))
    flags = df["ingredient_categories"].map(lambda c: diet_flags(set(c)))
    df = pd.concat([df, pd.DataFrame(flags.tolist(), index=df.index)], axis=1)
    df["courses"] = df["tags"].map(lambda t: [c for c in COURSE_TAGS if c in t])
    df["is_main"] = df["tags"].map(lambda t: "main-dish" in t or "one-dish-meal" in t)
    return df


def sample(df, report):
    rng = np.random.default_rng(SEED)
    main, other = df[df["is_main"]], df[~df["is_main"]]
    n_main = min(len(main), int(N_RECIPES * MAIN_SHARE))
    picked = pd.concat([
        main.sample(n_main, random_state=rng.integers(1e9)),
        other.sample(N_RECIPES - n_main, random_state=rng.integers(1e9)),
    ])
    report["after_ingredient_parse"] = len(df)
    report["sampled"] = len(picked)
    return picked.sort_values("recipe_id")


def to_records(df):
    df = df.copy()
    df["directions"] = df["directions"].map(safe_list)
    df = df.rename(columns={**NUTRITION, "servingSize [g]": "serving_size_g",
                            "servingsPerRecipe": "servings", "duration": "duration_min"})
    keep = ["recipe_id", "title", "description", "duration_min", "servings", "serving_size_g",
            *NUTRITION.values(), "ingredients", "ingredients_with_quantities",
            "ingredient_categories", "vegetarian", "pescatarian", "vegan", "dairy_free",
            "gluten_free", "nut_free", "egg_free", "courses", "is_main", "tags", "directions",
            "average_rating", "number_of_ratings", "who_score", "fsa_score", "nutri_score",
            "recipe_url"]
    df = df[keep]
    df["description"] = df["description"].fillna("")
    for col in NUTRITION.values():
        df[col] = df[col].round(1)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {}
    df = load_candidates(report)
    df = df.rename(columns={"ingredients": "ingredients_raw"})
    df = enrich(df)
    df = sample(df, report)
    out = to_records(df)
    out.to_json(OUT_DIR / "recipes.jsonl", orient="records", lines=True, force_ascii=False)

    for col in ["vegetarian", "vegan", "gluten_free", "dairy_free", "nut_free", "is_main"]:
        report[f"share_{col}"] = round(float(out[col].mean()), 3)
    report["duration_min_quantiles"] = out["duration_min"].quantile([.1, .25, .5, .75, .9]).to_dict()
    report["share_with_mushroom"] = round(float(out["ingredient_categories"].map(
        lambda c: "mushroom" in c).mean()), 3)
    (OUT_DIR / "prep_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
