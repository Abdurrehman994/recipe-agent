# Data notes: HUMMUS exploration and normalization

Source file: `data/pp_recipes/pp_recipes.csv` (HUMMUS, 507,335 recipes, 37 columns).
Pipeline: `python -m src.data_prep` → `data/processed/recipes.jsonl` (5,000 recipes) and
`data/processed/prep_report.json` (counts from each filtering step).

## 1. Nutrition is per serving, not per 100 g

Columns carry explicit units (`calories [cal]`, `protein [g]`, `sodium [mg]`, …) plus
`servingSize [g]` and `servingsPerRecipe`. Evidence that values are **per serving**:

| Test | Result | Interpretation |
|---|---|---|
| fat + carbs + protein > 100 g | 22.5% of recipes | impossible if values were per 100 g |
| fat + carbs + protein > servingSize | 1.6% of recipes | consistent with per serving (residue = data errors) |
| median calories / servingSize | 1.63 kcal/g (p99: 7.1) | realistic energy density for food |
| (4·carbs + 4·protein + 9·fat) / calories | median 1.015 | energy and macros are internally consistent |
| corr(calories, servingSize) | 0.30 | calories scale with portion size |

Therefore numeric constraints such as "under 500 calories" are applied per serving.
`duration` is in minutes (98% of recipes tagged `15-minutes-or-less` have duration ≤ 15).

## 2. `ingredient_food_kg_names` is misaligned and not used

In the raw file this column lists another recipe's ingredients for most rows. For example,
"Cajun Chicken Salad" lists flour, yeast and milk (the bread recipe in the next row). Only
~22% of recipes share ≥30% of their KG ingredient names with their own ingredient text.
The raw `ingredients` column (`{section: [(text, quantity), …]}`) matches titles, tags
and nutrition, so all ingredient processing uses it:

- Meat-containing recipes have a median of 25.7 g protein, against 5.4 g for the rest.
- Recipes tagged `desserts` have a median of 24.5 g sugar, against 4.8 g for the rest.
- 98% of recipes with a meat word in the title also have meat in the ingredient text.

Consequence for the agent: ingredient matching works on free-text ingredient lines.

## 3. Missing values and anomalies (full dataset)

| Field | Issue |
|---|---|
| `tags` | 54.5% missing |
| `serves` | 14.1% missing (`servingsPerRecipe` is complete and used instead) |
| `duration` | 1.4% missing; max 1,425 min |
| `servingSize [g]` | 1,307 rows ≤ 0, max 1.6 million g |
| `calories [cal]` | 2,950 rows = 0, max 612,854 |
| `normalization_comment` | 2.8% flagged (`*`, `**`) as uncertain nutrition normalization |
| raw ingredient lists | 134 contain bare `nan`; `$template2$` placeholders in text |

## 4. Filters (applied in this order)

1. 1 ≤ duration ≤ 600 min; tags and ingredients present
2. no `normalization_comment` flag
3. 20 g ≤ servingSize ≤ 1,500 g; 10 ≤ calories ≤ 2,500 kcal per serving
4. fat + carbs + protein ≤ serving size (physically possible)
5. ≥ 3 ratings (some community validation)
6. ≥ 3 parseable ingredient lines

→ 86,368 candidates. Stratified sample with seed 42: 60% main dishes (`main-dish` or
`one-dish-meal` tag) and 40% other courses, giving **5,000 recipes**.

## 5. Ingredient normalization and aliases

There are 17k+ distinct raw ingredient strings in the full data. Normalization
(`normalize_ingredient`) does the following:

- drops parenthetical notes, `$templateN$` junk, text after the first comma or "or"
- removes descriptors (fresh, chopped, large, …), unit words (cloves, package, can, …) and numbers
- applies an alias table (garbanzo beans → chickpeas, scallions → green onion, …)

This leaves 5,836 distinct names in the 5,000-recipe set. The result is approximate, so
**constraints are not matched on these names alone**. "Avoid mushrooms" uses the keyword
category `mushroom`, which also catches cremini, shiitake, porcini and cream of mushroom soup.

## 6. Dietary flags are derived from ingredients

Categories (`meat, poultry, fish, shellfish, dairy, egg, gluten, nuts, honey, mushroom`) are
assigned by word-boundary keyword matching. Plurals are allowed, and exception lists handle
look-alikes (eggplant ≠ egg, peanut butter ≠ dairy, rice flour ≠ gluten,
vegetarian bacon ≠ meat, poultry seasoning ≠ poultry, …). Three data quirks shaped the rules:

- **Ingredient in the quantity field.** Some raw entries look like `('each', '6 salmon fillets
  about 6 oz')`, so categories are computed on the full line (quantity + text), not the text
  alone. Before this fix, "Grilled Salmon With Horseradish Sauce" was flagged vegetarian.
- **Main protein missing from the list.** "Honey-Mustard Tilapia" lists no fish at all. As a
  conservative safety net, an animal word in the *title* also counts, unless the title marks
  a non-dish or substitute (sauce, marinade, gravy, meatless, veggie, "-style", …).
- **Parenthetical notes are ignored** ("white wine (or beef broth)", "(Regulate heat by
  discarding the chile's ribs)"), so optional substitutions do not flip a flag.

The flags follow from the categories:

- vegetarian = no meat, poultry, fish or shellfish
- pescatarian = no meat or poultry
- vegan = vegetarian, no dairy, egg or honey
- gluten_free / dairy_free / nut_free / egg_free

Audit:

- Agreement with the HUMMUS `vegetarian` tag: of 738 tagged recipes, 721 (97.7%) are also
  derived vegetarian. The 17 disagreements are mostly real catches (chicken broth in lentil
  dishes, anchovy-containing Worcestershire sauce).
- 1,261 derived-vegetarian recipes carry no tag, because the tags are incomplete.
- Of derived-vegetarian recipes whose title contains an animal word, all 39 remaining cases
  were checked by hand and are correct: sauces *for* chicken or fish, goat cheese, scalloped
  potatoes, veggie sausages.

Limitations: gluten detection is keyword-based (oats and generic "sauce" are not flagged), and
an ingredient line offering a meat *alternative* outside parentheses counts as non-vegetarian.

## 7. Resulting database (5,000 recipes)

- vegetarian 39.6%, vegan 10.3%, gluten-free 53.5%, dairy-free 39.8%, nut-free 89.8%
- 60% main dishes; 9.6% contain mushrooms
- duration: median 40 min (p10 = 15, p90 = 140)
- calories per serving: median ≈ 350 kcal; protein median ≈ 17 g
- example feasibility: vegetarian main dish ≤ 25 min → 128 recipes; without mushrooms → 116;
  vegan + gluten-free main dish ≤ 15 min → 5 (useful for near-infeasible scenarios)
