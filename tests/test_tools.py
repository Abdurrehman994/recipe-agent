import pytest

from src.schema import Constraint, ConstraintUpdate, Objective, TurnExtraction
from src.tools import (RecipeDB, check, diagnose_constraints, find_ingredient,
                       get_recipe_details, search_recipes, validate_constraints)


def make_recipe(rid, title, lines, duration=20, calories=400, protein=20, **flags):
    from src.data_prep import diet_flags, ingredient_categories
    cats = set().union(*(ingredient_categories(l) for l in lines))
    r = {"recipe_id": rid, "title": title, "description": "", "duration_min": duration,
         "servings": 4, "serving_size_g": 300, "calories_kcal": calories, "protein_g": protein,
         "fat_g": 10, "saturated_fat_g": 3, "cholesterol_mg": 10, "sodium_mg": 300,
         "carbs_g": 40, "fiber_g": 4, "sugar_g": 5, "ingredients": [], "tags": [],
         "ingredients_with_quantities": lines, "ingredient_categories": sorted(cats),
         "courses": ["main-dish"], "is_main": True, "directions": ["Cook."],
         "average_rating": 4.5, "number_of_ratings": 10, "recipe_url": ""}
    r.update(diet_flags(cats))
    r.update(flags)
    return r


@pytest.fixture
def toy_db():
    return RecipeDB([
        make_recipe(1, "Mushroom Risotto", ["1 cup arborio rice", "200 g cremini mushrooms, sliced",
                                             "1 cup parmesan cheese"], duration=40),
        make_recipe(2, "Quick Chickpea Curry", ["1 can chickpeas", "1 onion", "1 cup coconut milk"],
                    duration=20, protein=12),
        make_recipe(3, "Chicken Stir Fry", ["2 chicken breasts", "1 tbsp soy sauce"],
                    duration=15, protein=35),
        make_recipe(4, "Eggplant Parmesan", ["1 large eggplant", "1 cup mozzarella cheese",
                                              "1 cup breadcrumbs"], duration=60, protein=18),
    ])


@pytest.fixture(scope="module")
def db():
    return RecipeDB.load()


# ---------------------------------------------------------------- schema
def test_constraint_validation():
    with pytest.raises(ValueError):
        Constraint("diet", "carnivore")
    with pytest.raises(ValueError):
        Constraint("max", "duration_min")
    assert Constraint("max", "duration_min", 25).value == 25.0
    assert Constraint("exclude", " Mushroom ").target == "mushroom"


def test_roundtrip_dicts():
    ext = TurnExtraction([ConstraintUpdate("add", Constraint("max", "calories_kcal", 500), True)],
                         [Objective("protein_g", "max")])
    assert TurnExtraction.from_dict(ext.to_dict()).to_dict() == ext.to_dict()


def test_describe():
    assert Constraint("max", "duration_min", 25).describe() == "time at most 25 min"
    assert Constraint("min", "protein_g", 30).describe() == "protein at least 30 g"


# ---------------------------------------------------------------- ingredient matching
def test_category_match_catches_variants(toy_db):
    assert find_ingredient(toy_db.get(1), "mushrooms") == ["200 g cremini mushrooms, sliced"]
    assert find_ingredient(toy_db.get(2), "mushroom") == []


def test_plain_term_match_handles_plurals(toy_db):
    assert find_ingredient(toy_db.get(2), "chickpea")
    assert find_ingredient(toy_db.get(2), "onions")
    assert not find_ingredient(toy_db.get(4), "egg")  # eggplant is not egg


def test_include_ignores_flavorings_but_exclude_does_not():
    r = make_recipe(9, "Vegetable Soup", ["4 cups chicken broth", "2 carrots"])
    assert not check(r, Constraint("include", "chicken"))[0]
    assert not check(r, Constraint("exclude", "chicken"))[0]
    r2 = make_recipe(10, "Roast", ["1 whole chicken", "1 cup chicken stock"])
    assert check(r2, Constraint("include", "chicken"))[0]


def test_diet_check_gives_evidence(toy_db):
    ok, evidence = check(toy_db.get(3), Constraint("diet", "vegetarian"))
    assert not ok and "chicken breasts" in evidence
    assert check(toy_db.get(2), Constraint("diet", "vegan"))[0]  # coconut milk is not dairy


def test_numeric_check(toy_db):
    ok, evidence = check(toy_db.get(1), Constraint("max", "duration_min", 25))
    assert not ok and "40" in evidence
    assert check(toy_db.get(3), Constraint("min", "protein_g", 30))[0]


# ---------------------------------------------------------------- tools on toy data
def test_search_filters_and_ranks(toy_db):
    cons = [Constraint("diet", "vegetarian"), Constraint("max", "duration_min", 45)]
    res = search_recipes(toy_db, constraints=cons)
    assert res["total_matches"] == 2
    assert {r["recipe_id"] for r in res["results"]} == {1, 2}
    res = search_recipes(toy_db, constraints=cons + [Constraint("exclude", "mushroom")])
    assert [r["recipe_id"] for r in res["results"]] == [2]


def test_search_objective_ordering(toy_db):
    res = search_recipes(toy_db, objectives=[Objective("duration_min", "min")])
    assert [r["recipe_id"] for r in res["results"]] == [3, 2, 1, 4]
    res = search_recipes(toy_db, objectives=[{"target": "protein_g", "direction": "max"}])
    assert res["results"][0]["recipe_id"] == 3


def test_search_accepts_dicts_and_query(toy_db):
    res = search_recipes(toy_db, query="curry", constraints=[{"kind": "diet", "target": "vegan"}])
    assert res["results"][0]["title"] == "Quick Chickpea Curry"


def test_validate_reports_each_check(toy_db):
    out = validate_constraints(toy_db, 1, [{"kind": "diet", "target": "vegetarian"},
                                           {"kind": "exclude", "target": "mushroom"}])
    assert not out["all_satisfied"]
    assert [c["satisfied"] for c in out["checks"]] == [True, False]


def test_diagnose_infeasible(toy_db):
    cons = [Constraint("diet", "vegetarian"), Constraint("max", "duration_min", 10)]
    out = diagnose_constraints(toy_db, cons)
    assert out["total_matches"] == 0
    time_entry = out["per_constraint"][1]
    assert time_entry["matches_if_dropped"] == 3 and time_entry["closest_available"] == 20


def test_unknown_recipe_id(toy_db):
    with pytest.raises(KeyError):
        get_recipe_details(toy_db, 999)


# ---------------------------------------------------------------- real database
def test_real_db_loads(db):
    assert len(db) == 5000


def test_spec_example_is_feasible(db):
    cons = [Constraint("diet", "vegetarian"), Constraint("course", "main-dish"),
            Constraint("max", "duration_min", 25), Constraint("exclude", "mushroom")]
    res = search_recipes(db, constraints=cons, k=3)
    assert res["total_matches"] > 50
    for r in res["results"]:
        assert validate_constraints(db, r["recipe_id"], cons)["all_satisfied"]


def test_search_results_always_satisfy_constraints(db):
    cons = [Constraint("diet", "gluten_free"), Constraint("max", "calories_kcal", 400),
            Constraint("min", "protein_g", 25), Constraint("include", "chicken")]
    res = search_recipes(db, query="spicy", constraints=cons, k=10)
    assert res["results"]
    for r in res["results"]:
        assert validate_constraints(db, r["recipe_id"], cons)["all_satisfied"]
