from src.preprocessing import preprocess_text
from src.hybrid_model import expand_vibe_query, calculate_feature_match
from src.query_constraints import extract_constraints

def test_preprocess():
    assert preprocess_text("Fresh, WOODY! fragrance") == "fresh woody fragrance"
def test_vibe_expansion():
    text=expand_vibe_query("seductive fragrance")
    assert "amber" in text and "musk" in text
def test_feature_match():
    score,matched=calculate_feature_match("woody spicy","woody spicy perfume")
    assert score == 1 and "woody" in matched and "spicy" in matched
def test_budget():
    assert extract_constraints("under ₹3,000")["max_price"] == 3000
