
import unittest

from src.data_loader import load_perfumes
from src.query_constraints import extract_constraints, apply_constraints


class TestQueryConstraints(unittest.TestCase):

    def setUp(self):
        self.perfumes = load_perfumes()

    def test_budget_under_3000(self):
        constraints = extract_constraints("Perfume under 3000")
        results = apply_constraints(self.perfumes, constraints)

        self.assertEqual(results["name"].tolist(), ["Fresh Green"])

    def test_summer_and_not_sweet(self):
        constraints = extract_constraints(
            "Summer perfume under 3000, not sweet"
        )
        results = apply_constraints(self.perfumes, constraints)

        self.assertEqual(results["name"].tolist(), ["Fresh Green"])

    def test_no_matches_under_2000(self):
        constraints = extract_constraints("Perfume under 2000")
        results = apply_constraints(self.perfumes, constraints)

        self.assertTrue(results.empty)

    def test_winter_under_5000(self):
        constraints = extract_constraints("Perfume for winter under 5000")
        results = apply_constraints(self.perfumes, constraints)

        self.assertEqual(
            set(results["name"]),
            {"Sweet Amber", "Woody Spice"},
        )


if __name__ == "__main__":
    unittest.main()
