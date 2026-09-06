import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fake_news import build_pipeline, clean_text


class FakeNewsTests(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("Hello!!! https://example.com"), "hello")

    def test_pipeline_structure(self):
        model = build_pipeline()
        self.assertIn("tfidf", model.named_steps)
        self.assertIn("classifier", model.named_steps)


if __name__ == "__main__":
    unittest.main()
