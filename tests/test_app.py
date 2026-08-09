import unittest

from streamlit.testing.v1 import AppTest


class StreamlitSmokeTest(unittest.TestCase):
    def test_application_and_prediction(self):
        app = AppTest.from_file("app.py", default_timeout=90).run()
        self.assertEqual(len(app.exception), 0)
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        labels = [metric.label for metric in app.metric]
        self.assertTrue(any(label.startswith("Probabilidade de") for label in labels))


if __name__ == "__main__":
    unittest.main()
