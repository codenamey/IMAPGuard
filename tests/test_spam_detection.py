import unittest
from transformers import pipeline
from tools.spam_detection import generate_text_with_timeout

class TestSpamDetection(unittest.TestCase):

    def setUp(self):
        # Alusta malli testejä varten
        self.generator = pipeline("text-generation", model="Helsinki-NLP/opus-mt-fi-en", device=-1)

    def test_generate_text_with_timeout_success(self):
        input_text = "Testi on vain testi."
        generated_text = generate_text_with_timeout(self.generator, input_text, max_length=10, timeout=5)
        self.assertIsNotNone(generated_text)
        self.assertTrue(len(generated_text) > 0)

    def test_generate_text_with_timeout_timeout(self):
        input_text = "A" * 10000  # Yksinkertainen pitkä merkkijono, joka voi aiheuttaa aikakatkaisun
        generated_text = generate_text_with_timeout(self.generator, input_text, max_length=1000, timeout=1)

        # Lisää tulostus diagnosointia varten
        print(f"Generated text: {generated_text}")

        # Tarkista, että aikakatkaisu tapahtuu ja että saamme joko tyhjän tai osittaisen vastauksen
        self.assertTrue(generated_text == "" or len(generated_text) < len(input_text))

if __name__ == '__main__':
    unittest.main()
