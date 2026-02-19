import os
import tempfile
import unittest

from subtitles import create_ass_subtitles_from_words, group_words_into_phrases


class TestSubtitlePhrases(unittest.TestCase):
    def test_group_words_split_by_gap_and_punctuation_and_limit(self):
        words = [
            {"word": "Hello", "start": 0.00, "end": 0.20},
            {"word": "world!", "start": 0.22, "end": 0.45},
            {"word": "Next", "start": 1.20, "end": 1.35},
            {"word": "phrase", "start": 1.38, "end": 1.62},
            {"word": "now", "start": 1.65, "end": 1.90},
            {"word": "again", "start": 1.92, "end": 2.10},
        ]

        phrases = group_words_into_phrases(
            words,
            max_words_per_phrase=3,
            max_gap_sec=0.45,
            max_phrase_duration=1.2
        )

        self.assertEqual(len(phrases), 3)
        self.assertEqual(phrases[0]["text"], "Hello world!")
        self.assertEqual(phrases[1]["text"], "Next phrase now")
        self.assertEqual(phrases[2]["text"], "again")
        for phrase in phrases:
            self.assertGreater(phrase["end"], phrase["start"])

    def test_create_ass_with_alignment_and_escaping(self):
        words = [
            {"word": "Text", "start": 0.00, "end": 0.20},
            {"word": "{danger}", "start": 0.21, "end": 0.40},
            {"word": "slash\\", "start": 0.41, "end": 0.65},
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            ass_file = create_ass_subtitles_from_words(
                words,
                output_dir=tmp_dir,
                style_name="subtitle",
                position="top",
                max_words_per_phrase=3
            )

            self.assertIsNotNone(ass_file)
            self.assertTrue(os.path.exists(ass_file))

            with open(ass_file, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Style: Subtitle", content)
            self.assertIn(",8,", content)  # Alignment=8 (top)
            self.assertIn("Dialogue:", content)
            self.assertIn(r"\{danger\}", content)
            self.assertIn(r"slash\\", content)


if __name__ == "__main__":
    unittest.main()
