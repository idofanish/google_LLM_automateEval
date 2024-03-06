import unittest

from google_LLM_automateEval.test_assistant import *


class LLM_EVAL_TestCases(unittest.TestCase):

	def test_science_quiz(self):
		question = "Generate a quiz about science."
		expected_subjects = ["davinci", "telescope", "physics", "curie"]
		eval_expected_words(
			system_message,
			question,
			expected_subjects)

	def test_geography_quiz(self):
		question = "Generate a quiz about geography."
		expected_subjects = ["paris", "france", "louvre"]
		eval_expected_words(
			system_message,
			question,
			expected_subjects)