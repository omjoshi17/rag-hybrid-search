import unittest

from src.generation.citation_judge import CitationJudge, calculate_confidence, extract_citation_numbers
from src.generation.prompts import build_grounded_messages


class TrueJudgeLLM:
    def complete(self, messages):
        return "True"


class GenerationTests(unittest.TestCase):
    def test_prompt_formats_numbered_documents(self) -> None:
        chunks = [{"content": "Reset links expire after 30 minutes.", "metadata": {"source": "doc.md"}}]

        messages, documents = build_grounded_messages("When do reset links expire?", chunks)

        self.assertIn("[Document 1]", messages[1]["content"])
        self.assertEqual(documents[0]["document_index"], 1)

    def test_extracts_short_and_document_citations(self) -> None:
        self.assertEqual(extract_citation_numbers("Claim [1]. Another claim [Document 2]."), [1, 2])

    def test_judge_verifies_cited_sentence(self) -> None:
        judge = CitationJudge(TrueJudgeLLM())
        documents = [{"document_index": 1, "content": "Reset links expire after 30 minutes.", "metadata": {"source": "doc.md"}}]

        result = judge.verify_answer("Reset links expire after 30 minutes [Document 1].", documents)

        self.assertEqual(result["citation_accuracy"], 1.0)
        self.assertEqual(result["supported_citations"], 1)

    def test_confidence_combines_relevance_and_citation_accuracy(self) -> None:
        chunks = [{"cross_encoder_score": 1.0}]
        verification = {"citation_accuracy": 0.5}

        confidence = calculate_confidence(chunks, verification)

        self.assertGreater(confidence["composite_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
