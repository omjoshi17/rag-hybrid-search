import unittest

from src.data.chunker import ChunkingConfig, chunk_documents, markdown_sections


class ChunkerTests(unittest.TestCase):
    def test_markdown_sections_track_heading_paths(self) -> None:
        text = "# Root\nIntro\n\n## Child\nDetails"

        sections = markdown_sections(text)

        self.assertEqual(sections[0][1], "Root")
        self.assertEqual(sections[1][1], "Root > Child")

    def test_fixed_chunking_adds_metadata(self) -> None:
        documents = [
            {
                "content": "abcdefghijklmnopqrstuvwxyz",
                "metadata": {"source": "alpha.txt", "page": None, "content_type": "text"},
            }
        ]

        chunks = chunk_documents(documents, ChunkingConfig(strategy="fixed", chunk_size=10, chunk_overlap=2))

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0]["metadata"]["strategy"], "fixed")
        self.assertIn("chunk_id", chunks[0]["metadata"])
        self.assertEqual(chunks[0]["metadata"]["char_count"], len(chunks[0]["content"]))


if __name__ == "__main__":
    unittest.main()
