from pathlib import Path
import tempfile
import unittest

from src.data.loader import load_documents


class LoaderTests(unittest.TestCase):
    def test_load_markdown_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs_dir = Path(temp_dir)
            (docs_dir / "handbook.md").write_text("# Handbook\n\nWelcome aboard.", encoding="utf-8")

            records = load_documents(docs_dir)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["metadata"]["source"], "handbook.md")
            self.assertEqual(records[0]["metadata"]["content_type"], "markdown")
            self.assertIn("Welcome aboard.", records[0]["content"])


if __name__ == "__main__":
    unittest.main()
