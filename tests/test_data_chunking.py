import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prepare_text_data import discover_input_files, iter_texts, write_token_chunks  # noqa: E402


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=False, verbose=False):
        del add_special_tokens, truncation, verbose
        return {"input_ids": [ord(char) % 251 for char in text]}


class DataChunkingTest(unittest.TestCase):
    def test_discovers_supported_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("alpha")
            (root / "b.jsonl").write_text('{"text":"beta"}\n')
            (root / "ignore.md").write_text("ignored")
            files = discover_input_files(input_dir=root)
            self.assertEqual([path.name for path in files], ["a.txt", "b.jsonl"])

    def test_iter_texts_from_jsonl_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jsonl = root / "rows.jsonl"
            csv_path = root / "rows.csv"
            jsonl.write_text('{"text":"alpha"}\n{"text":"beta"}\n')
            csv_path.write_text("text\ncharlie\n")

            self.assertEqual(list(iter_texts(jsonl)), ["alpha", "beta"])
            self.assertEqual(list(iter_texts(csv_path)), ["charlie"])

    def test_write_token_chunks_drops_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sample.txt"
            output = root / "chunks.jsonl"
            source.write_text("abcdefghij")

            stats = write_token_chunks(
                files=[source],
                tokenizer=FakeTokenizer(),
                jsonl_path=output,
                max_seq_len=4,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]

            self.assertEqual(stats["chunks"], 2)
            self.assertEqual(stats["tokens"], 8)
            self.assertEqual(stats["dropped_tail_tokens"], 2)
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[0]["input_ids"]), 4)
            self.assertEqual(rows[0]["labels"], rows[0]["input_ids"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
