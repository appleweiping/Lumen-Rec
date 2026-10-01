import json
import subprocess
import sys

from src.confrec.pseudonymize import pseudonym, sub


def test_pseudonym_is_deterministic_and_consistent():
    assert pseudonym("LEGO") == pseudonym("lego")
    assert pseudonym("LEGO") != pseudonym("Hasbro")
    m = {"LEGO": pseudonym("LEGO")}
    assert sub("LEGO City Fire Truck by lego", m).count(pseudonym("LEGO")) == 2


def test_panels_written_with_flags(tmp_path):
    rec = {"user_id": "u", "source_event_id": "u::1",
           "history_titles": ["LEGO Star Wars", "Barbie Doll"], "history_ratings": [5, 2],
           "history_brands": ["LEGO", "Mattel"],
           "candidate_titles": ["LEGO Technic Car", "Generic Ball"], "candidate_texts": ["by LEGO", ""],
           "candidate_brands": ["LEGO", ""], "candidate_popularity": [100, 3],
           "candidate_labels": [1, 0], "candidate_item_ids": ["a", "b"]}
    p = tmp_path / "toy_rated.jsonl"
    p.write_text(json.dumps(rec))
    subprocess.run([sys.executable, "-m", "src.confrec.pseudonymize", "--panel", str(p),
                    "--out_dir", str(tmp_path)], check=True, capture_output=True)
    ps = json.loads((tmp_path / "toy_rated_pseudo.jsonl").read_text())
    assert ps["candidate_brand_in_title"] == [True, False]
    assert "LEGO" not in ps["candidate_titles"][0] and pseudonym("LEGO") in ps["candidate_titles"][0]
    assert ps["history"][0].startswith(pseudonym("LEGO")) and ps["history"][0].endswith("(rated 5/5)")
    assert ps["candidate_titles"][1] == "Generic Ball"
