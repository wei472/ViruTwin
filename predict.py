# -*- coding: utf-8 -*-
# ViruTwin CLI: virulence factor prediction from protein/mRNA sequences.
# Usage:
#   python predict.py --input seqs.fasta --model superensemble --out result.json
#   python predict.py --seq "MKKILCAVAVGISASVQATK" --model mlp
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("VT_WEIGHTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_weights"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import app
from seq_utils import parse_fasta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="FASTA file path")
    ap.add_argument("--seq", help="single raw sequence")
    ap.add_argument("--model", default="superensemble",
                    choices=["superensemble","mlp","vfiter_cat","vfiter_esm2",
                             "vfiter_p5","dpf","dpf_m1","dpf_m2","dpf_m3","dpf_m4"])
    ap.add_argument("--out", default=None, help="output JSON file (default: stdout)")
    a = ap.parse_args()

    app.load_plms(); app.load_classifiers()
    assert app.STATE["models"], "no classifier loaded, check model_weights/ dir"

    if a.input:
        entries = parse_fasta(open(a.input).read())
    else:
        entries = [("user_input", a.seq)]
    results = [app.one_record(sid, seq, a.model) for sid, seq in entries]
    ok = sum(1 for r in results if r["confidence"] > 0)
    payload = {"batch_info": {"total": len(results), "success": ok, "fail": len(results) - ok},
               "result_list": results,
               "disclaimer": "AI-assisted bioinformatic prediction for research reference only."}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if a.out:
        open(a.out, "w").write(text); print("saved:", a.out)
    else:
        print(text)
