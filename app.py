# -*- coding: utf-8 -*-
"""
ViruTwin inference API server (runs on AutoDL GPU instance).
Input : raw protein sequences / mRNA (auto-detected, mRNA translated)
Output: strict JSON per user-mandated schema.
Model selection is strictly honored: one user-chosen model per request,
no cross-model weighting inside a single prediction.
"""
import os, glob, json, traceback
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

from seq_utils import detect_type, translate_mrna, validate_protein, parse_fasta

# ---------------------------------------------------------------- config
WEIGHTS_DIR = os.environ.get("VT_WEIGHTS_DIR", "/root/autodl-tmp/negatome/matrices/model_weights")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN_ESM, MAX_LEN_T5 = 1022, 1000

app = Flask(__name__)
STATE = {"esm": None, "esm_tok": None, "t5": None, "t5_tokenizer": None,
         "scaler_map": {}, "models": {}, "ready": False, "load_error": ""}


class NPZScaler:
    """Minimal scaler compatible with sklearn's mean_/scale_ interface."""
    def __init__(self, path):
        d = np.load(path)
        if "mean_" in d.files and "scale_" in d.files:
            self.mean_, self.scale_ = d["mean_"], d["scale_"]
        else:  # fallback: first two arrays
            self.mean_, self.scale_ = d[d.files[0]], d[d.files[1]]

# ------------------------------------------------------------- PLM loaders
def load_plms():
    """Load ESM-2 via HuggingFace transformers (fair-esm/torch.hub has a known
    version conflict); load ProtT5 the same way for consistency."""
    from transformers import AutoTokenizer, AutoModel, T5EncoderModel, T5Tokenizer
    STATE["esm_tok"] = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    STATE["esm"] = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D").eval().to(DEVICE)

    STATE["t5_tokenizer"] = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc", do_lower_case=False)
    STATE["t5"] = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        torch_dtype=torch.float16).eval().to(DEVICE)


@torch.no_grad()
def embed_esm2(seqs):
    enc = STATE["esm_tok"]([s[:MAX_LEN_ESM] for s in seqs],
                           return_tensors="pt", padding=True)
    out = STATE["esm"](**{k: v.to(DEVICE) for k, v in enc.items()}).last_hidden_state
    emb = []
    for i, s in enumerate(seqs):
        L = min(len(s), MAX_LEN_ESM)
        mask = enc["attention_mask"][i, 1:L + 1].bool().to(DEVICE)
        emb.append(out[i, 1:L + 1][mask].mean(0).float().cpu().numpy())
    return np.stack(emb)


@torch.no_grad()
def embed_prott5(seqs):
    seqs_sp = [" ".join(list(s[:MAX_LEN_T5])) for s in seqs]
    ids = STATE["t5_tokenizer"](seqs_sp, return_tensors="pt", padding=True)
    out = STATE["t5"](ids.input_ids.to(DEVICE), attention_mask=ids.attention_mask.to(DEVICE))
    emb = []
    for i in range(len(seqs_sp)):
        L = min(len(seqs[i]), MAX_LEN_T5)
        mask = ids.attention_mask[i, :L + 1].bool().to(DEVICE)
        emb.append(out.last_hidden_state[i][mask].mean(0).float().cpu().numpy())
    return np.stack(emb)


# ------------------------------------------------------- classifier helpers
class MLPHead(nn.Module):
    """Generic vector-processing head: input_dim -> 256 -> 64 -> 2 (scikit MLP mimic)."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 2))
    def forward(self, x):
        return self.net(x)


# Explicit filename mapping discovered from the actual weights directory.
# SuperEnsemble = equal-weight average of the five DPF variants (M0~M4),
# consistent with the manuscript (SLSQP showed equal weights are optimal).
MODEL_FILES = {
    "mlp":          ["cat__MLP.joblib", "cat__MLP.pth"],
    "vfiter_cat":   ["cat__VFIter.pth"],
    "vfiter_esm2":  ["esm2__VFIter.pth"],
    "vfiter_p5":    ["p5__VFIter.pth"],
    "dpf":          ["cat__M0_Full.pth"],
    "dpf_m1":       ["cat__M1_wo_Gating.pth"],
    "dpf_m2":       ["cat__M2_wo_SelfAttn.pth"],
    "dpf_m3":       ["cat__M3_wo_CrossAttn.pth"],
    "dpf_m4":       ["cat__M4_wo_Balancing.pth"],
}
SUPER_COMPONENTS = ["dpf", "dpf_m1", "dpf_m2", "dpf_m3", "dpf_m4"]
INPUT_DIM = {"mlp": 2304, "vfiter_cat": 2304, "vfiter_esm2": 1280,
             "vfiter_p5": 1024, "dpf": 2304, "dpf_m1": 2304,
             "dpf_m2": 2304, "dpf_m3": 2304, "dpf_m4": 2304}


_BUILDER = None
def _get_builder():
    """Optional hook: user-trained model classes (VFIter/DPF) supplied via a
    python file. The file must define build_model(name, in_dim, state_dict)
    returning an nn.Module already loaded with the state_dict."""
    global _BUILDER
    if _BUILDER is not None:
        return _BUILDER
    path = os.environ.get("VT_MODELS_PY", "")
    if path and os.path.exists(path):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("vt_model_defs", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _BUILDER = getattr(mod, "build_model", None)
            if _BUILDER:
                print(f"[ok] custom model builder loaded from {path}")
        except Exception as e:
            print(f"[warn] VT_MODELS_PY load failed: {e}")
    return _BUILDER


def load_classifiers():
    """Load classifier weights. .joblib files (sklearn) are tried first since
    they may embed their own scaler; .pth files may be full modules or state_dicts."""
    import joblib
    for name, candidates in MODEL_FILES.items():
        for fname in candidates:
            f = os.path.join(WEIGHTS_DIR, fname)
            if not os.path.exists(f):
                continue
            try:
                if fname.endswith(".joblib"):
                    obj = joblib.load(f)
                else:
                    obj = torch.load(f, map_location=DEVICE, weights_only=False)
                    if isinstance(obj, dict):  # bare state_dict
                        try:
                            head = MLPHead(INPUT_DIM[name]).eval().to(DEVICE)
                            head.load_state_dict(obj)
                            obj = head
                        except Exception:
                            builder = _get_builder()
                            if builder is None:
                                raise RuntimeError(
                                    "state_dict does not match generic head and "
                                    "no VT_MODELS_PY builder available")
                            obj = builder(name, INPUT_DIM[name], obj)
                            obj = obj.eval().to(DEVICE)
                    elif isinstance(obj, nn.Module):
                        obj = obj.eval().to(DEVICE)
                    if isinstance(obj, nn.Module) and any(
                            p.is_cuda is False for p in obj.parameters()):
                        obj = obj.to(DEVICE)
                STATE["models"][name] = (obj, f)
                print(f"[ok] loaded {name} <- {f}")
                break
            except Exception as e:
                print(f"[warn] failed to load {name} from {f}: {e}")
    # Per-source scalers: check both the weights dir and its parent (matrices/)
    search_dirs = [WEIGHTS_DIR, os.path.dirname(WEIGHTS_DIR.rstrip("/"))]
    for src, fname in [("cat", "scaler_cat.npz"), ("esm2", "scaler_esm2.npz"),
                       ("p5", "scaler_p5.npz")]:
        for d in search_dirs:
            f = os.path.join(d, fname)
            if os.path.exists(f):
                try:
                    STATE["scaler_map"][src] = NPZScaler(f)
                    print(f"[ok] scaler[{src}] <- {f}")
                    break
                except Exception as e:
                    print(f"[warn] scaler[{src}] load failed from {f}: {e}")


FEATURE_SRC = {"mlp": "cat", "vfiter_cat": "cat", "vfiter_esm2": "esm2",
               "vfiter_p5": "p5"}


def _scaled(emb, src):
    sc = STATE["scaler_map"].get(src)
    x = emb.astype(np.float32)
    if sc is not None:
        x = (x - sc.mean_) / sc.scale_
    return x


@torch.no_grad()
def clf_predict(name, emb_esm, emb_p5, emb_cat):
    """Return VF probability from the selected model (strictly one model).
    DPF models take dual inputs (esm2, p5); all others take a single vector."""
    entry = STATE["models"].get(name)
    if entry is None:
        return None, f"model '{name}' weights not found in {WEIGHTS_DIR}"
    obj = entry[0]
    if name.startswith("dpf"):
        x1 = torch.tensor(_scaled(emb_esm, "esm2"), dtype=torch.float32, device=DEVICE)
        x2 = torch.tensor(_scaled(emb_p5, "p5"), dtype=torch.float32, device=DEVICE)
        logits = obj(x1, x2)
        return float(torch.softmax(logits, -1)[0, 1].cpu()), None
    src = FEATURE_SRC.get(name, "cat")
    emb = {"cat": emb_cat, "esm2": emb_esm, "p5": emb_p5}[src]
    if hasattr(obj, "predict_proba"):  # sklearn (Pipeline already scales internally)
        if hasattr(obj, "named_steps"):
            return float(obj.predict_proba(emb)[0, 1]), None
        return float(obj.predict_proba(_scaled(emb, src))[0, 1]), None
    x = torch.tensor(_scaled(emb, src), dtype=torch.float32, device=DEVICE)
    logits = obj(x)
    return float(torch.softmax(logits, -1)[0, 1].cpu()), None


@torch.no_grad()
def superensemble_predict(emb_esm, emb_p5, emb_cat):
    """Equal-weight average over the five DPF component models M0-M4
    (manuscript Table 7: M0~M4 AUROC 0.9047-0.9110; equal weights optimal)."""
    probs = []
    for name in SUPER_COMPONENTS:
        p, _ = clf_predict(name, emb_esm, emb_p5, emb_cat)
        if p is not None:
            probs.append(p)
    if not probs:
        return None, "no DPF component model (M0-M4) available for SuperEnsemble"
    return float(np.mean(probs)), None

# --------------------------------------------------------------- pipeline
def one_record(sid, raw_seq, model):
    rec = {"seq_id": sid, "input_type": "invalid", "translated_protein": None,
           "prediction_model": model, "prediction_label": "Uncertain",
           "confidence": 0.0, "risk_level": "N/A",
           "evidence": "", "warning": ""}
    if not raw_seq:
        rec["warning"] = "Empty sequence"
        return rec
    t = detect_type(raw_seq)
    rec["input_type"] = t
    if t == "invalid":
        rec["warning"] = "Unrecognized sequence type: neither mRNA nor standard protein"
        return rec
    if t == "mRNA":
        prot, warn = translate_mrna(raw_seq)
        rec["warning"] = warn
        if prot is None:
            rec["translated_protein"] = None
            if not warn:
                rec["warning"] = "mRNA translation failed"
            return rec
        rec["translated_protein"] = prot
    else:
        prot, warn = validate_protein(raw_seq)
        rec["warning"] = warn
        if prot is None:
            return rec
        rec["translated_protein"] = prot

    if len(prot) > 5000:
        rec["prediction_label"] = "Uncertain"
        rec["risk_level"] = "N/A"
        rec["warning"] = (rec["warning"] + " " if rec["warning"] else "") + "Abnormal sequence length (>5000 aa); prediction skipped"
        return rec

    try:
        emb_esm = embed_esm2([prot])
        emb_p5 = embed_prott5([prot])
        emb_cat = np.concatenate([emb_esm, emb_p5], axis=1)

        if model == "superensemble":
            prob, err = superensemble_predict(emb_esm, emb_p5, emb_cat)
        else:
            prob, err = clf_predict(model, emb_esm, emb_p5, emb_cat)
        if prob is None:
            rec["prediction_label"] = "Uncertain"
            rec["risk_level"] = "无法评估"
            rec["warning"] = (rec["warning"] + " " if rec["warning"] else "") + err
            return rec

        conf = float(np.clip(prob, 0.0, 1.0))
        rec["confidence"] = round(conf, 4)
        if conf < 0.5:
            # symmetric decision rule around 0.5
            rec["prediction_label"] = "Non-hazardous" if conf < 0.5 else "Uncertain"
            if abs(conf - 0.5) < 0.1:
                rec["prediction_label"] = "Uncertain"
                rec["risk_level"] = "N/A"
                rec["evidence"] = (f"ESM-2+ProtT5 fused embeddings scored by {model}: "
                                   f"P(VF)={conf:.3f}, near the decision threshold; insufficient evidence")
                return rec
            rec["risk_level"] = "Low"
            rec["evidence"] = (f"ESM-2+ProtT5 fused embeddings scored by {model}: "
                               f"P(VF)={conf:.3f}, below the hazardous threshold")
        else:
            rec["prediction_label"] = "Hazardous"
            rec["risk_level"] = "High" if conf >= 0.8 else "Medium"
            rec["evidence"] = (f"ESM-2+ProtT5 fused embeddings scored by {model}: "
                               f"P(VF)={conf:.3f}; matches virulence-factor embedding-space signatures")
    except Exception as e:
        rec["prediction_label"] = "Uncertain"
        rec["risk_level"] = "N/A"
        rec["warning"] = f"Inference error: {e}"
        traceback.print_exc()
    return rec


# ------------------------------------------------------------------ routes
@app.route("/health")
def health():
    return jsonify({"status": "ok" if STATE["ready"] else "degraded",
                    "device": str(DEVICE),
                    "loaded_models": list(STATE["models"].keys()),
                    "weights_dir": WEIGHTS_DIR, "error": STATE["load_error"]})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    sequences = str(data.get("sequences", ""))
    model = str(data.get("model", "superensemble")).lower().strip()
    valid = {"mlp", "vfiter_cat", "vfiter_esm2", "vfiter_p5",
             "dpf", "dpf_m1", "dpf_m2", "dpf_m3", "dpf_m4", "superensemble"}
    if model not in valid:
        return jsonify({"batch_info": {"total": 0, "success": 0, "fail": 0},
                        "result_list": [],
                        "error": f"unknown model '{model}'. Allowed: {sorted(valid)}"}), 400

    entries = parse_fasta(sequences)
    results = [one_record(sid, seq, model) for sid, seq in entries]
    succ = sum(1 for r in results if r["input_type"] != "invalid"
               and r["translated_protein"] and r["confidence"] > 0)
    return jsonify({
        "batch_info": {"total": len(results), "success": succ,
                       "fail": len(results) - succ},
        "result_list": results,
        "disclaimer": "本预测为AI辅助生物信息预测结果，仅用于科研参考，不能作为临床诊断、生物安全判定唯一依据，结果存在假阳性/假阴性风险"
    })


if __name__ == "__main__":
    try:
        load_plms()
    except Exception as e:
        STATE["load_error"] = f"PLM load failed: {e}"
        traceback.print_exc()
    try:
        load_classifiers()
    except Exception as e:
        STATE["load_error"] += f" | classifier load failed: {e}"
        traceback.print_exc()
    STATE["ready"] = bool(STATE["esm"] and STATE["models"])
    print("Loaded models:", list(STATE["models"].keys()))
    app.run(host="0.0.0.0", port=5000, threaded=True)
