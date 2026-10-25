# -*- coding: utf-8 -*-
"""Sequence type detection, validation and mRNA -> protein translation."""
import re
from typing import Optional, Tuple

CODON_TABLE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

AMINO_STD = set("ACDEFGHIKLMNPQRSTVWY")
DEGENERATE = set("NRYMKSWHBVD")


def clean_seq(raw: str) -> str:
    return re.sub(r"\s+", "", raw.upper())


def detect_type(seq: str) -> str:
    """Return 'mRNA', 'protein' or 'invalid'."""
    if not seq or len(seq) < 10:
        return "invalid"
    chars = set(seq)
    dna_chars = set("ACGTUN") | DEGENERATE
    # mRNA: essentially all characters are nucleotide alphabet
    if chars <= dna_chars and len(chars) <= 8:
        return "mRNA"
    # protein: standard amino acids, allow X/B/Z/U/O as rare/ambiguous
    prot_chars = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
    if chars <= prot_chars:
        return "protein"
    return "invalid"


def translate_mrna(seq: str) -> Tuple[Optional[str], str]:
    """Translate mRNA (U->T). Returns (protein_or_None, warning)."""
    seq = seq.replace("U", "T")
    warning = ""
    # ambiguous bases
    amb = [c for c in seq if c in DEGENERATE]
    if amb:
        warning = (f"Contains {len(amb)} degenerate/ambiguous base(s) "
                   f"({''.join(sorted(set(amb)))}); confidence reduced")
        for c in DEGENERATE:
            seq = seq.replace(c, "N")
    # find first ATG
    start = seq.find("ATG")
    if start == -1:
        return None, (warning + " " if warning else "") + "No start codon ATG found; translation failed"
    prot = []
    for i in range(start, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if "N" in codon:
            prot.append("X")
            continue
        aa = CODON_TABLE.get(codon)
        if aa is None:
            return None, (warning + " " if warning else "") + f"Unrecognized codon {codon}; translation failed"
        if aa == "*":
            break
        prot.append(aa)
    if len(prot) < 10:
        return None, (warning + " " if warning else "") + "Translated product too short (<10 aa); treated as invalid"
    return "".join(prot), warning


def validate_protein(seq: str) -> Tuple[Optional[str], str]:
    """Validate amino-acid sequence. Returns (seq_or_None, warning)."""
    bad = [c for c in seq if c not in AMINO_STD and c not in "XBZUO"]
    if bad:
        return None, f"Illegal character(s): {''.join(sorted(set(bad)))}"
    warn = ""
    if any(c in seq for c in "XBZUO"):
        warn = "Contains non-standard amino acid(s) (X/B/Z/U/O); treated as standard residues"
    seq = re.sub(r"[XBZUO]", "A", seq)  # neutral replacement for embedding
    if len(seq) < 10:
        return None, (warn + " " if warn else "") + "Protein sequence too short (<10 aa)"
    return seq, warn


def parse_fasta(text: str) -> list:
    """Parse FASTA (or raw single sequence) into list of (seq_id, sequence)."""
    text = text.strip()
    if not text:
        return []
    entries = []
    if ">" in text:
        blocks = re.split(r"(?=>)", text)
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            lines = b.splitlines()
            header = lines[0].lstrip(">").strip()
            sid = header.split()[0] if header else f"seq_{len(entries)+1}"
            entries.append((sid, clean_seq("".join(lines[1:]))))
    else:
        entries.append(("user_input", clean_seq(text)))
    return entries
