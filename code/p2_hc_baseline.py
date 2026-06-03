"""Phase 2 H-C v2 — recompute H-C with a proper control-doc baseline.

Subagent critique: the 1.6% image-attention number at L16 has no baseline.
It could be normal positional decay on far-back tokens.

Fix: now that we've cached jrcy0227_p98 (a control doc) with attention, compute
the same image-attention statistic at L16 on the control's last 257 query
positions. If the control also shows ~1-2% image attention, the pgjw0227 result
is normal positional decay; if the control shows substantially HIGHER image
attention (e.g., 5-10%), pgjw0227's 1.6% IS abnormally low.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ACT_DIR = ROOT / "results" / "activations"
TRIG_DIR = ROOT / "results" / "p1_trigger_v2"
OUT_DIR = ROOT / "results" / "p2_pilot"
FIG_DIR = ROOT / "docs" / "figures"

IMAGE_TOKEN_IDS = {151655, 151654}
DOCS_TO_COMPARE = ["docvqa_pgjw0227_p5", "docvqa_jrcy0227_p98"]


def compute_attention_breakdown(doc_id: str):
    cache_dir = ACT_DIR / doc_id
    meta = json.loads((cache_dir / "meta.json").read_text())
    attn = torch.load(cache_dir / "attn_weights.pt", weights_only=True)
    layer_indices = torch.load(cache_dir / "layer_indices.pt", weights_only=True).tolist()
    T = meta["T"]
    prompt_len = meta["prompt_len"]

    # Re-derive image positions via the processor.
    from transformers import AutoProcessor
    from PIL import Image
    proc = AutoProcessor.from_pretrained(
        "nanonets/Nanonets-OCR2-3B",
        revision="c3886ff00bb037ce7da24988c9eafaf1fe2bed72",
    )
    # Find image path
    for set_dir in ("public_corpus", "real_loopy"):
        m = ROOT / "data" / set_dir / "manifest.json"
        if m.exists():
            ents = json.loads(m.read_text())
            for e in ents:
                if e["doc_id"] == doc_id:
                    img_path = ROOT / e["path"]
                    break
            else:
                continue
            break
    img = Image.open(img_path).convert("RGB")
    USER_PROMPT = (
        "Extract the text from the above document as if you were reading it naturally. "
        "Return the tables in HTML format. Return the equations in LaTeX representation. "
        "If there is an image in the document and image caption is not present, add a small "
        "description of the image inside the <img></img> tag; otherwise, add the image caption "
        "inside <img></img>. Watermarks should be wrapped in brackets. Ex: "
        "<watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. "
        "Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. "
        "Prefer using ☐ and ☑ for check boxes."
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": USER_PROMPT}]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt")
    prompt_ids = inputs["input_ids"][0].tolist()
    image_positions = [i for i, t in enumerate(prompt_ids) if t in IMAGE_TOKEN_IDS]

    L_kept, H, W_rows, T_keys = attn.shape
    image_pos_t = torch.tensor(image_positions, dtype=torch.long)
    text_prompt_pos_t = torch.tensor([i for i in range(prompt_len) if i not in set(image_positions)], dtype=torch.long)
    gen_keys = torch.arange(prompt_len, T)

    by_layer = []
    for li in range(L_kept):
        per_head_img = attn[li, :, :, image_pos_t].float().sum(dim=-1).mean(dim=-1).mean().item()
        per_head_txt = attn[li, :, :, text_prompt_pos_t].float().sum(dim=-1).mean(dim=-1).mean().item()
        per_head_gen = attn[li, :, :, gen_keys].float().sum(dim=-1).mean(dim=-1).mean().item()
        by_layer.append({"layer": int(layer_indices[li]),
                          "img_attn_mass": per_head_img,
                          "text_prompt_attn_mass": per_head_txt,
                          "gen_attn_mass": per_head_gen})
    return by_layer, layer_indices, len(image_positions)


def main() -> int:
    print(f"[hc-bl] comparing image-attention between:")
    for doc in DOCS_TO_COMPARE:
        print(f"        {doc} ({'POS' if doc.endswith('pgjw0227_p5') else 'ctl'})")

    results = {}
    for doc in DOCS_TO_COMPARE:
        by_layer, lyrs, n_img = compute_attention_breakdown(doc)
        results[doc] = {"by_layer": by_layer, "n_image_positions": n_img}
        print(f"\n[hc-bl] {doc}  (n_image_positions={n_img}):")
        print(f"  {'layer':>5}  {'img':>8}  {'text-prompt':>13}  {'gen':>8}")
        for r in by_layer:
            print(f"  {r['layer']:>5}  {r['img_attn_mass']:>8.4f}  {r['text_prompt_attn_mass']:>13.4f}  {r['gen_attn_mass']:>8.4f}")

    # Comparison plot.
    pos_by = {r["layer"]: r for r in results[DOCS_TO_COMPARE[0]]["by_layer"]}
    ctl_by = {r["layer"]: r for r in results[DOCS_TO_COMPARE[1]]["by_layer"]}
    layers = sorted(pos_by.keys())

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ax.plot(layers, [pos_by[l]["img_attn_mass"] for l in layers], "o-",
            color="tab:red", linewidth=2.4, markersize=8,
            label=f"POSITIVE (pgjw0227, in-loop): image-attn")
    ax.plot(layers, [ctl_by[l]["img_attn_mass"] for l in layers], "o-",
            color="tab:blue", linewidth=2.4, markersize=8,
            label=f"CONTROL (jrcy0227_p98, pre-EOS): image-attn")
    ax.plot(layers, [pos_by[l]["text_prompt_attn_mass"] for l in layers], "s--",
            color="tab:red", linewidth=1.4, alpha=0.5, label="POSITIVE: text-prompt-attn")
    ax.plot(layers, [ctl_by[l]["text_prompt_attn_mass"] for l in layers], "s--",
            color="tab:blue", linewidth=1.4, alpha=0.5, label="CONTROL: text-prompt-attn")
    ax.set_xlabel("decoder layer", fontsize=11)
    ax.set_ylabel("mean attention mass (over heads × 257 query positions)", fontsize=11)
    ax.set_title("H-C v2: is the 1.6% image-attention at L16 abnormally low?\n"
                  "Comparison of in-loop positive vs pre-EOS control on same prompt structure.",
                  fontsize=11)
    ax.set_xticks(layers)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    out_fig = FIG_DIR / "hc_attention_with_baseline.png"
    plt.savefig(out_fig, dpi=170)
    plt.close()
    print(f"\n[hc-bl] wrote {out_fig}")

    # Per-layer ratio.
    print(f"\n[hc-bl] image-attention POSITIVE / CONTROL ratio per layer:")
    print(f"{'layer':>5}  {'pos_img':>10}  {'ctl_img':>10}  {'pos/ctl':>10}")
    ratios = []
    for l in layers:
        pos = pos_by[l]["img_attn_mass"]
        ctl = ctl_by[l]["img_attn_mass"]
        ratio = pos / max(ctl, 1e-9)
        ratios.append({"layer": l, "pos_img": pos, "ctl_img": ctl, "ratio_pos_ctl": ratio})
        print(f"{l:>5}  {pos:>10.4f}  {ctl:>10.4f}  {ratio:>10.3f}")

    # Verdict: is positive's image attention significantly LOWER than control's?
    pos_img_all = np.array([pos_by[l]["img_attn_mass"] for l in layers])
    ctl_img_all = np.array([ctl_by[l]["img_attn_mass"] for l in layers])
    # Layers where positive < control significantly:
    diff = ctl_img_all - pos_img_all
    diff_signed_layers = [(l, float(d)) for l, d in zip(layers, diff)]
    n_pos_lower = sum(1 for l, d in diff_signed_layers if d > 0)
    print(f"\n[hc-bl] layers where POSITIVE image-attn < CONTROL image-attn: {n_pos_lower}/{len(layers)}")

    if n_pos_lower >= 7:
        verdict = ("PASS — POSITIVE has lower image attention than CONTROL on most layers, "
                    "supporting H-C (vision-grounding collapse). Note: still N=1 vs N=1; "
                    "needs N=40 to be statistically meaningful.")
    elif n_pos_lower >= 5:
        verdict = "WEAK — POSITIVE has lower image attention than CONTROL on a majority of layers but not strongly. Marginal H-C support."
    else:
        verdict = "FAIL — POSITIVE does NOT have systematically lower image attention than CONTROL. The 1.6% number was likely normal positional decay."

    print(f"[hc-bl] verdict: {verdict}")

    (OUT_DIR / "hc_baseline_results.json").write_text(json.dumps({
        "docs_compared": DOCS_TO_COMPARE,
        "n_image_positions_per_doc": {d: results[d]["n_image_positions"] for d in DOCS_TO_COMPARE},
        "by_layer": ratios,
        "n_layers_pos_lower_than_ctl": n_pos_lower,
        "verdict": verdict,
        "figure": str(out_fig.relative_to(ROOT)),
    }, indent=2))
    print(f"[hc-bl] wrote {OUT_DIR/'hc_baseline_results.json'}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
