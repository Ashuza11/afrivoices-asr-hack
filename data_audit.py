"""Grounded data audit for the AfriVoices ASR hackathon.
"""
import os, sys, json, urllib.request, urllib.parse

def get_token():
    t = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not t:
        p = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(p):
            t = open(p).read().strip()
    if not t:
        import getpass
        t = getpass.getpass("HF token: ").strip()
    return t

TOK = get_token()

def api(path, **params):
    url = "https://datasets-server.huggingface.co/" + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOK}"})
    try:
        with urllib.request.urlopen(req, timeout=40) as r:
            return json.load(r)
    except Exception as e:
        body = ""
        try: body = e.read().decode()[:200]
        except Exception: pass
        return {"error": f"{e} {body}"}

DATASETS = [
    "Anv-ke/kikuyu", "Anv-ke/Dholuo", "Anv-ke/Kalenjin",
    "Anv-ke/Maasai", "Anv-ke/Somali",
    "MCAA1-MSU/anv_data_ke",
    "DigitalUmuganda/Afrivoice", "DigitalUmuganda/Afrivoice_Swahili",
]
DUR_HINTS  = ("dur", "length", "seconds", "secs")
CAT_HINTS  = ("type", "dialect", "domain", "language", "source", "scripted")

def fmt_freq(stats):
    f = stats.get("frequencies") or stats.get("value_counts")
    if isinstance(f, dict):
        items = sorted(f.items(), key=lambda kv: -kv[1])[:8]
        return ", ".join(f"{k}={v}" for k, v in items)
    return f"n_unique={stats.get('n_unique')}"

for ds in DATASETS:
    print("\n" + "#" * 70)
    print("#", ds)
    sp = api("splits", dataset=ds)
    if "error" in sp:
        print("  NOT ACCESSIBLE:", sp["error"][:160]); continue
    splits = sp.get("splits", [])
    sz = api("size", dataset=ds)
    sizemap = {}
    if "size" in sz:
        for s in sz["size"].get("splits", []):
            sizemap[(s["config"], s["split"])] = s.get("num_rows")
    print("  splits (config/split -> rows):")
    for it in splits:
        print(f"    {it['config']}/{it['split']} -> {sizemap.get((it['config'], it['split']))}")

    # detailed stats only for train splits (keeps output short but complete)
    for it in splits:
        cfg, split = it["config"], it["split"]
        if "train" not in split.lower():
            continue
        st = api("statistics", dataset=ds, config=cfg, split=split)
        if "error" in st:
            print(f"  stats [{cfg}/{split}] error: {st['error'][:120]}"); continue
        n = st.get("num_examples")
        print(f"  --- stats [{cfg}/{split}] rows={n} ---")
        cols = st.get("statistics", [])
        print("    columns:", [c["column_name"] for c in cols])
        for c in cols:
            name = c["column_name"].lower(); s = c["column_statistics"]
            if any(h in name for h in DUR_HINTS) and "mean" in s:
                mean = s.get("mean")
                hrs = (mean * n / 3600) if (mean and n) else None
                print(f"    DURATION '{c['column_name']}': mean={mean:.2f}s min={s.get('min')} "
                      f"max={s.get('max')} -> TOTAL {hrs:.1f} h" if hrs else
                      f"    DURATION '{c['column_name']}': {s}")
            elif any(h in name for h in CAT_HINTS):
                print(f"    {c['column_name']}: {fmt_freq(s)}")
print("\nDONE.")
