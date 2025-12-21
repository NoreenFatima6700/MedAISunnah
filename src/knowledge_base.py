import pandas as pd
import json

def load_quran(path="data/quran.csv"):
    df = pd.read_csv(path)
    # Full text includes reference
    df["text"] = df.apply(lambda x: f"Quran {x['surah']}:{x['ayah']}: {x['text']}", axis=1)
    df["source"] = df.apply(lambda x: f"Quran {x['surah']}:{x['ayah']}", axis=1)
    return df[["text", "source"]]

def load_hadith(path="data/hadith.csv"):
    df = pd.read_csv(path)
    # Full text includes reference
    df["text"] = df.apply(lambda x: f"{x['source']} {x['reference']}: {x['text']}", axis=1)
    df["source"] = df.apply(lambda x: f"{x['source']} {x['reference']}", axis=1)
    return df[["text", "source"]]

def load_herbs(path="data/herbs.json"):
    with open(path, "r") as f:
        data = json.load(f)
    texts = []
    for item in data:
        texts.append({
            "text": f"{item['name']}: {item['benefits']}",
            "source": item["source"]
        })
    return pd.DataFrame(texts)

def load_knowledge_base():
    quran = load_quran()
    hadith = load_hadith()
    herbs = load_herbs()
    return pd.concat([quran, hadith, herbs], ignore_index=True)
