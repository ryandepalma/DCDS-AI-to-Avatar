import pandas as pd
from textblob import TextBlob
import nltk
import os

# download at the first time
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# get excel
file_path = "/Users/zixizeng/Desktop/VAN-EDU_SCOUN_0006_B_1_1_SOLO.mp4.xlsx"
df = pd.read_excel(file_path)

# calculate sentiment
df["sentiment"] = df["text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# stopwords(can add others)
stopwords = set([
    "the","and","is","to","of","a","in","i","it","that","this",
    "for","on","with","as","was","are","be","at","by","an","or",
    "we","you","they","he","she","my","our","your"
])

# sotre data
word_data = {}

for i in range(len(df)):
    text = str(df.loc[i, "text"])
    sentiment = df.loc[i, "sentiment"]
    volume = df.loc[i, "avg_volume"]
    speed = df.loc[i, "words_per_second"]
    
    # 
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    
    for word, tag in tagged:
        word = word.lower()
        
        # keep useful words
        if tag.startswith("NN") and word not in stopwords and len(word) > 2:
            
            if word not in word_data:
                word_data[word] = {
                    "count": 0,
                    "sentiment": 0,
                    "volume": 0,
                    "speed": 0,
                    "sentences": []
                }
            
            word_data[word]["count"] += 1
            word_data[word]["sentiment"] += sentiment
            word_data[word]["volume"] += volume
            word_data[word]["speed"] += speed
            
            # ✅ only get high sentiment sentence
            if sentiment > 0:
                word_data[word]["sentences"].append(text)

# change to DataFrame
rows = []

for word, data in word_data.items():
    
    # get top3 sentences
    example_sentences = list(set(data["sentences"]))[:3]
    
    rows.append([
        word,
        data["count"],
        data["sentiment"] / data["count"],
        data["volume"] / data["count"],
        data["speed"] / data["count"],
        " | ".join(example_sentences)
    ])

result = pd.DataFrame(rows, columns=[
    "word", "frequency", "avg_sentiment", "avg_volume", "avg_speed", "example_sentences"
])

# ✅ frequency
result = result[result["frequency"] >= 3]

# order
result = result.sort_values(by="avg_sentiment", ascending=False)

# keep in desktop
desktop = os.path.join(os.path.expanduser("~"), "Desktop")
output_path = os.path.join(desktop, "VAN-EDU_SCOUN_0006_B_1_1_SOLO_word_emphasis_analysis.xlsx")

result.to_excel(output_path, index=False)

print("Done!", output_path)
