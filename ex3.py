from serpapi import GoogleSearch
import pandas as pd

params = {
    "api_key": "14c9957c14272f34073d41bd7a4820e7bd21d251fa178b4eb320275ce2a0a7cd",
    "engine": "google_scholar_author",
    "hl": "en",
    "author_id": "R9mwtaUAAAAJ",
    "num": "150"
}

search = GoogleSearch(params)
results = search.get_dict()

# Extract publication data
publications = results.get("articles", [])

# Convert to DataFrame
df = pd.json_normalize(publications)

# Save to CSV
output_file = "google_scholar_RCA_Naidu.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ Data saved successfully to: {output_file}")
print(df.head())
