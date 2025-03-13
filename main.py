import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Function to analyze dataset
def analyze_dataset(file_path):
    df = pd.read_csv(file_path)
    summary = df.describe().to_string()  # Basic stats
    prompt = f"Analyze this dataset summary and highlight potential issues:\n{summary}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data science expert identifying dataset issues."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return response.choices[0].message.content

# Function to suggest improvements
def suggest_improvements(analysis):
    prompt = f"Based on this analysis:\n{analysis}\nSuggest preprocessing steps to improve the dataset."
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a data science expert providing actionable advice."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    file_path = "ECS 260 Tracking - Sheet1.csv"  # Replace with your dataset
    analysis = analyze_dataset(file_path)
    print("Dataset Analysis:\n", analysis)
    suggestions = suggest_improvements(analysis)
    print("\nImprovement Suggestions:\n", suggestions)