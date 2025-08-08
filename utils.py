import matplotlib.pyplot as plt
import io
import base64
import seaborn as sns
import openai
import httpx
from bs4 import BeautifulSoup
import duckdb
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def plot_scatter_with_regression(x, y):
    plt.figure(figsize=(8,6))
    sns.regplot(x=x, y=y, scatter_kws={"color":"blue"}, line_kws={"color":"red", "linestyle":"dotted"})
    plt.xlabel("Rank")
    plt.ylabel("Peak")

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def ask_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def scrape_grossing_movies():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    resp = httpx.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", class_="wikitable")
    data = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all(["td", "th"])
        if len(cells) > 1:
            data.append([cell.text.strip() for cell in cells])
    return data

def count_cases():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")
    return con.execute("""
        SELECT COUNT(*) 
        FROM read_parquet('s3://.../metadata.parquet?s3_region=ap-south-1')
    """).fetchone()[0]