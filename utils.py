import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import io
import pandas as pd
from io import BytesIO
import base64
import seaborn as sns
import httpx
from bs4 import BeautifulSoup
import duckdb


def plot_scatter_with_regression(x, y):
    try:
        plt.figure(figsize=(8, 6))
        sns.regplot(x=x, y=y, scatter_kws={"color": "blue"}, line_kws={"color": "red", "linestyle": "dotted"})
        plt.xlabel("Rank")
        plt.ylabel("Peak")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        if len(encoded) > 100000:
            return "Image too large"
        return f"data:image/png;base64,{encoded}"
    except Exception as e:
        return f"Plot generation failed: {str(e)}"



def analyze_grossing_films(raw_data, question):
    header, data = raw_data
    df = pd.DataFrame(data, columns=header) 
    df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
    df["Peak"] = pd.to_numeric(df["Peak"], errors="coerce")

    if "how many $2 bn movies" in question.lower():
        df["Gross"] = pd.to_numeric(df["Worldwide gross"].str.replace("$", "").str.replace(" billion", "").str.replace(",", ""), errors="coerce")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        count = df[(df["Gross"] >= 2) & (df["Year"] < 2000)].shape[0]
        return count

    elif "earliest film" in question.lower():
        df["Gross"] = pd.to_numeric(df["Worldwide gross"].str.replace("$", "").str.replace(" billion", "").str.replace(",", ""), errors="coerce")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
        filtered = df[df["Gross"] > 1.5]
        earliest = filtered.sort_values("Year").iloc[0]["Title"]
        return earliest

    elif "correlation" in question.lower():
        corr = df["Rank"].corr(df["Peak"])
        return round(corr, 6)

    elif "scatterplot" in question.lower():
        return plot_scatter_with_regression(df["Rank"], df["Peak"])

    return "Unknown question"

def analyze_court_data(files):
    try:
        parquet_file = None
        for name, file in files.items():
            if name.endswith(".parquet"):
                parquet_file = file
                break

        if not parquet_file:
            return "Missing parquet file"

        df = pd.read_parquet(parquet_file)
        df["date_of_registration"] = pd.to_datetime(df["date_of_registration"], errors="coerce")
        df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")
        df["delay_days"] = (df["decision_date"] - df["date_of_registration"]).dt.days
        df = df[df["court"] == "33_10"]

        slope = df.groupby("year")[["delay_days"]].mean().reset_index()
        corr = slope["year"].corr(slope["delay_days"])
        
        uri = plot_scatter_with_regression(slope["year"], slope["delay_days"])
        return {
            "regression_slope": round(corr, 6),
            "plot": uri
        }
    except Exception as e:
        return f"Court data analysis failed: {str(e)}"



def process_question(question_text, files=None):
    questions = [q.strip() for q in question_text.strip().split("\n") if q.strip()]
    answers = []

    for q in questions:
        if "earliest film" in q.lower():
            header, data = scrape_grossing_movies()
            df = pd.DataFrame(data, columns=header)
            if not df.empty:
                context = df[["Title", "Year", "Worldwide gross"]].head(10).to_string(index=False)
                ans = ask_llm(q, context=context)
            else:
                ans = "No data available to answer this question."

        elif "highest grossing films" in q.lower():
            data = scrape_grossing_movies()
            ans = analyze_grossing_films(data, q)
            answers.append(ans)
        elif "correlation" in q.lower():
            corr = df["Rank"].corr(df["Peak"])
            return round(corr, 6) if pd.notnull(corr) else "Correlation could not be calculated"
        elif "indian high court" in q.lower():
            result = analyze_court_data(files)
            if isinstance(result, dict):
                answers.append(result["regression_slope"])
                answers.append(result["plot"])
            else:
                answers.append(result)
        else:
            ans = ask_llm(q)
            answers.append(ans)

    return answers



def ask_llm(prompt, context=None):
    headers = {
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIzZjIwMDM4NzlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.I82DRN0opE0oOyv01g6bybkwsKVQtCXzrwBobl6uusk",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "You are a helpful data analyst."}
    ]
    if context:
        messages.append({"role": "user", "content": f"Here is some relevant data:\n{context}"})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": "google/gemini-2.0-flash-lite-001",
        "messages": messages,
        "service_tier": "flex"
    }

    try:
        response = httpx.post("https://aipipe.org/openrouter/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict) and "choices" in data:
            return data["choices"][0]["message"]["content"]
        else:
            return f"Unexpected response format: {data}"

    except httpx.HTTPStatusError as e:
        return f"LLM request failed: {e.response.text}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def scrape_grossing_movies():
    url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
    resp = httpx.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", class_="wikitable")
    rows = table.find_all("tr")
    header = [th.text.strip() for th in rows[0].find_all("th")]
    data = []
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) == len(header):
            data.append([cell.text.strip() for cell in cells])
    return header, data


def count_cases():
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; INSTALL parquet; LOAD parquet;")
    return con.execute("""
        SELECT COUNT(*) 
        FROM read_parquet('s3://.../metadata.parquet?s3_region=ap-south-1')
    """).fetchone()[0]
