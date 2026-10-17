# FloodGuard BD — Intelligent Flood Prediction & Risk Awareness for Bangladesh

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-3.5-FF6F00?logo=tensorflow&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?logo=mongodb&logoColor=white)
![Folium](https://img.shields.io/badge/Maps-Folium-77B829)

**FloodGuard BD** is a web platform for early flood-risk prediction and awareness across the districts of [Bangladesh](https://en.wikipedia.org/wiki/Bangladesh). A **stacked-LSTM deep learning model** trained on 65 years (1948–2013) of weather-station records estimates flood probability from weather conditions. The results appear on an interactive **Streamlit** dashboard with risk maps and a subscription-based alert system.

---

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Model](#model)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Usage](#usage)
- [Roadmap](#roadmap)

---

## Key Features

| Feature | Description |
|---|---|
| 🔍 **Predictive analytics** | District-wise flood-risk probability from temperature, rainfall, humidity, wind speed, cloud cover, sunshine, altitude, and month |
| 🗺️ **Interactive risk map** | Folium map of Bangladesh with clustered markers coloured by predicted flood probability (red ≥ 0.5, blue < 0.5) |
| 📧 **Alert subscriptions** | Users subscribe with an email and location. Subscriptions are stored in MongoDB Atlas. |
| 📍 **Localised awareness** | Built-in coordinates for 61 districts, with recent-warning feeds and FAQs |
| ⚡ **Fast inference** | The model and scaler are cached with `st.cache_resource`, so they load once per server process |

---

## System Architecture

```text
┌──────────────────────────── Streamlit UI (app/main.py) ────────────────────────────┐
│  Home  │  Search Now (prediction form)  │  Flood-Prone Areas (map)  │ Notifications │
└────┬───────────────┬──────────────────────────────┬──────────────────────┬─────────┘
     │               │                              │                      │
     │      ┌────────▼─────────┐          ┌─────────▼─────────┐   ┌────────▼────────┐
     │      │ Feature pipeline │          │ Batch inference    │   │ app/mongodb.py  │
     │      │ MinMax scaling + │          │ over historical    │   │ save_subscription│
     │      │ cyclic month enc.│          │ station records    │   └────────┬────────┘
     │      └────────┬─────────┘          └─────────┬─────────┘            │
     │               └──────────┬───────────────────┘                      ▼
     │                          ▼                                   MongoDB Atlas
     │              flood_model.keras (stacked LSTM)                (subscriptions)
     │              scaler.pkl (fitted scaler)
     ▼
 Static assets (images, logo, dataset)
```

---

## Model

A binary classifier that outputs `P(flood)`. It is stored at `app/assets/flood_model.keras` and was saved with Keras 3.5.0.

```text
Input (12 × 1)
 → Dense(32)
 → LSTM(128, return_sequences) → Dropout(0.2) → LayerNorm
 → LSTM(128)                   → Dropout(0.2) → LayerNorm
 → Dense(64, ReLU)
 → Dense(1, Sigmoid)
```

**Input features (12):**

| # | Feature | Preprocessing |
|---|---|---|
| 1–7 | Max Temp, Min Temp, Rainfall, Relative Humidity, Wind Speed, Cloud Coverage, Bright Sunshine | scaled with `scaler.pkl` |
| 8–10 | X_COR, Y_COR (district coordinates), ALT (altitude) | scaled with `scaler.pkl` |
| 11–12 | `month_sin`, `month_cos` | cyclic encoding: `sin/cos(2π · month / 12)` |

The risk label is **High Risk** when `P(flood) ≥ 0.5` and **Low Risk** otherwise.

---

## Dataset

`app/assets/flood_dataset.csv` holds about 21,000 monthly records from **35 weather stations** across Bangladesh, covering **1948–2013**.

| Column group | Columns |
|---|---|
| Identity | `Station Names`, `Station Number`, `YEAR`, `Month`, `Period` |
| Weather | `Max Temp`, `Min Temp`, `Rainfall`, `Relative Humidity`, `Wind Speed`, `Cloud Coverage`, `Bright Sunshine` |
| Location | `X_COR`, `Y_COR`, `LATITUDE`, `LONGITUDE`, `ALT` |

---

## Project Structure

```text
FloodGaurd/
├── README.md
└── flood-prediction/
    ├── requirements.txt
    ├── app/
    │   ├── main.py              # Streamlit entry point: pages, routing, inference
    │   ├── mongodb.py           # MongoDB Atlas client + save_subscription()
    │   └── assets/
    │       ├── flood_model.keras    # Trained stacked-LSTM model
    │       ├── flood_model.pkl      # Alternative serialized model
    │       ├── scaler.pkl           # Fitted feature scaler
    │       ├── flood_dataset.csv    # Historical station data
    │       └── *.jpg / *.jpeg       # UI imagery
    └── pages/                   # Experimental / alternate page implementations
        ├── dashboard.py
        ├── flood_prone_area.py  # Open-Meteo forecast fetch prototype
        ├── notifications.py
        └── search_now.py
```

---

## Getting Started

### Prerequisites

- Python **3.10–3.12**
- A MongoDB Atlas cluster, if you want alert subscriptions

### Installation

```bash
git clone https://github.com/RehenumaIlman/FloodGaurd.git
cd FloodGaurd/flood-prediction

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install "keras>=3.5" pymongo pillow
```

> `requirements.txt` lists the core stack. `pymongo` (for subscriptions) and `pillow` are also imported by the app. The prototypes in `pages/` additionally use `openmeteo-requests`, `requests-cache`, `retry-requests`, and `geopy`.

### Run

Run the app **from the `flood-prediction/` directory**, because asset paths are relative to it:

```bash
streamlit run app/main.py
```

Then open <http://localhost:8501>.

---

## Configuration

The MongoDB connection is set in `app/mongodb.py`. For any shared or deployed environment, keep credentials **out of source control**. Supply them through an environment variable or Streamlit secrets (`.streamlit/secrets.toml` is already git-ignored):

```toml
# flood-prediction/.streamlit/secrets.toml
MONGODB_URI = "mongodb+srv://<user>:<password>@<cluster>/flood?retryWrites=true&w=majority"
```

> ⚠️ If real credentials were ever committed, rotate them in MongoDB Atlas.

---

## Usage

| Page | What it does |
|---|---|
| **Home** | Project overview, how it works, regional flood probabilities, and FAQs |
| **Search Now** | Choose a district, enter weather conditions and a month, then click **Predict Flood Risk** to get the risk level and probability |
| **Flood-Prone Areas** | Runs batch inference over the historical dataset and plots the risk per station on a clustered map |
| **Notifications** | Shows current warnings and lets users subscribe for email alerts |

You can deep-link to a page with `?page=search`, `?page=flood`, or `?page=notifications`.

---

## Roadmap

- [ ] Fill the form automatically with live forecasts from the Open-Meteo API (prototype in `pages/flood_prone_area.py`)
- [ ] Send email alerts automatically when a subscribed district crosses the risk threshold
- [ ] Add river water-level and soil-moisture features
- [ ] Publish the training notebook and evaluation metrics
- [ ] Containerise the app with Docker and deploy it to Streamlit Community Cloud

---

<sub>Built to strengthen community resilience and disaster preparedness in Bangladesh.</sub>
