# Sleep Health Analytics & Web Dashboard

Problem Statement 1 — analyzes sleep, stress, and lifestyle data, classifies
individuals into sleep health tiers, and presents results via an interactive
Streamlit dashboard.

## Pipeline (run in this exact order)

```
1) Neev_data_engg.py     Raw data -> cleaned_data.csv        (Person 1)
2) feature-engineer.py   cleaned_data.csv -> processed_data.csv  (Person 2, adds Sleep_Health_Tier)
3) visualization.py      Chart functions used by app.py       (Person 3, no direct run needed)
4) app.py                Streamlit dashboard + executive summary (Person 4 & 5)
```
## Deployment

```
This project is deployed and live on Streamlit Community Cloud: https://sleep-health-dashboardgit.streamlit.app/ 
```

## Demo Video

```
A short walkthrough video demonstrating the dashboard's features and functionality is available here: 
```
## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python Neev_data_engg.py
python feature-engineer.py
streamlit run app.py or python -m streamlit run app.py
```

Make sure `Sleep_health_and_lifestyle_dataset.csv` (the raw dataset) is in
the same folder before running step 1.

Each script prints its own validation report to the console as it runs, so
you can confirm cleaning, tier counts, and record counts at every stage
before moving to the next one.
