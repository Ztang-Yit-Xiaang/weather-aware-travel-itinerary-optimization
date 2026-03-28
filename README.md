# Weather-Aware Travel Itinerary Optimization

A data-driven itinerary planning system that integrates **machine learning, geospatial analytics, and optimization** to recommend travel routes under weather, time, and travel constraints.

This project was developed for **IE5533 – Operations Research for Data Science**.

The system combines attraction utility modeling, weather-aware congestion estimation, and travel-time optimization to generate realistic travel itineraries.

---

# Project Overview

Urban tourists often face the challenge of selecting attractions within limited time while considering travel distance, weather conditions, and expected congestion.

Traditional recommendation systems rank attractions independently and ignore routing constraints.

This project proposes a **three-layer decision framework**:

1. **Attraction Utility Modeling**  
2. **Weather-Aware Congestion Estimation**  
3. **Route Optimization with Time Constraints**

The system integrates real-world datasets including Yelp attraction metadata and historical weather data to generate optimized travel itineraries.

---

# Methodology

The project pipeline consists of several stages:

## 1 Data Collection

Datasets used:

- **Yelp Open Dataset**
  - Business metadata
  - Reviews
- **Open-Meteo Weather API**
- Geographic coordinates for attractions

Key attraction variables:

- Name
- City
- Rating
- Review count
- Latitude / Longitude
- Categories

---

## 2 Attraction Utility Model

Each attraction receives a baseline utility score:

$$
U_i = rating_i \cdot \log(review\_count_i)
$$

This captures both:

- popularity (review count)
- quality (rating)

Top attractions are selected based on this score.

---

## 3 Weather Feature Engineering

Historical weather data is collected via the **Open-Meteo API**.

Features include:

- average temperature
- precipitation
- rain flag
- cold flag
- day-of-week
- weekend indicator

These features are used to approximate visitor congestion conditions.

---

## 4 Travel Time Matrix

Distances between attractions are calculated using geographic coordinates.

Distance is computed using **geodesic distance**:

```python
geopy.distance.geodesic()
```
Travel time is estimated using an assumed average travel speed.

This produces a full travel-time matrix between attractions.

## 5 Visit Duration Simulation

Visit duration is estimated using attraction category heuristics.

Examples:

| Category  | Estimated Duration |
|-----------|-------------------|
| Museum    | 60–120 minutes    |
| Zoo       | 90–180 minutes    |
| Park      | 60–120 minutes    |
| Landmark  | 45–75 minutes     |

A stochastic simulation is used:

visit_duration ~ Normal(mean, sigma)

This captures variability in visitor behavior.

## 6 Review Density Estimation

Review activity is extracted from the Yelp review dataset.

Daily review counts are used as a proxy for temporal congestion patterns.

## 7 Travel Planning Agent

A travel planning assistant is implemented using Anthropic Claude API.

The agent can call the following tools:

- search_attractions
- get_weather
- plan_daily_itinerary

The system generates:

attraction recommendations
weather insights
multi-day travel itineraries

Example query:

I want to visit Santa Barbara for 3 days. I like beaches and museums.
Repository Structure

```
weather-aware-itinerary-optimization
│
├── README.md
├── requirements.txt
│
├── notebook
│   └── itinerary_optimization_pipeline.ipynb
│
├── data
│   ├── yelp_business.json
│   ├── yelp_review.json
│   └── weather_data.csv
│
├── results
│   ├── attraction_map.png
│   ├── travel_time_matrix.csv
│   └── itinerary_outputs
│
└── report
    └── IE5533_project_report.pdf

```

Installation

Clone the repository:
```
git clone https://github.com/yourusername/weather-aware-itinerary-optimization.git
cd weather-aware-itinerary-optimization
```
Install dependencies:
```
pip install -r requirements.txt
```

Dependencies

Main libraries used:

pandas
numpy
matplotlib
requests
geopy
anthropic
jupyter

Install them with:
```
pip install -r requirements.txt
```

## Example Output

The system can generate:

ranked attraction lists
geographic attraction maps
travel time matrices
weather-aware itinerary recommendations

## Example visualization:

Top 100 Attractions (California Coast)

Displays attraction locations and rankings based on utility scores.

## Future Improvements

Possible extensions include:

integrating real traffic data
reinforcement learning itinerary updates
multi-day dynamic optimization
crowd prediction models
real-time routing APIs
## Authors

Yit Xiaang Ztang
University of Minnesota

Course: IE5533 – Operations Research for Data Science

## License

This project is for academic and research purposes.
