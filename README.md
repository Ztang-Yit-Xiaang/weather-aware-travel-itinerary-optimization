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
# Mathematical Formulation

The itinerary planning problem is formulated as a **mixed-integer optimization problem** that balances attraction utility, travel time, congestion, and budget constraints.

---

# Decision Variables

Let

$$
N
$$

be the set of candidate attractions.

We define the following decision variables:

$$
x_i =
\begin{cases}
1 & \text{if attraction } i \text{ is visited}\\
0 & \text{otherwise}
\end{cases}
$$

$$
y_{ij} =
\begin{cases}
1 & \text{if travel occurs from attraction } i \text{ to } j\\
0 & \text{otherwise}
\end{cases}
$$

Additionally, ordering variables are introduced to eliminate subtours:

$$
u_i
$$

which represents the visiting order of attraction \(i\).

---

# Attraction Utility

The baseline utility of an attraction is computed as

$$
U_i = r_i \cdot \log(n_i)
$$

where

- $r_i$ = Yelp rating  
- $n_i$ = review count  

This formulation captures both **quality** and **popularity**.

Utility values are normalized:

$$
\tilde{U}_i =
\frac{U_i - U_{min}}
{U_{max} - U_{min}}
$$

---

# Weather-Aware Congestion Estimation

Visitor congestion is estimated using a machine learning model trained on historical review density and weather conditions.

Let

$$
X = (temperature, precipitation, weekend, month)
$$

be the contextual feature vector.

An XGBoost regression model estimates expected review density:

$$
\hat{R} = f_{XGBoost}(X)
$$

Assuming a constant participation rate $p$, visitor density is estimated as

$$
\hat{V} = \frac{\hat{R}}{p}
$$

Expected waiting time is modeled using a queue-based approximation:

$$
W_i =
w_i \cdot
\log
\left(
1 + \frac{\hat{V}}{C_i}
\right)
$$

where

- $C_i$ = attraction capacity
- $w_i$ = base waiting coefficient

---

# Travel Time Estimation

Travel distance between attractions is calculated using geodesic distance:

$$
d_{ij} = \text{geodesic}(i,j)
$$

Travel time is estimated assuming average speed $v$:

$$
t_{ij} = \frac{d_{ij}}{v}
$$

---

# Visit Duration Model

Visit durations are sampled from category-specific distributions:

$$
T_i \sim \mathcal{N}(\mu_c, \sigma_c)
$$

where $c$ represents the attraction category.

This introduces stochastic variability in visitor behavior.

---

# Cost Model

Each attraction is assigned an estimated cost:

$$
c_i
$$

To discourage expensive attractions, a concave penalty is applied:

$$
P(c_i) = c_i^{p}
$$

where

$$
0 < p < 1
$$

This ensures that marginal cost penalties diminish for higher-cost attractions.

---

# Optimization Objective

The itinerary optimization maximizes overall visitor satisfaction while penalizing travel, congestion, and cost.

$$
\max
\sum_{i \in N} \tilde{U}_i x_i
-
\alpha \sum_{i \in N} \tilde{W}_i x_i
-
\beta \sum_{i,j \in N} \tilde{t}_{ij} y_{ij}
-
\gamma \sum_{i \in N} P(c_i) x_i
+
\lambda \sum_{i \in N} x_i
$$

where

- $\alpha$ = waiting time penalty  
- $\beta$ = travel cost weight  
- $\gamma$ = cost penalty weight  
- $\lambda$ = attraction exploration bonus

---

# Time Constraint

Total itinerary time must not exceed the daily budget $T$:

$$
\sum_{i \in N}
(T_i + \eta W_i)x_i
+
\sum_{i,j \in N}
t_{ij}y_{ij}
\le T
$$

where

$$
\eta
$$

represents the proportion of waiting time experienced during the visit.

---

# Budget Constraint

$$
\sum_{i \in N} c_i x_i \le B
$$

where $B$ is the travel budget.

---

# Route Consistency Constraints

Each visited attraction must have exactly one incoming and one outgoing edge:

$$
\sum_{j} y_{ij} = x_i
$$

$$
\sum_{i} y_{ij} = x_j
$$

---

# Subtour Elimination (MTZ Formulation)

To prevent disconnected routes:

$$
u_i - u_j + K y_{ij} \le K - 1
$$

where $K$ is the maximum number of visited attractions.

---

# Overall System Architecture

The system integrates **machine learning predictions with optimization decision making**:
Historical Data
│
▼
Weather + Reviews
│
▼
ML Congestion Prediction
│
▼
Waiting Time Estimation
│
▼
Mixed-Integer Optimization
│
▼
Recommended Travel Itinerary
----

---

# Why This Formulation Works

The model captures multiple realistic travel considerations:

- attraction popularity
- travel efficiency
- congestion conditions
- budget limitations
- visitor preferences

By combining **predictive modeling** with **optimization**, the system produces itineraries that are both **data-driven and operationally feasible**.
-------



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
