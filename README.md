# 🚀 GetYourService – ML Recommendation Microservice

This repository contains the FASTAPI-Based ML Service powering the recommendation engine of **GetYourService**, a smart service-booking platform.

The system implements a **hybrid recommendation model** combining:

-   Content-Based Filtering (CBF)
    
-   User-Based Collaborative Filtering (CF)
    
-   Auto-adaptive switching based on interaction history
    

The ML service is built using **FastAPI** and integrates with a Spring Boot backend and MySQL database.


## 🏗 Architecture

```
Spring Boot Backend
        ↓
FastAPI ML Microservice (/recommend)
        ↓
MySQL Database
  • users
  • bookings
  • reviews
  • services
```


The backend sends contextual data to the ML service:

-   user\_id
    
-   city
    
-   category\_id
    
-   weights
    
-   method
    

The ML service returns the top-N ranked providers in JSON format.

## 🔄 Hybrid Recommendation Logic

### Interaction Threshold

`INTERACTION_THRESHOLD = 15`

### Auto Mode Behavior

-   If user interactions < 15 → Content-Based Filtering
    
-   If user interactions ≥ 15 → Collaborative Filtering
    
-   If CF produces no results → Fallback to Content-Based
    

## Content-Based Filtering (CBF)

Used for:

-   Cold-start users
    
-   Low interaction users
    

### Scoring Formula


```python
score =
    rating_weight     * rating +
    experience_weight * experience +
    charges_weight    * normalized_cost
```

Default weights:


```json
{
  "rating": 0.6,
  "experience": 0.25,
  "charges": 0.15
}
```

Returns:

-   score
    
-   rating
    
-   experience
    
-   charges
    

Note: `score` is a ranking value, not a predicted rating.

## Collaborative Filtering (CF)

Used for:

-   Active users (≥ 15 interactions)
    

### Steps

1.  Build user–provider interaction matrix
    
2.  Mean normalize ratings
    
3.  Compute cosine similarity
    
4.  Predict ratings using neighborhood weighted aggregation
    
5.  Return top 5 providers
    

### 📐Prediction Formula

`r̂(u,p) = user_mean(u) + weighted_deviation`

Returns:

-   rating (global average)
    
-   predicted\_rating (personalized estimate)
    
-   experience
    
-   charges
    

## 🔗 API Endpoint

### POST `/recommend`


## ⚡Performance Considerations

-   In-memory caching implemented for repeated queries
    
-   Mean normalization prevents user bias
    
-   Cosine similarity ensures personalized ranking
    
-   Hybrid fallback handles cold-start scenarios
    

## 🛠 Technologies Used

-   Python 3.11
    
-   FastAPI
    
-   Pandas
    
-   NumPy
    
-   MySQL
    
-   Uvicorn
    

## ▶ Running Locally

1️⃣ Set environment variables:

```
DB_HOST=localhost 
DB_USER=your_username 
DB_PASSWORD=your_password 
DB_NAME=getyourservice
```

2️⃣ Start the server:

`uvicorn ml_service:app --reload`

Swagger UI available at:

`http://localhost:8000/docs`

## 💎 Project Strengths

-   Hybrid adaptive recommendation
    
-   Personalized predictions
    
-   Explainable scoring
    
-   Cold-start handling
    
-   Backend–ML integration
    
    
## 🎬 Demo

A full system demonstration video is available in the `/video` directory of this repository.