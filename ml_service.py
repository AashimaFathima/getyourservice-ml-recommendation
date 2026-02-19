from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import pandas as pd
import numpy as np
import mysql.connector
import traceback
import os
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="GetYourService ML Recommendation API")

# ---------- Pydantic Model ----------
from pydantic import BaseModel, Field

class Weights(BaseModel):
    rating: float = 0.6
    experience: float = 0.25
    charges: float = 0.15

class RecommendationRequest(BaseModel):
    user_id: Optional[str] = None
    city: Optional[str] = None
    category_id: Optional[int] = None
    weights: Optional[Weights] = Weights()
    method: Optional[str] = "auto"



# ---------- MySQL Configuration ----------
db_config = {
    'host': os.getenv("DB_HOST"),
    'user': os.getenv("DB_USER"),
    'password': os.getenv("DB_PASSWORD"),
    'database': os.getenv("DB_NAME")
}


INTERACTION_THRESHOLD = 15

recommendation_cache = {}


def get_user_ratings():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    query = """
    SELECT ub.user_id AS user_id,
           b.provider_id AS provider_id,
           r.rating    AS rating,
           b.id        AS booking_id
    FROM user_bookings ub
    JOIN booking b ON ub.bookings_id = b.id
    LEFT JOIN review r  ON r.booking_id = b.id
    """
    cursor.execute(query)
    result = cursor.fetchall()
    
    conn.close()

    user_ratings = {}
    for row in result:
        uid = str(row['user_id'])
        pid = str(row['provider_id'])
        rating = float(row['rating']) if row['rating'] is not None else 0.0
        booking_id = row['booking_id']

        user_ratings.setdefault(uid, {}).setdefault(pid, []).append({'booking_id': booking_id, 'rating': rating})
    return user_ratings

def get_providers_info(city=None, category_id=None):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT 
            s.user_id AS provider_id,
            MAX(u.experince) AS experience,
            MAX(s.cost) AS charges,
            COALESCE(AVG(r.rating), 0) AS avg_rating
        FROM services s
        JOIN user u ON u.id = s.user_id
        LEFT JOIN booking b ON b.service_id = s.id
        LEFT JOIN review r ON r.booking_id = b.id
        WHERE (%s IS NULL OR s.category_id = %s)
         AND (%s IS NULL OR u.city = %s)
        GROUP BY s.user_id;

    """
    cursor.execute(query, (category_id, category_id, city, city))
    data = cursor.fetchall()
    conn.close()

    providers_info = {}
    for row in data:
        provider_id = str(row['provider_id'])
        providers_info[provider_id] = {
            'rating': float(row['avg_rating']),
            'experience': float(row['experience']) if row.get('experience') is not None else 0.0,
            'charges': float(row['charges']) if row.get('charges') is not None else 0.0
        }
    return providers_info
    

def get_providers_info_with_user(user_id=None, city=None, category_id=None):
    providers_info = get_providers_info(city, category_id)
    if user_id:
        user_ratings = get_user_ratings()
        for pid in user_ratings.get(user_id, {}):
            if pid not in providers_info:
                conn = mysql.connector.connect(**db_config)
                cursor = conn.cursor(dictionary=True)
                cursor.execute("""
                    SELECT IFNULL(u.experince,0) AS experience, IFNULL(s.cost,0) AS charges,
                           COALESCE(AVG(r.rating), 0) AS avg_rating
                    FROM services s
                    JOIN user u ON u.id = s.user_id
                    LEFT JOIN booking b ON b.service_id = s.id
                    LEFT JOIN review r ON r.booking_id = b.id
                    WHERE s.user_id = %s
                    GROUP BY u.experince, s.cost
                """, (pid,))
                row = cursor.fetchone()
                conn.close()
                if row:
                    providers_info[pid] = {
                        'rating': float(row['avg_rating']),
                        'experience': float(row['experience']),
                        'charges': float(row['charges'])
                    }
    return providers_info

def user_based_cf(user_id: str, providers_info, top_n=5):
    user_ratings = get_user_ratings()
    if user_id not in user_ratings:
        return []

    allowed_providers = set(providers_info.keys())
    all_users = list(user_ratings.keys())
    all_providers = allowed_providers

    # Step 1: Build rating matrix
    matrix = pd.DataFrame(index=all_users, columns=list(all_providers), dtype=float).fillna(0.0)
    for u in all_users:
        for p, ratings in user_ratings[u].items():
            if p in allowed_providers:
                avg_rating = np.mean([r['rating'] for r in ratings]) if ratings else 0
                matrix.loc[u, p] = avg_rating

    # Step 2: Compute user means
    user_mean = matrix.replace(0, np.nan).mean(axis=1).fillna(0)

    # Step 3: Normalize by subtracting user mean
    matrix_norm = matrix.sub(user_mean, axis=0).fillna(0)

    # Step 4: Compute similarity with cosine
    user_vec = matrix_norm.loc[user_id].values
    sim_scores = {}
    for other_user in all_users:
        if other_user == user_id:
            continue
        vec = matrix_norm.loc[other_user].values
        norm_user = np.linalg.norm(user_vec)
        norm_vec = np.linalg.norm(vec)
        sim = np.dot(user_vec, vec) / (norm_user * norm_vec) if norm_user > 0 and norm_vec > 0 else 0
        sim_scores[other_user] = sim

    # Step 5: Predict ratings using weighted sum of deviations
    pred_ratings = {}
    for pid in all_providers:
        if pid in user_ratings[user_id]:
            continue 

        numerator, denominator = 0.0, 0.0
        for other_user, sim in sim_scores.items():
            if pid in user_ratings[other_user]:
                avg_rating_other = np.mean([r['rating'] for r in user_ratings[other_user][pid]])
                numerator += sim * (avg_rating_other - user_mean[other_user])
                denominator += abs(sim)
        if denominator > 0:
            pred = user_mean[user_id] + (numerator / denominator)
            pred = max(0, min(5, pred))
            pred_ratings[pid] = pred

    # Step 6: Return top N recommendations
    top_pred = sorted(pred_ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
    results = []
    for pid, pred_rating in top_pred:
        info = providers_info.get(pid, {'rating': 0.0, 'experience': 0.0, 'charges': 0.0})
        results.append({
            'id': pid,
            'rating': float(info.get('rating', 0.0)),
            'predicted_rating': round(pred_rating, 2),
            'source': 'collaborative',
            'experience': float(info.get('experience', 0.0)),
            'charges': float(info.get('charges', 0.0))
        })
    return results

def content_based_recommend(providers_info, weights, top_n=5):
    df = pd.DataFrame(providers_info).T
    if df.empty:
        return []
    df['id'] = df.index
    w = weights or {}
    max_charges = df["charges"].max() if df["charges"].max() > 0 else 1.0
    df["score"] = (
        df["rating"] * w.get("rating", 0) +
        df["experience"] * w.get("experience", 0) +
        (1 - df["charges"]/max_charges) * w.get("charges", 0)
    )
    df = df.sort_values("score", ascending=False).head(top_n)
    results = []
    for _, row in df.iterrows():
        results.append({
            'id': str(row["id"]),
            'score': round(float(row["score"]), 2),
            'rating': float(row["rating"]),
            'source': 'content-based',
            'experience': float(row.get("experience", 0.0)),
            'charges': float(row.get("charges", 0.0))
        })
    return results

# ---------- FastAPI Endpoint ----------
@app.post("/recommend")
def recommend(req: RecommendationRequest):
    try:
        if not req.user_id or str(req.user_id).strip() == "":
            raise HTTPException(status_code=400, detail="user_id is required")


        cache_key = f"{req.user_id}_{req.city}_{req.category_id}_{req.method}"

        if cache_key in recommendation_cache:
            return recommendation_cache[cache_key]

        providers_info = get_providers_info_with_user(req.user_id, req.city, req.category_id)
        user_ratings = get_user_ratings()
        num_interactions = sum(len(ratings) for ratings in user_ratings.get(req.user_id, {}).values())

        # -------- HYBRID SWITCH --------
        if req.method == "collaborative" or (req.method == "auto" and num_interactions >= INTERACTION_THRESHOLD):
            recs = user_based_cf(req.user_id, providers_info)
            if recs:
                recommendation_cache[cache_key] = recs
                return recs

        recs = content_based_recommend(providers_info, req.weights)

        recommendation_cache[cache_key] = recs

        return recs

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "ML recommendation service is running"}
