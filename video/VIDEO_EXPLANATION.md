## Demo

A full system demonstration video is available here: [▶ Watch Demo Video](https://drive.google.com/file/d/1dQ5XhXZd15FpspZ7uDQxfmmG2JeJlZco/view?usp=sharing) 

The video demonstrates three phases:

### Phase 1 – Booking Flow
- User logs in and books a service provider
- Provider receives booking notification
- Provider accepts and marks service as completed
- Review is submitted and stored in the database

This phase validates the end-to-end backend and database integration.

---

### Phase 2 – Collaborative Filtering
- Access `/docs` (Swagger UI)
- Call POST `/recommend`
- Uses historical user–provider interactions
- Returns top-N providers ranked by `predicted_rating`

Here:
- `rating` = global average provider rating  
- `predicted_rating` = personalized estimate for that specific user  

Ranking is based on `predicted_rating`.

---

### Phase 3 – Content-Based Filtering
- Modify weights in the request JSON
- Adjust importance of rating / experience / cost
- Returns top-N providers ranked by computed `score`

Here:
- `score` is calculated using the weighted formula
- No predicted rating is used
- Ranking is based purely on weighted scoring


> Note: The service providers, pricing, descriptions, and ratings used in this demo are sample data created for demonstration purposes as part of a larger academic project.  
