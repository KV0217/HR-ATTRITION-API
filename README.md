# HR Attrition Prediction API

Production REST API serving an XGBoost attrition model — AUC 0.79 on IBM HR data.

## Live
| | URL |
|--|--|
| API | https://hr-attrition-api-qzeq.onrender.com |
| Docs | https://hr-attrition-api-qzeq.onrender.com/docs |
| Streamlit | https://hr-attrition-api-kv.streamlit.app |



## Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /predict | Single employee prediction |
| POST | /predict/batch | Batch predictions |

## What Makes This Different
- Returns annual salary + replacement cost ($) per employee
- Ranked improvement suggestions with estimated attrition reduction %
- Risk score out of 7 based on overtime, satisfaction, tenure, income, travel

## Run Locally
```bash
git clone https://github.com/KV0217/HR-Attrition-API.git
cd HR-Attrition-API
pip install -r requirements.txt
uvicorn main:app --reload
```

## Tech Stack
FastAPI · XGBoost · Scikit-learn · Docker · Render

## Related
- Analysis notebook: [HR-Attrition](https://github.com/KV0217/HR-Attrition)
