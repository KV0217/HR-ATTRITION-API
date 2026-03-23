from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, List, Dict
import joblib
import pandas as pd

app = FastAPI(
    title="HR Attrition Prediction API",
    description="Predicts employee attrition using XGBoost trained on IBM HR dataset — AUC 0.79",
    version="1.0.0"
)

model = joblib.load("attrition_model.joblib")

FEATURE_NAMES = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager', 'PromotionGap', 'IncomePerYear', 'TenureRatio',
    'IsNewEmployee', 'LowSatisfaction', 'PoorWorkLife', 'ManagerTenure', 'RiskScore'
]

DEPT_MAP    = {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2}
ROLE_MAP    = {
    'Healthcare Representative': 0, 'Human Resources': 1,
    'Laboratory Technician': 2,     'Manager': 3,
    'Manufacturing Director': 4,    'Research Director': 5,
    'Research Scientist': 6,        'Sales Executive': 7,
    'Sales Representative': 8
}
TRAVEL_MAP  = {'Non-Travel': 0, 'Travel_Frequently': 1, 'Travel_Rarely': 2}
EDU_MAP     = {
    'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2,
    'Medical': 3,         'Other': 4,          'Technical Degree': 5
}
MARITAL_MAP = {'Divorced': 0, 'Married': 1, 'Single': 2}


class EmployeeData(BaseModel):
    Age:                      int
    Gender:                   Literal["Male", "Female"]
    MaritalStatus:            Literal["Single", "Married", "Divorced"]
    Department:               Literal["Sales", "Research & Development", "Human Resources"]
    JobRole:                  Literal[
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ]
    Education:                int
    EducationField:           Literal[
        "Life Sciences", "Medical", "Marketing",
        "Technical Degree", "Human Resources", "Other"
    ]
    MonthlyIncome:            float
    JobLevel:                 int
    YearsAtCompany:           int
    YearsInCurrentRole:       int
    YearsSinceLastPromotion:  int
    YearsWithCurrManager:     int
    TotalWorkingYears:        int
    NumCompaniesWorked:       int
    BusinessTravel:           Literal["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
    OverTime:                 Literal["Yes", "No"]
    JobSatisfaction:          int
    WorkLifeBalance:          int
    EnvironmentSatisfaction:  int
    JobInvolvement:           int
    RelationshipSatisfaction: int
    PerformanceRating:        int
    PercentSalaryHike:        int
    StockOptionLevel:         int
    TrainingTimesLastYear:    int
    DistanceFromHome:         int
    DailyRate:                int
    HourlyRate:               int
    MonthlyRate:              int

    class Config:
        json_schema_extra = {
            "example": {
                "Age": 27, "Gender": "Male", "MaritalStatus": "Single",
                "Department": "Sales", "JobRole": "Sales Representative",
                "Education": 3, "EducationField": "Marketing",
                "MonthlyIncome": 2800, "JobLevel": 1, "YearsAtCompany": 1,
                "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 0,
                "YearsWithCurrManager": 1, "TotalWorkingYears": 2,
                "NumCompaniesWorked": 1, "BusinessTravel": "Travel_Frequently",
                "OverTime": "Yes", "JobSatisfaction": 1, "WorkLifeBalance": 1,
                "EnvironmentSatisfaction": 2, "JobInvolvement": 2,
                "RelationshipSatisfaction": 2, "PerformanceRating": 3,
                "PercentSalaryHike": 11, "StockOptionLevel": 0,
                "TrainingTimesLastYear": 2, "DistanceFromHome": 25,
                "DailyRate": 400, "HourlyRate": 45, "MonthlyRate": 10000
            }
        }


def get_suggestions(emp: EmployeeData) -> List[Dict]:
    suggestions = []
    replacement_cost = f"${emp.MonthlyIncome * 12 * 1.5:,.0f}"

    if emp.OverTime == "Yes":
        suggestions.append({"area": "Overtime", "current": "Working overtime",
            "recommended": "Reduce or eliminate overtime", "impact": "High",
            "estimated_attrition_reduction": "20-30%",
            "action": "Redistribute workload or add headcount",
            "replacement_cost_at_risk": replacement_cost})

    if emp.JobSatisfaction <= 2:
        suggestions.append({"area": "Job Satisfaction",
            "current": f"Score {emp.JobSatisfaction}/4 (Low)",
            "recommended": "Improve to 3+", "impact": "High",
            "estimated_attrition_reduction": "15-25%",
            "action": "1:1 with manager — uncover specific concerns immediately",
            "replacement_cost_at_risk": replacement_cost})

    if emp.WorkLifeBalance <= 2:
        suggestions.append({"area": "Work-Life Balance",
            "current": f"Score {emp.WorkLifeBalance}/4 (Poor)",
            "recommended": "Improve to 3+", "impact": "High",
            "estimated_attrition_reduction": "10-20%",
            "action": "Offer flexible hours or remote work options",
            "replacement_cost_at_risk": replacement_cost})

    if emp.YearsAtCompany <= 2:
        suggestions.append({"area": "New Employee Risk",
            "current": f"{emp.YearsAtCompany} year(s) at company",
            "recommended": "Structured onboarding for first 2 years", "impact": "High",
            "estimated_attrition_reduction": "15-20%",
            "action": "Assign mentor + enrol in structured onboarding programme",
            "replacement_cost_at_risk": replacement_cost})

    if emp.MonthlyIncome < 4000:
        suggestions.append({"area": "Compensation",
            "current": f"${emp.MonthlyIncome:,.0f}/month (below average)",
            "recommended": "Review against market benchmark", "impact": "Medium",
            "estimated_attrition_reduction": "10-15%",
            "action": "Immediate compensation review and market benchmarking",
            "replacement_cost_at_risk": replacement_cost})

    if emp.BusinessTravel == "Travel_Frequently":
        suggestions.append({"area": "Business Travel",
            "current": "Frequent travel", "recommended": "Reduce travel frequency",
            "impact": "Medium", "estimated_attrition_reduction": "8-12%",
            "action": "Replace in-person with virtual meetings where possible",
            "replacement_cost_at_risk": replacement_cost})

    if emp.YearsSinceLastPromotion >= 4:
        suggestions.append({"area": "Career Growth",
            "current": f"No promotion in {emp.YearsSinceLastPromotion} years",
            "recommended": "Promotion or career path discussion", "impact": "Medium",
            "estimated_attrition_reduction": "8-12%",
            "action": "Immediate career path discussion + promotion plan",
            "replacement_cost_at_risk": replacement_cost})

    if emp.YearsWithCurrManager <= 1:
        suggestions.append({"area": "Manager Relationship",
            "current": f"{emp.YearsWithCurrManager} year(s) with current manager",
            "recommended": "Strengthen manager-employee bond", "impact": "Low",
            "estimated_attrition_reduction": "5-8%",
            "action": "Schedule regular 1:1s and manager coaching",
            "replacement_cost_at_risk": replacement_cost})

    if not suggestions:
        suggestions.append({"area": "Overall", "current": "Good employee profile",
            "recommended": "Maintain current engagement", "impact": "Low",
            "estimated_attrition_reduction": "0%",
            "action": "Send recognition or loyalty reward to reinforce satisfaction",
            "replacement_cost_at_risk": "$0"})

    return suggestions


def encode(emp: EmployeeData) -> pd.DataFrame:
    overtime = 1 if emp.OverTime == "Yes" else 0
    gender   = 1 if emp.Gender == "Male" else 0
    tenure   = emp.YearsAtCompany
    role_yrs = emp.YearsInCurrentRole
    income   = emp.MonthlyIncome
    job_sat  = emp.JobSatisfaction
    wlb      = emp.WorkLifeBalance
    promo    = emp.YearsSinceLastPromotion

    row = {
        'Age': emp.Age, 'BusinessTravel': TRAVEL_MAP[emp.BusinessTravel],
        'DailyRate': emp.DailyRate, 'Department': DEPT_MAP[emp.Department],
        'DistanceFromHome': emp.DistanceFromHome, 'Education': emp.Education,
        'EducationField': EDU_MAP[emp.EducationField],
        'EnvironmentSatisfaction': emp.EnvironmentSatisfaction,
        'Gender': gender, 'HourlyRate': emp.HourlyRate,
        'JobInvolvement': emp.JobInvolvement, 'JobLevel': emp.JobLevel,
        'JobRole': ROLE_MAP[emp.JobRole], 'JobSatisfaction': job_sat,
        'MaritalStatus': MARITAL_MAP[emp.MaritalStatus],
        'MonthlyIncome': income, 'MonthlyRate': emp.MonthlyRate,
        'NumCompaniesWorked': emp.NumCompaniesWorked, 'OverTime': overtime,
        'PercentSalaryHike': emp.PercentSalaryHike,
        'PerformanceRating': emp.PerformanceRating,
        'RelationshipSatisfaction': emp.RelationshipSatisfaction,
        'StockOptionLevel': emp.StockOptionLevel,
        'TotalWorkingYears': emp.TotalWorkingYears,
        'TrainingTimesLastYear': emp.TrainingTimesLastYear,
        'WorkLifeBalance': wlb, 'YearsAtCompany': tenure,
        'YearsInCurrentRole': role_yrs, 'YearsSinceLastPromotion': promo,
        'YearsWithCurrManager': emp.YearsWithCurrManager,
        'PromotionGap': promo, 'IncomePerYear': income * 12,
        'TenureRatio': round(role_yrs / (tenure + 1), 3),
        'IsNewEmployee': int(tenure <= 2), 'LowSatisfaction': int(job_sat <= 2),
        'PoorWorkLife': int(wlb <= 2), 'ManagerTenure': emp.YearsWithCurrManager,
        'RiskScore': (
            overtime + int(job_sat <= 2) + int(wlb <= 2) +
            int(tenure <= 2) + int(income < 4000) +
            int(emp.BusinessTravel == 'Travel_Frequently') + int(promo >= 4)
        )
    }
    return pd.DataFrame([row])[FEATURE_NAMES]


@app.get("/")
def root():
    return {
        "message": "HR Attrition Prediction API is live",
        "model": "XGBoost — AUC 0.79 on IBM HR dataset (1,470 records)",
        "docs": "/docs", "version": "1.0"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(employee: EmployeeData):
    try:
        input_df        = encode(employee)
        prob            = round(float(model.predict_proba(input_df)[0][1]), 3)
        annual_salary   = employee.MonthlyIncome * 12
        replacement_cost= annual_salary * 1.5

        if prob >= 0.7:
            risk = "high"
            recommendation = "Immediate retention action required"
        elif prob >= 0.3:
            risk = "medium"
            recommendation = "Monitor closely — schedule check-in within 2 weeks"
        else:
            risk = "low"
            recommendation = "No immediate action needed"

        suggestions = get_suggestions(employee)

        return {
            "attrition_probability": prob,
            "prediction": "Will Leave" if prob > 0.4 else "Will Stay",
            "risk_level": risk,
            "recommendation": recommendation,
            "annual_salary": f"${annual_salary:,.0f}",
            "replacement_cost": f"${replacement_cost:,.0f}",
            "risk_score": int(input_df['RiskScore'].values[0]),
            "improvement_suggestions": suggestions,
            "summary": f"{len(suggestions)} area(s) identified for retention improvement"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(employees: list[EmployeeData]):
    results = [predict(e) for e in employees]
    return {"predictions": results, "count": len(results)}