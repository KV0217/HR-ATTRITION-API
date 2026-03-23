import streamlit as st
import pandas as pd
import requests

API_URL = "https://hr-attrition-api-qzeq.onrender.com"

st.set_page_config(page_title="HR Attrition Predictor", page_icon="🧑‍💼", layout="wide")
st.title("🧑‍💼 HR Employee Attrition Predictor")
st.markdown("Real-time attrition prediction powered by XGBoost — AUC 0.79 on IBM HR dataset.")

try:
    requests.get(f"{API_URL}/health", timeout=5)
    st.success("API is live and healthy")
except:
    st.warning("API is waking up — first prediction may take 30 seconds.")

tab1, tab2, tab3 = st.tabs(["👤 Single Employee", "📁 Batch Prediction", "💡 Retention Strategies"])

with tab1:
    st.markdown("### Enter Employee Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal**")
        age            = st.slider("Age", 18, 60, 30)
        gender         = st.radio("Gender", ["Male", "Female"], horizontal=True)
        marital        = st.radio("Marital Status", ["Single", "Married", "Divorced"])
        distance       = st.slider("Distance From Home (km)", 1, 30, 5)
        education      = st.slider("Education Level (1-5)", 1, 5, 3)
        edu_field      = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing",
            "Technical Degree", "Human Resources", "Other"])

    with col2:
        st.markdown("**Job Details**")
        department     = st.selectbox("Department", [
            "Sales", "Research & Development", "Human Resources"])
        job_role       = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"])
        job_level      = st.slider("Job Level (1-5)", 1, 5, 2)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000)
        overtime       = st.radio("OverTime", ["Yes", "No"], horizontal=True)
        travel         = st.radio("Business Travel", [
            "Non-Travel", "Travel_Rarely", "Travel_Frequently"])

    with col3:
        st.markdown("**Experience & Satisfaction**")
        years_company  = st.slider("Years at Company", 0, 40, 3)
        years_role     = st.slider("Years in Current Role", 0, 20, 2)
        years_promo    = st.slider("Years Since Last Promotion", 0, 15, 1)
        years_manager  = st.slider("Years With Current Manager", 0, 17, 2)
        total_working  = st.slider("Total Working Years", 0, 40, 8)
        num_companies  = st.slider("Num Companies Worked", 0, 9, 2)
        job_sat        = st.slider("Job Satisfaction (1-4)", 1, 4, 3)
        wlb            = st.slider("Work Life Balance (1-4)", 1, 4, 3)
        env_sat        = st.slider("Environment Satisfaction (1-4)", 1, 4, 3)
        job_inv        = st.slider("Job Involvement (1-4)", 1, 4, 3)
        rel_sat        = st.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
        perf_rating    = st.slider("Performance Rating (1-4)", 1, 4, 3)
        salary_hike    = st.slider("Percent Salary Hike", 10, 25, 13)
        stock          = st.slider("Stock Option Level (0-3)", 0, 3, 0)
        training       = st.slider("Training Times Last Year", 0, 6, 3)
        daily_rate     = st.number_input("Daily Rate", 100, 1500, 800)
        hourly_rate    = st.number_input("Hourly Rate", 30, 100, 65)
        monthly_rate   = st.number_input("Monthly Rate", 2000, 27000, 14000)

    if st.button("🔍 Predict Attrition", type="primary"):
        payload = {
            "Age": age, "Gender": gender, "MaritalStatus": marital,
            "Department": department, "JobRole": job_role,
            "Education": education, "EducationField": edu_field,
            "MonthlyIncome": monthly_income, "JobLevel": job_level,
            "YearsAtCompany": years_company, "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo,
            "YearsWithCurrManager": years_manager,
            "TotalWorkingYears": total_working,
            "NumCompaniesWorked": num_companies,
            "BusinessTravel": travel, "OverTime": overtime,
            "JobSatisfaction": job_sat, "WorkLifeBalance": wlb,
            "EnvironmentSatisfaction": env_sat, "JobInvolvement": job_inv,
            "RelationshipSatisfaction": rel_sat,
            "PerformanceRating": perf_rating,
            "PercentSalaryHike": salary_hike, "StockOptionLevel": stock,
            "TrainingTimesLastYear": training, "DistanceFromHome": distance,
            "DailyRate": daily_rate, "HourlyRate": hourly_rate,
            "MonthlyRate": monthly_rate
        }

        with st.spinner("Predicting..."):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                result   = response.json()
            except Exception as e:
                st.error(f"API call failed: {e}")
                st.stop()

        st.markdown("---")
        st.markdown("### Prediction Results")

        prob       = result["attrition_probability"]
        prediction = result["prediction"]
        risk       = result["risk_level"]
        rec        = result["recommendation"]
        risk_score = result.get("risk_score", "-")
        annual_sal = result.get("annual_salary", "-")
        rep_cost   = result.get("replacement_cost", "-")
        summary    = result.get("summary", "")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Attrition Probability", f"{prob*100:.1f}%")
        c2.metric("Prediction", prediction)
        c3.metric("Risk Level",
            "🔴 High" if risk == "high" else "🟡 Medium" if risk == "medium" else "🟢 Low")
        c4.metric("Annual Salary", annual_sal)
        c5.metric("Replacement Cost", rep_cost)

        if risk == "high":
            st.error(f"⚠️ {rec}")
        elif risk == "medium":
            st.warning(f"⚡ {rec}")
        else:
            st.success(f"✅ {rec}")

        suggestions = result.get("improvement_suggestions", [])
        if suggestions:
            st.markdown(f"### 🎯 Retention Improvement Areas — {summary}")
            df_s = pd.DataFrame(suggestions)

            def color_impact(val):
                if val == "High":
                    return "background-color: #ffcccc; color: #8b0000; font-weight: bold"
                elif val == "Medium":
                    return "background-color: #fff3cc; color: #7d5a00; font-weight: bold"
                else:
                    return "background-color: #ccffcc; color: #1a5c1a; font-weight: bold"

            st.dataframe(
                df_s.style.applymap(color_impact, subset=["impact"]),
                use_container_width=True, hide_index=True
            )

            st.markdown("### ✅ Recommended Actions")
            for _, row in df_s.iterrows():
                if row["impact"] == "High":
                    st.error(f"🔴 **{row['area']}** — {row['action']} *(Est. reduction: {row['estimated_attrition_reduction']})*")
                elif row["impact"] == "Medium":
                    st.warning(f"🟡 **{row['area']}** — {row['action']} *(Est. reduction: {row['estimated_attrition_reduction']})*")
                else:
                    st.info(f"🟢 **{row['area']}** — {row['action']} *(Est. reduction: {row['estimated_attrition_reduction']})*")

with tab2:
    st.markdown("### Batch Attrition Prediction")
    st.markdown("Upload a CSV with employee details — results include attrition probability + replacement cost.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} employees")
        st.dataframe(batch_df.head())
        if st.button("Run Batch Prediction"):
            payload = batch_df.to_dict(orient="records")
            with st.spinner(f"Predicting for {len(payload)} employees..."):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch", json=payload, timeout=60)
                    results  = response.json()["predictions"]
                    out_df   = batch_df.copy()
                    out_df["attrition_probability"] = [r["attrition_probability"] for r in results]
                    out_df["prediction"]            = [r["prediction"] for r in results]
                    out_df["risk_level"]            = [r["risk_level"] for r in results]
                    out_df["replacement_cost"]      = [r["replacement_cost"] for r in results]
                    st.success(f"Done! Predicted for {len(results)} employees.")
                    st.dataframe(out_df, use_container_width=True)
                    st.download_button("Download Results CSV",
                        out_df.to_csv(index=False),
                        "attrition_predictions.csv", "text/csv")
                except Exception as e:
                    st.error(f"Batch prediction failed: {e}")

with tab3:
    st.markdown("### 📊 HR Retention Strategy Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### 1. Overtime Management
        - Overtime is the single strongest attrition predictor
        - **Action:** Cap overtime at 10% of workforce, redistribute workload

        #### 2. New Employee Onboarding
        - Highest attrition in first 2 years
        - **Action:** 30-60-90 day structured onboarding + mentor assignment

        #### 3. Compensation Review
        - Below-market pay ($<4,000/month) strongly predicts leaving
        - **Action:** Annual market benchmarking for all roles
        """)
    with col2:
        st.markdown("""
        #### 4. Career Development
        - No promotion in 4+ years = high flight risk
        - **Action:** Annual career path discussions for all employees

        #### 5. Work-Life Balance
        - Poor WLB score correlates with 2x attrition rate
        - **Action:** Flexible hours, remote options, travel reduction

        #### 6. Manager Relationships
        - New manager = transition risk period
        - **Action:** Structured 1:1s during first 6 months with new manager
        """)

    st.markdown("---")
    st.markdown("### 💰 Cost of Attrition Calculator")
    col1, col2 = st.columns(2)
    with col1:
        headcount      = st.number_input("Total Headcount", 10, 10000, 100)
        avg_salary     = st.number_input("Average Monthly Salary ($)", 1000, 20000, 5000)
        attrition_rate = st.slider("Current Attrition Rate (%)", 1, 50, 16)
    with col2:
        annual_leavers  = int(headcount * attrition_rate / 100)
        cost_per_leaver = avg_salary * 12 * 1.5
        total_cost      = annual_leavers * cost_per_leaver
        saved_10pct     = total_cost * 0.1
        st.metric("Employees Leaving/Year", annual_leavers)
        st.metric("Cost Per Leaver", f"${cost_per_leaver:,.0f}")
        st.metric("Total Annual Attrition Cost", f"${total_cost:,.0f}")
        st.metric("Savings if Attrition Reduced 10%", f"${saved_10pct:,.0f}")
