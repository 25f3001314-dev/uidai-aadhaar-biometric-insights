# Unlocking Societal Trends in Aadhaar Biometric Updates (Mar–Dec 2025)

## UIDAI Datacathon 2025 Submission

This project analyses Aadhaar biometric update data to uncover regional patterns, operational gaps, and time-based trends that can support evidence-driven government planning and service delivery improvements.

**Dataset Source:** UIDAI (Official, Anonymized)  
**Time Period:** March – December 2025  
**Coverage:** 36 States & Union Territories, 900+ Districts  
**Tools Used:** Python, Pandas, NumPy, Matplotlib, Google Colab  

---

## 1. Problem Statement & Objective

Aadhaar biometric updates are critical to maintaining accurate and inclusive identity records. However, update activity is not uniform across India and varies significantly by geography, age group, and time.

### Objectives of this Study
- Identify regions with high and low biometric update activity  
- Detect districts with near-zero or zero enrolment  
- Compare child (5–17) and adult (18+) update patterns  
- Analyse monthly and weekly trends  
- Detect unusual spikes or drops in activity  
- Translate analytical findings into actionable recommendations for government planning  

The analysis focuses on **clear, interpretable insights** rather than complex or black-box models.

---

## 2. Data Ingestion

- Aadhaar biometric update data was provided by UIDAI in multiple CSV files  
- All files were consolidated into a single dataset using Python  
- The final dataset contains approximately **1.86 million records**

### Key Fields Included
- Date  
- State / Union Territory  
- District  
- Pincode  
- Child (5–17) biometric updates  
- Adult (18+) biometric updates  

---

## 3. Data Understanding (Initial Checks)

Before analysis, the following checks were performed:
- Verified dataset size and structure  
- Confirmed absence of missing values in key columns  
- Validated the date range (March–December 2025)  
- Verified presence of all 36 States and Union Territories  
- Identified duplicate records for later handling  

**Conclusion:**  
The dataset is complete, reliable, and suitable for national, state, and district-level analysis.

---

## 4. Data Cleaning & Standardization

To ensure consistency and accuracy:
- Converted date fields into standard datetime format  
- Standardized state and Union Territory names to official administrative definitions  
- Cleaned district names and handled unknown values safely  
- Removed duplicate records using aggregation logic  
- Verified that biometric update counts are non-negative  

After cleaning, the dataset accurately represents unique district-level biometric activity over time.

---

## 5. Feature Engineering

Additional features were created to enable deeper analysis:
- Total biometric updates (child + adult)  
- Adult-to-child update ratio  
- Time-based features (year, month, day, weekday)  
- Region classification (State vs Union Territory)  

These features enabled demographic, geographic, and temporal comparisons.

---

## 6. Exploratory Data Analysis (EDA)

### Insight 1: Child vs Adult Aadhaar Updates (State-wise)
- States show distinct age-group update patterns  
- Some states are more child-focused, while others show higher adult update activity  
- Adult-heavy patterns are likely linked to workforce mobility  

This supports age-specific enrolment strategies.

---

### Insight 2: State-wise Enrolment Inequality
- A small number of states account for a large share of total updates  
- Union Territories and smaller states naturally show lower volumes  
- States and UTs require different service delivery approaches  

This insight supports targeted resource allocation.

---

### Insight 3: District Hotspots and Low-Enrolment Districts
- A limited number of districts act as biometric update hotspots  
- Several districts show extremely low or near-zero activity  
- Low activity often reflects access challenges or temporary inactivity  

This helps identify districts requiring infrastructure or outreach support.

---

### Insight 4: Near-Zero Enrolment Zones
- Districts with minimal activity were identified using data-driven thresholds  
- Some large states contain a high number of near-zero districts  
- These districts require close monitoring  

This insight supports mobile enrolment units and district-level interventions.

---

### Insight 5: Time Trends and Seasonality
- Biometric updates exhibit clear monthly variation  
- Certain weekdays consistently show higher activity  
- Seasonal drops align with holidays and administrative cycles  

This helps optimise staffing and scheduling.

---

### Insight 6: Anomaly Detection
- Rolling statistical methods were used to detect sudden spikes or drops  
- Detected anomalies likely reflect:
  - Campaign launches  
  - Temporary system outages  
  - Operational disruptions  

Anomalies indicate operational events rather than fraud.

---

### Insight 7: Districts with Zero Updates
- A very small number of districts recorded zero biometric updates  
- Zero activity does not necessarily indicate failure  
- Such districts should be reviewed individually  

This highlights the need for ground-level verification.

---

### Insight 8: States Requiring Priority Government Attention
Multiple factors were combined to identify priority states:
- Low overall activity  
- Presence of near-zero districts  
- Uneven district performance  
- Declining trends over time  

**Result:**  
A clear set of states where focused government intervention would have the highest impact.

---

## Key Takeaways
- Aadhaar biometric activity is uneven across geography and time  
- District-level analysis is critical, as state averages mask local gaps  
- Near-zero zones require monitoring rather than assumptions  
- Data-driven insights enable targeted and efficient interventions  

---

## Recommendations
- Prioritise resources for low-performing states and districts  
- Deploy mobile enrolment units in near-zero zones  
- Plan enrolment campaigns around seasonal and weekly trends  
- Monitor anomalies as early warning indicators  
- Share best practices from high-performing districts  

---

## Technical Summary
- **Platform:** Google Colab  
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib  
- **Outputs:** Power BI–ready CSV files  
- **Reproducibility:** Fully scripted and repeatable, with no manual steps  

---

## Streamlit Dashboard (Premium UI + Real-Time AI Insights)

An interactive Streamlit dashboard is included in `app.py` with notebook-aligned analytics:
- Child vs Adult updates (top states)
- Top/Bottom states by total updates
- States vs UT comparison
- Top/Bottom district hotspots
- Monthly and weekday trends
- Near-zero activity states
- Rolling anomaly detection
- Priority state ranking for government focus

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Data Input

Dashboard supports:
- **Demo data mode** (instant visuals without CSV)
- **Upload CSV mode** for your UIDAI cleaned dataset

Required columns (or close aliases handled automatically):
- `date`
- `state`
- `district`
- `pincode`
- `bio_age_5_17`
- `bio_age_17_`

### AI Insight API

API config is supported via Streamlit secrets in `.streamlit/secrets.toml`:
- `AISTAL_API_KEY`
- `AISTAL_API_ENDPOINT`
- `AISTAL_MODEL`

Use the **Generate Real-Time Insights** button in the app to get policy-ready AI insights from current dashboard filters.
