📘 PRODUCT REQUIREMENTS DOCUMENT (PRD)
Project Title

E-Commerce Intelligence Suite: The Smart Manager’s Dashboard

🧭 1. Executive Summary

The E-Commerce Intelligence Suite is a data-driven, interactive dashboard built to help small-scale e-commerce store managers make smarter inventory and sales decisions.
Version 1 focuses on two powerful insights:

Cross-Selling Engine – Discover which products are bought together.

Inventory Forecaster – Predict product demand for upcoming periods.

The tool transforms raw transaction data into actionable recommendations through an intuitive web interface — deployable directly on Streamlit Cloud.

🎯 2. Objectives & Goals
Goal	Description
📊 Increase sales	Identify cross-selling opportunities.
🧮 Optimize inventory	Forecast stock needs accurately.
💻 Empower non-technical users	Provide clean visuals and plain-language insights.
☁️ Deploy easily	Streamlit-based online deployment with minimal setup.
👥 3. Target Audience

Primary User: Retail store owner / e-commerce manager

Profile: Limited technical knowledge, familiar with sales data and basic reports

Pain Points Solved:

Can’t identify which items sell together

Struggles to forecast future sales accurately

Finds spreadsheets confusing or static

🏗️ 4. System Overview (Version 1 Focus)

The suite will be a Streamlit web application that integrates:

Analytical backend (Python) – for cleaning, analysis, and forecasting

Relational warehouse (SQLite/MySQL) – for structured star-schema storage

Interactive UI (Streamlit + Plotly) – for visualization and user interaction

🧩 5. Core Features – Version 1 (Deliverable for this Semester)
5.1 Cross-Selling Engine (Market Basket Analysis)

Goal: Recommend products that are frequently purchased together.

Workflow:

Load and clean transactional data (remove cancelled orders, handle missing values).

Group by InvoiceNo to form baskets.

Apply Apriori Algorithm (via mlxtend) to extract frequent itemsets.

Generate association rules using Support, Confidence, and Lift.

Display simple, easy-to-read rules:

Example: “Customers who buy Garden Hose often buy Watering Can.”

UI Elements:

Input field → Select or search product name

Output → List of recommended items + bar chart showing support/confidence

Tooltip → Explains each metric in one sentence

5.2 Inventory Forecaster (Time-Series Regression)

Goal: Predict future demand for each product.

Workflow:

Aggregate daily/weekly sales quantity per product.

Apply ARIMA model (statsmodels) for forecasting next N days.

Display forecast with upper & lower confidence intervals.

Highlight actionable insights, e.g.:

“Expected demand = 120 units; consider restocking.”

UI Elements:

Dropdown → Select product

Chart → Line plot with actual vs. forecasted sales (Plotly)

Info box → “Suggested stock level” summary

5.3 Upload-Your-Own-Data Module

Goal: Make the dashboard dynamic and personalized.

Workflow:

User uploads a CSV (predefined column structure).

Backend validates schema (column names, date format).

Data stored temporarily in SQLite or in-memory Pandas DF.

Both Apriori & ARIMA rerun automatically → Dashboard refreshes.

UI Elements:

File Uploader + Validation Status indicator

Sample CSV download option

“Processing…” spinner + completion success message

5.4 Dashboard Layout
Sidebar:
 ├── Home
 ├── Cross-Selling
 ├── Sales Forecast
 └── Upload Data

Main Area:
 [Header]  – Project Title + Summary
 [Tabs]    – Each analysis module
 [Charts]  – Plotly bar/line/pie charts
 [Insight] – Text summary boxes


Design Tone:
Professional yet friendly → large text labels, icon-based navigation, muted corporate palette (navy + white + teal).

⚙️ 6. Technical Architecture
6.1 Tech Stack
Layer	Component	Tool
Frontend/UI	Streamlit, Plotly	For visualization and user interaction
Logic Layer	Python, Pandas, NumPy, Mlxtend, Statsmodels	For data processing & ML
Database	SQLite (local) / MySQL (cloud)	Implements star schema
Deployment	Streamlit Cloud	Hosted interactive web app
6.2 Star-Schema Design

Fact Table:
FactSales (InvoiceNo, ProductID, CustomerID, DateID, Quantity, TotalPrice)

Dimension Tables:

DimProduct (ProductID, Description, UnitPrice)

DimCustomer (CustomerID, Country)

DimDate (DateID, Date, Month, Year, Weekday)

6.3 Data Flow
User Upload → Validation → ETL to Warehouse
           ↓
Apriori Model     ARIMA Model
           ↓             ↓
Association Rules   Sales Forecast
           ↓             ↓
       Streamlit UI Dashboard

🔐 7. Non-Functional Requirements
Type	Specification
Performance	Load & render under 5 seconds for ≤ 100k rows
Security	No external data storage; temporary caching only
Usability	Clear metrics, legends, and callouts
Portability	Must run locally or on Streamlit Cloud
Reliability	Must handle missing/invalid rows gracefully
🧱 8. Development Milestones
Phase	Description	Deliverable
Phase 1	Data Cleaning + Warehouse setup	SQLite DB with Fact & Dim tables
Phase 2	Cross-Selling Engine	Association rules + UI tab
Phase 3	Inventory Forecaster	ARIMA forecast + UI tab
Phase 4	Upload Module + Dashboard Integration	Final integrated dashboard
Phase 5	Testing & Deployment	Working Streamlit Cloud app
🚀 9. Version Roadmap
🩵 Version 1.0 (Current Semester Target)

Complete all core features above.

Deliver functional Streamlit dashboard hosted online.

SQL (SQLite/MySQL) backend.
✅ This version will be fully developed and submitted.

🧡 Version 2.0 (Future Expansion – “Smart Commerce”)

Add:

Customer Segmentation (K-Means) → Segment high-value vs. new customers.

Dynamic Pricing Engine → Rule-based price suggestions using forecast data.

Advanced Reports Page → Summaries by month, country, or customer segment.

DB Update: Integrate MySQL for persistent multi-user data.
Goal: Move toward predictive + prescriptive analytics.

💜 Version 3.0 (Future Expansion – “Intelligent Retail Suite”)

Add:

Sentiment Analysis (NLP) on product reviews (TextBlob / NLTK).

MongoDB Atlas integration for unstructured review data.

“Product Health” Tab – Combine sales forecast + sentiment trends.

Live Data API Connector for real-time updates.

Goal: Full-scale intelligent assistant for store management.

🧠 10. Example Use-Case Flow (Version 1)

Manager logs in → Sees “Upload your data” option.

Uploads sales CSV → System cleans & loads data.

Navigates to “Cross-Selling” → Selects product → Gets top 3 related items.

Switches to “Forecast” → Chooses product → Sees demand trend & stock suggestion.

Applies insights → Restocks & cross-promotes accordingly.

🎨 11. Visual Design & UX Guidelines

Palette: White background + Navy headers + Teal accents

Typography: Sans-serif, large readable fonts

Charts:

Bar chart → Cross-selling metrics

Line chart → Sales forecast

Pie chart → Category contribution

Icons: Simple (🛒, 📈, 📦) for intuitive navigation

Layout Principle: “One insight per screen” to avoid clutter

📦 12. Deliverables to Windsurf

app.py (main Streamlit file)

/data (cleaned dataset or sample CSV)

/models (Apriori & ARIMA modules)

/utils (data cleaning & DB setup scripts)

requirements.txt

README.md with setup & deployment steps

🧭 13. Success Metrics
Metric	Target
Data upload success rate	≥ 95 %
Dashboard load time	< 5 s
Forecast accuracy (MAPE)	≤ 20 %
User comprehension rate (from tests)	≥ 90 % understand “recommendations” & “forecast” outputs
🪜 14. Future Scalability Notes

Replace SQLite with MySQL in v2.

Add MongoDB Atlas in v3 for flexible data.

Containerize app (Docker) for enterprise-grade deployment.

Integrate authentication (Streamlit Login Component) if multi-user needed.

✅ 15. Summary

This PRD authorizes the complete implementation of Version 1 only.
It ensures a fully functional, user-friendly, and analytically sound dashboard deployed on Streamlit Cloud.
Versions 2 and 3 are documented here for future continuity and scalability.

End of PRD Document — Windsurf Implementation Ready