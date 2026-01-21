# AWS Cost Optimization Tool

## Overview
This project is an advanced AWS Cost Optimization and Pricing Analysis tool. It leverages multi-agent AI systems to provide data-driven pricing recommendations, demand forecasting, and "What-If" scenario analysis. The tool is designed to help organizations optimize their cloud spend and improve ROI through intelligent pricing strategies.

### Demo Video

<div align="center">
  <video src="demo_video.mp4" controls width="100%" style="max-height: 500px;">
    Your browser does not support the video tag.
  </video>
</div>


## Detailed Features
- **Dynamic Pricing Orchestrator**: A sequential multi-agent pipeline that processes pricing from base calculation through competitive and inventory adjustments.
- **Scenario Simulation (What-If)**: Real-time simulation of market changes, competitor price drops, or promotional events with instant ROI impact analysis.
- **Agentic Workflow Trace**: Visual representation of the decision-making process for every pricing recommendation, showing the logic of each specialized agent.
- **Integrated Forecast Dashboard**: Visualizes historical trends alongside predicted future demand and revenue growth.
- **Automated SOP Guardrails**: Built-in compliance checks for minimum margin, maximum discount depth, and approval workflow triggers.
- **Natural Language Scenario Parsing**: Interpret complex market scenarios written in plain English to configure simulations automatically.
- **Global Data Filtering**: Drill down into pricing metrics by SKU, sales channel (MT, GT, ECOM), and geographic region.

## AI & Machine Learning Architecture
The project employs a hybrid AI architecture combining traditional statistical models with modern deep learning and LLM capabilities:

### 1. Multi-Agent System (MAS)
- **Base Agent**: Uses historical median smoothing and seasonality indexing for baseline pricing.
- **Promo Agent**: Heuristic-based optimization considering historical promo depth and SOP guardrails.
- **Competitor Agent**: Real-time market gap analysis and response positioning (Conservative/Aggressive).
- **Inventory Agent**: Supply-chain signal processing to adjust prices based on stock levels and "Days of Cover".

### 2. Machine Learning Models
- **Regression Analysis**: Gradient Boosting Regressors and Random Forests for demand prediction based on price sensitivity.
- **Time-Series Forecasting**: Exponential Smoothing (Holt-Winters) and ARIMA models for capturing weekly and monthly seasonality.
- **Elasticity Estimation**: Log-log OLS regression for calculating Price Elasticity of Demand (PED) at the SKU and Category level.

### 3. Large Language Models (LLM)
- **AWS Bedrock Integration**: Utilizes models like Claude or GPT (via Bedrock Runtime) for complex reasoning tasks.
- **LLM-Powered Explanations**: Generates human-readable rationales for why specific price changes are recommended, translating agent weights into business insights.
- **NL Scenario Interpretation**: Advanced parsing of user-defined scenarios into structured parameter overrides for the simulation engine.

## Usage
Used by several companies as tools to optimize their AWS costs and streamline their pricing operations.

## Technology Stack
- **Frontend**: React 18+, Tailwind CSS, Lucide React, Recharts.
- **Backend**: FastAPI (Python 3.8+), Pandas, NumPy.
- **ML Lifecycle**: Scikit-Learn, Statsmodels, SciPy.
- **AI Infrastructure**: AWS Bedrock SDK (boto3) for LLM orchestration.

## Installation

### Prerequisites
- Node.js (v16+)
- Python (3.8+)
- npm or yarn

### Frontend Setup
1. Navigate to the project directory.
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```

### Backend Setup
1. Navigate to the backend directory (if applicable, or root).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the backend server:
   ```bash
   python app.py
   ```
