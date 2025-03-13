# Questionnaire Analysis Dashboard

## Overview

This project provides a comprehensive dashboard for analyzing questionnaire data with demographic breakdowns. It fetches survey responses from an API, processes the data, and presents visualizations and AI-generated insights based on demographic categories (age, gender, location).

## Features

- **Data Retrieval**: Automatically fetches questionnaire data from a configurable API endpoint
- **Demographic Analysis**: Breaks down responses by age groups, gender, and location
- **Question-Level Analysis**: Detailed analysis of individual questions with demographic segmentation
- **Interactive Visualizations**: Charts and graphs showing response distributions
- **AI-Generated Insights**: Automated analysis of trends and patterns in the data
- **Caching System**: Efficient data caching to improve performance

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/questionnaire-analysis-dashboard.git
   cd questionnaire-analysis-dashboard
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure the API endpoint by creating a `.env` file:
   ```
   POLLS_API_ENDPOINT=https://backend
   ```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run inc_analysis_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## Project Structure

- `inc_analysis_dashboard.py`: Main Streamlit dashboard application
- `inc_polls_analysis_agent.py`: Backend analysis workflow and AI agents
- `data/questionnaire_model.py`: Pydantic models for data validation
- `.env`: Configuration file for API endpoints

## Requirements

- Python 3.8+
- Streamlit
- Pandas
- Plotly
- Matplotlib
- Pydantic
- Agno (for AI agent workflow)
- Ollama (for local LLM inference)

## How It Works

1. The dashboard fetches data from the configured API endpoint
2. Responses are categorized by demographic segments (age, gender, location)
3. The system analyzes overall demographic distributions
4. Users can select specific questions for detailed analysis
5. AI agents generate insights about patterns and trends in the data
6. Visualizations are created to show response distributions

## AI Analysis Components

The system uses three specialized AI agents:
- **Data Retrieval Agent**: Handles fetching and processing raw data
- **Data Analysis Agent**: Analyzes trends by demographic categories
- **Insights Agent**: Generates high-level insights and recommendations

## Customization

You can customize the API endpoint by modifying the `.env` file. The dashboard also supports different LLM backends by modifying the agent configurations in `inc_polls_analysis_agent.py`.

## License

[Specify your license here]

## Contributors

[List contributors here] 