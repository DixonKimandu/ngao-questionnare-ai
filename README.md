# Questionnaire Analysis Dashboard

## Overview

This project provides a comprehensive dashboard for analyzing questionnaire data with demographic breakdowns. It fetches survey responses from an API, stores them in a PostgreSQL database, and presents visualizations and AI-generated insights based on demographic categories (age, gender, location).

## Features

- **Data Retrieval**: Automatically fetches questionnaire data from a configurable API endpoint and stores in PostgreSQL
- **Demographic Analysis**: Breaks down responses by age groups, gender, and location
- **Question-Level Analysis**: Detailed analysis of individual questions with demographic segmentation
- **Interactive Visualizations**: Charts and graphs showing response distributions
- **AI-Generated Insights**: Automated analysis of trends and patterns in the data
- **Caching System**: Efficient data caching to improve dashboard performance
- **Background Data Synchronization**: Cron job service to keep database updated with latest responses

## Project Structure

- `docker-compose.yml`: Docker Compose configuration for containerized deployment
- `Dockerfile`: Docker configuration for the dashboard application

## Requirements

### For Local Installation:
- Python 3.8+
- PostgreSQL database
- Streamlit
- Pandas
- Plotly
- SQLAlchemy
- Pydantic
- Agno (for AI agent workflow)
- Ollama (for local LLM inference)

### For Docker Deployment:
- Docker
- Docker Compose

## Installation

### Option 1: Local Installation

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

### Option 2: Docker Deployment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/questionnaire-analysis-dashboard.git
   cd questionnaire-analysis-dashboard
   ```

2. Build and run using Docker Compose:
   ```bash
   docker-compose up -d
   ```

   This will start:
   - The main dashboard application
   - PostgreSQL database
   - Background data fetcher service
   - Ollama LLM service

3. Access the dashboard at `http://localhost:8501`

## Configuration

Configure the environment by creating a `.env` file based on `.env.example`:
```
BASE_URL=""
POSTGRES_DB=
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_HOST=192.168.1.xx
POSTGRES_PORT=5432
OLLAMA_HOST=http://192.168.1.xx:11434
```

b. Configure the environment by creating a `.streamlit.toml` file based on `.streamlit.example.toml`:
```
BASE_URL=
POSTGRES_DB=
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_HOST=192.168.1.xx
POSTGRES_PORT=5432
OLLAMA_HOST=http://192.168.1.xx:11434
```

## Usage

### Dashboard Application

Run the Streamlit dashboard:
```bash
streamlit run questionnaire_analysis.py
```