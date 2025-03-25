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

4. Configure the environment by creating a `.env` file based on `.env.example`:
   ```
   BASE_URL=https://inc-citizen.cabinex.co.ke
   POSTGRES_DB=assistant
   POSTGRES_USER=assistant
   POSTGRES_PASSWORD=assistant
   POSTGRES_HOST=192.168.1.37
   POSTGRES_PORT=5432
   OLLAMA_HOST=http://192.168.1.37:11434
   ```

## Usage

### Dashboard Application

Run the Streamlit dashboard:
```bash
streamlit run questionnaire_analysis.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

### Background Data Fetcher

Set up the background data fetcher to run as a service:

1. Run manually once:
   ```bash
   python cron/fetch_cron.py --run-once
   ```

2. Run as a scheduled job:
   ```bash
   python cron/fetch_cron.py
   ```

3. Set up as a system service (Linux):
   ```bash
   # Create a systemd service file
   sudo nano /etc/systemd/system/questionnaire-fetcher.service
   
   # Add the following content
   [Unit]
   Description=Questionnaire Data Fetcher Service
   After=network.target

   [Service]
   User=your_username
   WorkingDirectory=/path/to/questionnaire-analysis-dashboard
   ExecStart=/path/to/questionnaire-analysis-dashboard/.venv/bin/python cron/fetch_cron.py
   Restart=always
   RestartSec=5

   [Install]
   WantedBy=multi-user.target
   
   # Enable and start the service
   sudo systemctl enable questionnaire-fetcher.service
   sudo systemctl start questionnaire-fetcher.service
   ```

## Project Structure

- `questionnaire_analysis.py`: Main Streamlit dashboard application
- `cron/fetch_cron.py`: Background service for data synchronization
- `data/questionnaire_model.py`: Pydantic models for data validation
- `logs/`: Directory for application logs
- `reports/`: Generated PDF reports
- `.env`: Configuration file for API endpoints and database

## Requirements

- Python 3.8+
- PostgreSQL database
- Streamlit
- Pandas
- Plotly
- SQLAlchemy
- Pydantic
- Agno (for AI agent workflow)
- Ollama (for local LLM inference)

## How It Works

1. The `fetch_cron.py` service periodically fetches new questionnaire data from the API
2. Data is stored in a PostgreSQL database, avoiding duplicates
3. The Streamlit dashboard connects to the database to visualize the data
4. Responses are categorized by demographic segments (age, gender, location)
5. Users can select specific questions for detailed analysis
6. AI agents generate insights about patterns and trends in the data
7. The dashboard can export data as CSV or generate PDF reports

## AI Analysis Components

The system uses specialized AI agents:
- **Data Retrieval Agent**: Handles fetching and processing raw data
- **Data Analysis Agent**: Analyzes trends by demographic categories
- **General Analysis Agent**: Generates reports with high-level insights and recommendations

## Customization

You can customize the API endpoint and database connection by modifying the `.env` file. The dashboard also supports different LLM backends by modifying the Ollama host configuration.

## License

[Specify your license here]

## Contributors

[List contributors here] 