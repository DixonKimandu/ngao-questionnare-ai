import os
import sys
import json
import time
from datetime import datetime
import requests
from sqlalchemy import create_engine, text
import schedule

# Add the parent directory to sys.path to import from other project modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import logging setup from logs module
from logs.fetch_cron_logs import setup_logger, get_log_level

# Load configuration
try:
    with open(os.path.join(parent_dir, "config.json"), "r") as f:
        config = json.load(f)
except FileNotFoundError:
    # Default configuration if file not found
    config = {
        "BASE_URL": os.getenv("BASE_URL", "https://inc-citizen.cabinex.co.ke"),
        "BATCH_SIZE": 100,
        "INTERVAL_MINUTES": 60,
        "CURSOR_FILE": os.path.join(parent_dir, "data", "last_cursor.json"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER", "ollama"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "ollama"),
        "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "192.168.100.46"),
        "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
        "POSTGRES_DB": os.getenv("POSTGRES_DB", "ollama"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO")
    }

# Set up logger with configuration
log_dir = os.path.join(parent_dir, "logs")
log_level = get_log_level(config.get("LOG_LEVEL", "INFO"))
logger = setup_logger("fetch_cron", log_dir, log_level)

if not os.path.exists(os.path.join(parent_dir, "config.json")):
    logger.warning("Config file not found. Using default configuration.")

# Function to fetch data from API with cursor support
def fetch_data_with_cursor(cursor=None, batch_size=100):
    """
    Fetch data from API endpoint with cursor support
    
    Args:
        cursor: Optional cursor position from previous fetch
        batch_size: Number of records to fetch at a time
        
    Returns:
        tuple: (transformed_data, new_cursor)
    """
    try:
        logger.info(f"Starting data fetch with cursor: {cursor}, batch_size: {batch_size}")
        
        # Authenticating
        auth_url = f"{config['BASE_URL']}/api/v1/user/login"
        auth_payload = {
            "telephone": config.get("API_USERNAME", "0797275002"),
            "password": config.get("API_PASSWORD", "1234")
        }
        auth_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        auth_response = requests.post(auth_url, json=auth_payload, headers=auth_headers)
        logger.info(f"Auth response status: {auth_response.status_code}")
            
        if auth_response.status_code != 200:
            logger.error(f"Authentication failed: {auth_response.text}")
            return [], None
        
        # Extract bearer token
        auth_data = auth_response.json()
        bearer_token = auth_data.get('data', {}).get('token')
        
        if not bearer_token:
            logger.error("No token in authentication response")
            return [], None
            
        logger.info("Authentication successful, got token")
        
        # Fetch questionnaire data with cursor parameter
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {bearer_token}"
        }

        # Build URL with cursor and batch size parameters
        questionnaire_url = f"{config['BASE_URL']}/api/v1/sub_module_data/module/8"
        params = {"limit": batch_size}
        
        if cursor:
            params["cursor"] = cursor
            
        logger.info(f"Fetching data from {questionnaire_url} with params {params}...")
            
        response = requests.get(questionnaire_url, headers=headers, params=params)
        logger.info(f"Data response status: {response.status_code}")
            
        if response.status_code == 200:
            raw_data = response.json()
            
            # Check structure
            if not isinstance(raw_data, dict) or 'data' not in raw_data:
                logger.error("Unexpected API response structure")
                return [], None
            
            # Get response items
            response_items = raw_data['data']
            
            if not isinstance(response_items, list):
                logger.error(f"Data is not a list, it's a {type(response_items)}")
                return [], None
                
            logger.info(f"Found {len(response_items)} responses in API data")

            # Get new cursor if provided in response
            new_cursor = raw_data.get('cursor', None)
            if not new_cursor and response_items:
                # If API doesn't provide cursor, use the last record ID as cursor
                new_cursor = response_items[-1].get('id', None)
            
            # Transform data
            transformed_data = []
            
            for response in response_items:
                # Extract common user info
                sub_module_id = response.get('sub_moduleId')
                user_data = response.get('user', {})
                iprs_data = user_data.get('iprs', {})
                sub_module_data = response.get('sub_module', {})
                id_no = iprs_data.get('id_no')
                gender = iprs_data.get('gender')
                dob = iprs_data.get('date_of_birth')
                location = iprs_data.get('district_of_birth') if iprs_data.get('district_of_birth') else iprs_data.get('county_of_birth') if iprs_data.get('county_of_birth') else iprs_data.get('division_of_birth') if iprs_data.get('division_of_birth') else iprs_data.get('location_of_birth') if iprs_data.get('location_of_birth') else None
                name = sub_module_data.get('name')
                description = sub_module_data.get('description')

                # Process question-answer pairs
                if 'formData' in response and isinstance(response['formData'], dict):
                    for question_id, answer in response['formData'].items():
                        record = {
                            'question_id': question_id,
                            'answer': answer,
                            'sub_module_id': sub_module_id,
                            'id_no': id_no,
                            'gender': gender,
                            'date_of_birth': dob,
                            'location': location,
                            'questionnaire_name': name,
                            'questionnaire_description': description
                        }
                        transformed_data.append(record)
            
            logger.info(f"Created {len(transformed_data)} question-answer records")
            logger.info(f"New cursor position: {new_cursor}")
            
            return transformed_data, new_cursor
        else:
            logger.error(f"Failed to fetch data: {response.status_code} - {response.text}")
            return [], None
            
    except Exception as e:
        logger.exception(f"Error fetching data: {e}")
        return [], None

# Function to store results in database
def store_results(results):
    """Store results in PostgreSQL database, avoiding duplicates"""
    if not results:
        logger.info("No results to store")
        return True
        
    try:
        # Create database URL
        db_url = f"postgresql://{config['POSTGRES_USER']}:{config['POSTGRES_PASSWORD']}@{config['POSTGRES_HOST']}:{config['POSTGRES_PORT']}/{config['POSTGRES_DB']}"
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        
        # Check existing records to avoid duplicates
        existing_records = {}
        with engine.connect() as conn:
            existing_query = text("""
                SELECT id_no, question_id, sub_module_id 
                FROM questionnaire_responses
            """)
            result = conn.execute(existing_query)
            
            for row in result:
                key = (row.id_no, row.question_id, row.sub_module_id)
                existing_records[key] = True
                
        logger.info(f"Found {len(existing_records)} existing records to check against")
        
        # Filter out duplicates
        new_records = []
        for record in results:
            key = (record.get('id_no'), record.get('question_id'), record.get('sub_module_id'))
            if None not in key and key not in existing_records:
                new_records.append(record)
                existing_records[key] = True
        
        if not new_records:
            logger.info("No new records to store")
            return True
            
        logger.info(f"Found {len(new_records)} new records to store")
        
        # Insert records in smaller batches 
        with engine.connect() as conn:
            batch_size = 100
            for i in range(0, len(new_records), batch_size):
                batch = new_records[i:i+batch_size]
                
                with conn.begin() as trans:
                    for record in batch:
                        conn.execute(
                            text("""
                                INSERT INTO questionnaire_responses 
                                (question_id, answer, sub_module_id, id_no, gender, date_of_birth, location, questionnaire_name, questionnaire_description)
                                VALUES (:question_id, :answer, :sub_module_id, :id_no, :gender, :date_of_birth, :location, :questionnaire_name, :questionnaire_description)
                            """),
                            {
                                "question_id": str(record.get('question_id', '')),
                                "answer": str(record.get('answer', '')),
                                "sub_module_id": record.get('sub_module_id'),
                                "id_no": record.get('id_no'),
                                "gender": record.get('gender'),
                                "date_of_birth": record.get('date_of_birth'),
                                "location": record.get('location'),
                                "questionnaire_name": record.get('questionnaire_name'),
                                "questionnaire_description": record.get('questionnaire_description')
                            }
                        )
                
            logger.info(f"Successfully stored {len(new_records)} new records in database")
            return True
        
    except Exception as e:
        logger.exception(f"Failed to store results: {e}")
        return False

# Function to save the cursor position
def save_cursor(cursor):
    """Save cursor to file for next run"""
    try:
        cursor_data = {
            "cursor": cursor,
            "last_updated": datetime.now().isoformat()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(config["CURSOR_FILE"]), exist_ok=True)
        
        with open(config["CURSOR_FILE"], "w") as f:
            json.dump(cursor_data, f)
            
        logger.info(f"Cursor saved: {cursor}")
        return True
    except Exception as e:
        logger.exception(f"Failed to save cursor: {e}")
        return False

# Function to load the last cursor
def load_cursor():
    """Load cursor from file"""
    try:
        if os.path.exists(config["CURSOR_FILE"]):
            with open(config["CURSOR_FILE"], "r") as f:
                cursor_data = json.load(f)
                logger.info(f"Loaded cursor: {cursor_data.get('cursor')} from {cursor_data.get('last_updated')}")
                return cursor_data.get("cursor")
        else:
            logger.info("No cursor file found, starting from beginning")
            return None
    except Exception as e:
        logger.exception(f"Failed to load cursor: {e}")
        return None

# Initialize the database
def initialize_database():
    """Initialize database tables"""
    try:
        # Create database URL
        db_url = f"postgresql://{config['POSTGRES_USER']}:{config['POSTGRES_PASSWORD']}@{config['POSTGRES_HOST']}:{config['POSTGRES_PORT']}/{config['POSTGRES_DB']}"
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        
        # Create tables if needed
        with engine.connect() as conn:
            # Ensure we're not in a transaction
            conn.execute(text("COMMIT"))
            
            # Create questionnaire_responses table
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS questionnaire_responses (
                    id SERIAL PRIMARY KEY,
                    question_id TEXT NOT NULL,
                    answer TEXT,
                    sub_module_id INTEGER,
                    id_no TEXT,
                    gender TEXT,
                    date_of_birth TEXT,
                    location TEXT,
                    questionnaire_name TEXT,
                    questionnaire_description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(id_no, question_id, sub_module_id)
                )
            """))
            
            # Create runs table to track cron job execution
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS cron_runs (
                    id SERIAL PRIMARY KEY,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    records_processed INTEGER,
                    records_added INTEGER,
                    cursor TEXT,
                    batch_size INTEGER,
                    status TEXT,
                    error_message TEXT
                )
            """))
            
            # Commit the changes
            conn.commit()
            
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.exception(f"Failed to initialize database: {e}")
        return False

# Main job function to run on schedule
def job():
    """Main job function that runs on schedule"""
    start_time = datetime.now()
    logger.info(f"Starting scheduled job at {start_time}")
    
    # Initialize counters for tracking
    records_processed = 0
    records_added = 0
    status = "success"
    error_message = None
    
    try:
        # Initialize database
        if not initialize_database():
            raise Exception("Failed to initialize database")
        
        # Get last cursor position
        cursor = load_cursor()
        batch_size = config.get("BATCH_SIZE", 100)
        
        # Fetch data with cursor
        results, new_cursor = fetch_data_with_cursor(cursor, batch_size)
        
        # Update processed counter
        records_processed = len(results)
        
        if results:
            # Store results in database
            success = store_results(results)
            if success:
                # Count newly added records (could also be returned by store_results)
                records_added = records_processed  # This is simplified
                
                # Save new cursor position if we have one
                if new_cursor:
                    save_cursor(new_cursor)
            else:
                status = "failed"
                error_message = "Failed to store results in database"
        else:
            logger.info("No new data to process")
            
    except Exception as e:
        logger.exception(f"Job failed: {e}")
        status = "failed"
        error_message = str(e)
    
    # Record job run in database
    try:
        end_time = datetime.now()
        db_url = f"postgresql://{config['POSTGRES_USER']}:{config['POSTGRES_PASSWORD']}@{config['POSTGRES_HOST']}:{config['POSTGRES_PORT']}/{config['POSTGRES_DB']}"
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            with conn.begin():
                conn.execute(
                    text("""
                        INSERT INTO cron_runs 
                        (start_time, end_time, records_processed, records_added, cursor, batch_size, status, error_message)
                        VALUES (:start_time, :end_time, :records_processed, :records_added, :cursor, :batch_size, :status, :error_message)
                    """),
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "records_processed": records_processed,
                        "records_added": records_added,
                        "cursor": load_cursor(),
                        "batch_size": batch_size,
                        "status": status,
                        "error_message": error_message
                    }
                )
        
        logger.info(f"Job completed. Status: {status}, Processed: {records_processed}, Added: {records_added}")
        
    except Exception as e:
        logger.exception(f"Failed to record job run: {e}")

# Schedule the job to run periodically
def run_scheduler():
    """Configure and run the scheduler"""
    interval_minutes = config.get("INTERVAL_MINUTES", 60)
    logger.info(f"Setting up scheduler to run every {interval_minutes} minutes")
    
    schedule.every(interval_minutes).minutes.do(job)
    
    # Run job immediately on startup
    logger.info("Running initial job on startup")
    job()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(1)

# For command-line execution
if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(parent_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(parent_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Handle command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Fetch questionnaire data on a schedule")
    parser.add_argument("--run-once", action="store_true", help="Run once and exit")
    parser.add_argument("--batch-size", type=int, help="Number of records to fetch at once")
    parser.add_argument("--interval", type=int, help="Interval in minutes between runs")
    parser.add_argument("--reset-cursor", action="store_true", help="Reset the cursor to start from beginning")
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.batch_size:
        config["BATCH_SIZE"] = args.batch_size
        logger.info(f"Batch size set to {args.batch_size} from command line")
        
    if args.interval:
        config["INTERVAL_MINUTES"] = args.interval
        logger.info(f"Interval set to {args.interval} minutes from command line")
        
    if args.reset_cursor and os.path.exists(config["CURSOR_FILE"]):
        os.remove(config["CURSOR_FILE"])
        logger.info("Cursor reset as requested")
    
    # Run once or start scheduler
    if args.run_once:
        logger.info("Running in single-run mode")
        job()
    else:
        logger.info("Starting scheduler")
        run_scheduler()
