import sys
import os
import streamlit as st
from datetime import datetime
import requests
import json
from typing import Optional, Dict, Any, List
from sqlalchemy import create_engine, text
import pandas as pd
import plotly.express as px

# Set page to wide mode by default
st.set_page_config(layout="wide")

# Get API endpoint from environment variable with fallback
BASE_URL = st.secrets.BASE_URL
# Define cache timeout (4 hours in seconds)
CACHE_TIMEOUT = 4 * 60 * 60

# Placeholder for title - we'll update it after getting data
title_placeholder = st.empty()

# Initial placeholder text
title_placeholder.title("Loading Questionnaire...")

st.write("This is the analysis page for NGAO Questionnaires.")

SUBMODULE_ID = st.query_params["sub-module"]

# Add a function to get the questionnaire name
@st.cache_data(ttl=CACHE_TIMEOUT)
def get_questionnaire_name(submodule_id):
    """Fetch questionnaire name from the database"""
    try:
        # Create database URL from secrets
        db_url = f"postgresql://{st.secrets['POSTGRES_USER']}:{st.secrets['POSTGRES_PASSWORD']}@{st.secrets['POSTGRES_HOST']}:{st.secrets['POSTGRES_PORT']}/{st.secrets['POSTGRES_DB']}"
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            # Query to get questionnaire name from submodule
            query = text("""
                SELECT questionnaire_name FROM questionnaire_responses WHERE sub_module_id = :submodule_id
                LIMIT 1
            """)
            
            result = conn.execute(query, {"submodule_id": submodule_id}).fetchone()
            
            if result:
                # Get the name from the result
                try:
                    # Try as dictionary if available
                    name = result._mapping['name']
                except (AttributeError, KeyError):
                    # Fall back to index access
                    name = result[0]
                
                return name
            
            # If no name found, return default
            return f"Questionnaire {submodule_id}"
            
    except Exception as e:
        st.warning(f"Failed to get questionnaire name: {e}")
        return f"Questionnaire {submodule_id}"

# Get questionnaire name
questionnaire_name = get_questionnaire_name(SUBMODULE_ID)

# Update the title with the actual questionnaire name
title_placeholder.title(f"{questionnaire_name} Analytics")

@st.cache_data(ttl=CACHE_TIMEOUT)
def fetch_data():
    """Fetch data from Database"""
    try:
        # Create database URL from secrets
        db_url = f"postgresql://{st.secrets['POSTGRES_USER']}:{st.secrets['POSTGRES_PASSWORD']}@{st.secrets['POSTGRES_HOST']}:{st.secrets['POSTGRES_PORT']}/{st.secrets['POSTGRES_DB']}"
            
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        
        # Default values
        results = None
        count_result = 0

        # Check if the table exists and has data
        with engine.connect() as conn:
            # Check table existence
            table_check = conn.execute(text("""
            SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_name = 'questionnaire_responses'
            )
            """))
            table_exists = table_check.scalar()

            if table_exists:
                # Count records
                count_query = text("SELECT COUNT(*) FROM questionnaire_responses WHERE sub_module_id = :SUBMODULE_ID")
                count_result = conn.execute(count_query, {"SUBMODULE_ID": SUBMODULE_ID}).scalar()

                # Check record sample if exists
                if count_result > 0:
                    sample_query = text("SELECT * FROM questionnaire_responses LIMIT 1")
                    sample_result = conn.execute(sample_query).fetchone()
                    if sample_result:
                        # Use _mapping attribute to access as dictionary
                        try:
                            sample_dict = dict(sample_result._mapping)
                        except AttributeError:
                            st.write("Sample record (could not convert to dict):", sample_result)

                # Fetch all records
                query = text("SELECT * FROM questionnaire_responses WHERE sub_module_id = :SUBMODULE_ID")
                results = conn.execute(query, {"SUBMODULE_ID": SUBMODULE_ID}).fetchall()

        return results, count_result

    except Exception as e:
        st.warning(f"Database query failed: {e}. Attempting to load fallback data.")
        
        # Get the path to the questionnaire_data.json file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "..", "data", "questionnaire_data.json")
        
        try:
            # Try to load the local JSON file
            with open(json_path, 'r') as file:
                fallback_data = json.load(file)
            st.info("Successfully loaded local fallback data.")
            # Return fallback data with a count of 0 to maintain consistent return format
            return fallback_data, 0
        except Exception as json_error:
            st.error(f"Error loading fallback data: {json_error}")
            # Return None with count 0 for consistent return structure
            return None, 0

def use_in_memory_data(data):
    """Display visualizations using in-memory data"""
    if not data:
        st.warning("No data available for visualization.")
        return
            
    df = pd.DataFrame(data)
    
    # Check if the dataframe is empty or has the expected columns
    if df.empty:
        st.warning("In-memory data is empty.")
        return
    
    # Verify required columns exist
    required_columns = ['question_id', 'answer', 'id_no', 'gender']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"In-memory data is missing required columns: {', '.join(missing_columns)}")
        st.write("Available columns:", df.columns.tolist())
        return
    
    # Overview metrics
    st.markdown('<div class="sub-header">In-Memory Data Overview</div>', unsafe_allow_html=True)
    
    # Count unique respondents (by id_no)
    unique_respondents = df['id_no'].nunique()
    st.metric("Total Respondents", unique_respondents)
    
    # Add metrics for unique males and females
    df['gender_clean'] = df['gender'].apply(
        lambda x: 'Not Specified' if pd.isna(x) or x == '' else x
    )
    gender_counts_unique = df[['id_no', 'gender_clean']].drop_duplicates().gender_clean.value_counts()
    
    unique_males = gender_counts_unique.get('male', 0) + gender_counts_unique.get('Male', 0)
    st.metric("Males", unique_males)
    
    unique_females = gender_counts_unique.get('female', 0) + gender_counts_unique.get('Female', 0)
    st.metric("Females", unique_females)
    
    with col3:
        # Calculate average age
        # First ensure age column exists by calculating from date of birth
        if 'age' not in df.columns:
            df['age'] = pd.to_datetime(df['date_of_birth']).dt.tz_localize(None).apply(
                lambda x: (datetime.now() - x).days // 365
            )
        
        # Get unique respondents with their ages
        age_df = df[['id_no', 'age']].drop_duplicates()
        avg_age = round(age_df['age'].mean(), 1)  # Round to 1 decimal place
        st.metric("Average Age", avg_age)
    
    # Gender distribution
    st.subheader("Gender Distribution")
    gender_counts = df[['id_no', 'gender']].drop_duplicates().gender.value_counts().reset_index()
    gender_counts.columns = ['Gender', 'Count']
    
    # Create a pie chart using Plotly
    fig = px.pie(
        gender_counts, 
        values='Count', 
        names='Gender',
        title='Distribution by Gender',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        #hole=0.3,  # Creates a donut chart if desired
    )
    
    # Customize layout
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    # Display the pie chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Question analysis
    st.subheader("Question Analysis")
    
    # Get unique questions
    questions = df['question_id'].unique()
    
    # Create database URL from secrets
    db_url = f"postgresql://{st.secrets['POSTGRES_USER']}:{st.secrets['POSTGRES_PASSWORD']}@{st.secrets['POSTGRES_HOST']}:{st.secrets['POSTGRES_PORT']}/{st.secrets['POSTGRES_DB']}"
        
    # Create SQLAlchemy engine
    engine = create_engine(db_url)
    
    # Loop through all questions and display their analysis
    for question_id in questions:
        # Create a section for each question
        st.markdown(f"### Question: {question_id}")
        
        # Run a specific query for the current question
        with engine.connect() as conn:
            question_query = text("""
                SELECT * FROM questionnaire_responses
                WHERE question_id = :question_id
            """)
            
            question_result = conn.execute(question_query, {"question_id": question_id})
            
            # Convert to DataFrame properly - same approach as above
            try:
                # Try using _mapping attribute
                question_rows = [dict(row._mapping) for row in question_result]
            except AttributeError:
                # Fall back to older approach
                column_names = question_result.keys()
                question_rows = [dict(zip(column_names, row)) for row in question_result]
            
            question_data = pd.DataFrame(question_rows) if question_rows else pd.DataFrame()
        
        # Display answer distribution
        st.write(f"Answer distribution:")
        answer_counts = question_data['answer'].value_counts()
        st.bar_chart(answer_counts)
        
        # Create two columns for gender and location breakdowns
        col1, col2 = st.columns(2)
        
        with col1:
            # Show answers by gender
            st.write("Answers by gender:")
            # Handle empty gender values by replacing them with 'Not Specified'
            question_data['gender_clean'] = question_data['gender'].apply(
                lambda x: 'Not Specified' if pd.isna(x) or x == '' else x
            )
            gender_answer = pd.crosstab(question_data['answer'], question_data['gender_clean'])
        st.bar_chart(gender_answer)
        
        with col2:
            # Show answers by location
            st.write("Answers by location:")
            # Handle empty location values similarly
            question_data['location_clean'] = question_data['location'].apply(
                lambda x: 'Not Specified' if pd.isna(x) or x == '' else x
            )
            location_answer = pd.crosstab(question_data['answer'], question_data['location_clean'])
            st.bar_chart(location_answer)
        
        # Add a divider between questions
        st.markdown("---")

def display_vizualizations(data=None, count_result=None):
    """Display visualizations based on database query results"""
    try:
        # Process the data without using 'with'
        if data is None:
            st.warning("No data available for visualization.")
            return
            
        # Better way to convert SQLAlchemy results to DataFrame
        try:
            # Try using _mapping attribute which is available in newer SQLAlchemy versions
            rows = [dict(row._mapping) for row in data]
        except (AttributeError, TypeError):
            # Fall back to other approaches based on the type of data
            if isinstance(data, list) and all(isinstance(item, dict) for item in data):
                # If it's already a list of dictionaries
                rows = data
            elif hasattr(data, 'keys'):
                # If it's a result set with keys method
                column_names = data.keys()
                rows = [dict(zip(column_names, row)) for row in data]
            else:
                # Try a generic approach
                rows = []
                for row in data:
                    if hasattr(row, '_asdict'):
                        # Named tuple approach
                        rows.append(row._asdict())
                    elif isinstance(row, tuple):
                        # Assume first we have column names from elsewhere
                        if hasattr(data, 'keys'):
                            columns = data.keys()
                            rows.append(dict(zip(columns, row)))
                        else:
                            # Use generic column names
                            rows.append({f"col_{i}": val for i, val in enumerate(row)})
                    elif isinstance(row, dict):
                        # Direct dict
                        rows.append(row)
                    else:
                        # Unknown type, try converting to string
                        rows.append({"value": str(row)})
        
        # Then create DataFrame from the list of dictionaries
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        
        if df.empty:
            st.warning("No data available in database for visualization.")
            # Fall back to passed-in data if available
            if data:
                st.info("Using in-memory data instead of database data.")
                use_in_memory_data(data)
            else:
                st.error("No data available in database or in-memory. Please refresh data.")
                return
        else:
            # Continue with database-based visualization
            # Overview metrics
            st.subheader("Repository Overview")
                           
            col1, col2 = st.columns(2)
            
            with col1:
                # Count unique respondents (by id_no)
                unique_respondents = df['id_no'].nunique()
                st.metric("Total Respondents", unique_respondents)
            
            col1, col2, col3 = st.columns(3)

            with col1:
                # Count unique males
                df['gender_clean'] = df['gender'].apply(
                    lambda x: 'Not Specified' if pd.isna(x) or x == '' else x
                )
                gender_counts_unique = df[['id_no', 'gender_clean']].drop_duplicates().gender_clean.value_counts()
                
                unique_males = gender_counts_unique.get('male', 0) + gender_counts_unique.get('Male', 0)
                st.metric("Males", unique_males)
            
            with col2:
                # Count unique females
                unique_females = gender_counts_unique.get('female', 0) + gender_counts_unique.get('Female', 0)
                st.metric("Females", unique_females)
            
            with col3:
                # Calculate average age
                # First ensure age column exists by calculating from date of birth
                if 'age' not in df.columns:
                    df['age'] = pd.to_datetime(df['date_of_birth']).dt.tz_localize(None).apply(
                        lambda x: (datetime.now() - x).days // 365
                    )
                
                # Get unique respondents with their ages
                age_df = df[['id_no', 'age']].drop_duplicates()
                avg_age = round(age_df['age'].mean(), 1)  # Round to 1 decimal place
                st.metric("Average Age", avg_age)

            with col1:
                # Gender distribution
                st.subheader("Gender Distribution")
                gender_counts = df[['id_no', 'gender']].drop_duplicates().gender.value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']
                
                # Create a pie chart using Plotly
                fig_gender = px.pie(
                    gender_counts, 
                    values='Count', 
                    names='Gender',
                    title='Distribution by Gender',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    # hole=0.3,  # Creates a donut chart if desired
                )
                
                # Customize layout for gender chart
                fig_gender.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                # Display the gender pie chart within the column
                st.plotly_chart(fig_gender, use_container_width=True)
            
            with col2:
                # Age Distribution
                st.subheader("Age Distribution")
                
                # Calculate age from date_of_birth, handling timezone-naive dates
                df['age'] = pd.to_datetime(df['date_of_birth']).dt.tz_localize(None).apply(
                    lambda x: (datetime.now() - x).days // 365
                )
                
                # Create age groups
                def categorize_age(age):
                    if age < 18:  # Handle edge case
                        return "Under 18"
                    elif 18 <= age <= 35:
                        return "18-35"
                    elif 36 <= age <= 65:
                        return "36-65" 
                    else:
                        return "65+"
                
                # Apply the age categorization
                df['age_group'] = df['age'].apply(categorize_age)
                
                # Count by age group (based on unique respondents)
                age_counts = df[['id_no', 'age_group']].drop_duplicates().age_group.value_counts().reset_index()
                age_counts.columns = ['Age Group', 'Count']
                
                # Ensure the age groups are displayed in a logical order
                age_order = ["Under 18", "18-35", "36-65", "65+"]
                age_counts['Age Group'] = pd.Categorical(
                    age_counts['Age Group'], 
                    categories=age_order, 
                    ordered=True
                )
                age_counts = age_counts.sort_values('Age Group')
                
                # Create a pie chart using Plotly
                fig_age = px.pie(
                    age_counts, 
                    values='Count', 
                    names='Age Group',
                    title='Distribution by Age Group',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )
                
                # Customize layout for age chart
                fig_age.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                # Display the age pie chart within the column
                st.plotly_chart(fig_age, use_container_width=True)
            
            with col3:
                # Location Distribution
                st.subheader("Location Distribution")

                # Add location chart code here when available
                location_counts = df[['id_no', 'location']].drop_duplicates().location.value_counts().reset_index()
                location_counts.columns = ['Location', 'Count']
                
                # Create a pie chart using Plotly
                fig_location = px.pie(
                    location_counts, 
                    values='Count', 
                    names='Location',
                    title='Distribution by Location',
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    # hole=0.3,  # Creates a donut chart if desired
                )
                
                # Customize layout for gender chart
                fig_location.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=0, xanchor="center", x=0.5),
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                
                # Display the location pie chart within the column
                st.plotly_chart(fig_location, use_container_width=True)
            
            # Question analysis
            st.subheader("Question Analysis")
            
            # Get unique questions
            questions = df['question_id'].unique()
            
            # Create database URL from secrets
            db_url = f"postgresql://{st.secrets['POSTGRES_USER']}:{st.secrets['POSTGRES_PASSWORD']}@{st.secrets['POSTGRES_HOST']}:{st.secrets['POSTGRES_PORT']}/{st.secrets['POSTGRES_DB']}"
                
            # Create SQLAlchemy engine
            engine = create_engine(db_url)

            # Loop through all questions and display their analysis
            for question_id in questions:
                # Create a section for each question
                st.markdown(f"### Question: {question_id}")
                
                # Run a specific query for the current question
                with engine.connect() as conn:
                    question_query = text("""
                        SELECT * FROM questionnaire_responses
                        WHERE question_id = :question_id
                    """)
                    
                    question_result = conn.execute(question_query, {"question_id": question_id})
                    
                    # Convert to DataFrame properly - same approach as above
                    try:
                        # Try using _mapping attribute
                        question_rows = [dict(row._mapping) for row in question_result]
                    except AttributeError:
                        # Fall back to older approach
                        column_names = question_result.keys()
                        question_rows = [dict(zip(column_names, row)) for row in question_result]
                    
                    question_data = pd.DataFrame(question_rows) if question_rows else pd.DataFrame()
                
                # Display answer distribution
                st.write(f"Answer distribution:")
                answer_counts = question_data['answer'].value_counts()
                st.bar_chart(answer_counts)
                
                # Create two columns for gender and location breakdowns
                col1, col2 = st.columns(2)
                
                with col1:
                # Show answers by gender
                    st.write("Answers by gender:")
                    # Handle empty gender values by replacing them with 'Not Specified'
                    question_data['gender_clean'] = question_data['gender'].apply(
                        lambda x: 'Not Specified' if pd.isna(x) or x == '' else x
                    )
                    gender_answer = pd.crosstab(question_data['answer'], question_data['gender_clean'])
                st.bar_chart(gender_answer)

                with col2:
                    # Show answers by location
                    st.write("Answers by location:")
                    # Handle empty location values similarly
                    question_data['location_clean'] = question_data['location'].apply(
                        lambda x: 'Not Specified' if pd.isna(x) or x == '' else x
                    )
                    location_answer = pd.crosstab(question_data['answer'], question_data['location_clean'])
                st.bar_chart(location_answer)
                
                # Add a divider between questions
                st.markdown("---")
                 
    except Exception as e:
        import traceback
        st.warning(f"Failed to query database for visualization: {str(e)}")
        st.error(f"Error details: {type(e).__name__}: {str(e)}")
        st.code(traceback.format_exc())
        
        # Use in-memory data as fallback
        st.info("Falling back to in-memory data...")
        use_in_memory_data(data)

def clear_questionnaire_data():
    """Clear all data from the questionnaire_responses table and reset the cache"""
    try:
        # Create database URL from secrets
        db_url = f"postgresql://{st.secrets['POSTGRES_USER']}:{st.secrets['POSTGRES_PASSWORD']}@{st.secrets['POSTGRES_HOST']}:{st.secrets['POSTGRES_PORT']}/{st.secrets['POSTGRES_DB']}"
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)
        
        # Delete all records from the database table
        with engine.connect() as conn:
            with conn.begin():
                # Get count before deletion for feedback
                count_query = text("SELECT COUNT(*) FROM questionnaire_responses")
                count_result = conn.execute(count_query).scalar()
                
                # Delete all records
                delete_query = text("DELETE FROM questionnaire_responses")
                conn.execute(delete_query)
        
        # Clear the Streamlit cache to force a fresh data fetch
        fetch_data.clear()
        
        # Clear session state data
        if 'data' in st.session_state:
            st.session_state.data = {}
        
        # Reset last fetch time
        if 'last_fetch_time' in st.session_state:
            del st.session_state.last_fetch_time
        
        st.success(f"Successfully cleared {count_result} records from the database and reset the cache.")
        return True
        
    except Exception as e:
        st.error(f"Failed to clear data: {str(e)}")
        return False

def clear_cache_only():
    """Clear only the in-memory cache without affecting the database"""
    try:
        # Clear the Streamlit cache to force a fresh data fetch
        fetch_data.clear()
        
        # Clear session state data
        if 'data' in st.session_state:
            st.session_state.data = {}
        
        # Reset last fetch time
        if 'last_fetch_time' in st.session_state:
            del st.session_state.last_fetch_time
        
        st.success("Successfully cleared the in-memory cache. Data will be refreshed on next load.")
        return True
        
    except Exception as e:
        st.error(f"Failed to clear cache: {str(e)}")
        return False

def generate_and_get_report():
    """Generate a PDF report using the GeneralAnalysisWorkflow and return the file contents"""
    try:
        from agents.general_analysis_agent import GeneralAnalysisWorkflow
        from datetime import datetime
        import os
        import traceback
        
        # Add debug output
        # st.info(f"Starting report generation for submodule: {SUBMODULE_ID}")
        
        # Create workflow without passing most parameters to avoid initialization error
        workflow = GeneralAnalysisWorkflow()
        
        # Set attributes directly after initialization
        workflow.name = "General Analysis"
        workflow.description = "An agent that can analyze data and provide insights"
        workflow.session_id = f"polls-analysis-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        workflow.debug_mode = True
        workflow.submodule_id = SUBMODULE_ID
        
        # Run workflow and collect response
        # st.info("Executing workflow...")
        response = None
        for r in workflow.run():
            response = r
        
        # Get the path to the latest report file
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        reports_dir = os.path.join(project_root, "reports")
        
        # st.info(f"Looking for reports in: {reports_dir}")
        
        if not os.path.exists(reports_dir):
            st.error("Reports directory not found")
            return None
            
        # Find the latest PDF file in the reports directory
        pdf_files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]
        if not pdf_files:
            st.error("No PDF reports found")
            return None
            
        # Sort by creation time (newest first)
        pdf_files.sort(key=lambda x: os.path.getctime(os.path.join(reports_dir, x)), reverse=True)
        latest_pdf = os.path.join(reports_dir, pdf_files[0])
        
        # st.info(f"Found latest report: {latest_pdf}")
        
        # Read and return the PDF content
        with open(latest_pdf, 'rb') as file:
            return file.read()
            
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        st.error(traceback.format_exc())  # Add detailed traceback
        return None

def main():

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = {}
    
    # Initialize operation status for button disabling
    if 'operation_in_progress' not in st.session_state:
        st.session_state.operation_in_progress = False

    # Simple sidebar
    with st.sidebar:
        st.title("Status")
        # Initialize status in session state if not present
        if 'status' not in st.session_state:
            st.session_state.status = 'ready'
        
        # Status indicator
        status_color = {
            'ready': '🟢',
            'loading': '🟡',
            'error': '🔴'
        }
        
        st.write(f"{status_color[st.session_state.status]} Status: {st.session_state.status.capitalize()}")
        
        # Show last data fetch time
        if 'last_fetch_time' in st.session_state:
            st.text(f"Last fetch: {st.session_state.last_fetch_time}")
        
        if st.session_state.status == 'error':
            st.error("An error occurred while loading data. Please try again.")
        elif st.session_state.status == 'loading':
            st.info("Loading data...")
        
        # Add a divider 
        st.markdown("---")
        
        # Data management section
        st.subheader("Data Management")
        
        # Option to clear cache only - disable when operation is in progress
        refresh_button = st.button(
            "🔄 Refresh Data Cache", 
            key="clear_cache",
            disabled=st.session_state.operation_in_progress
        )
        
        if refresh_button and not st.session_state.operation_in_progress:
            # Set operation in progress flag
            st.session_state.operation_in_progress = True
            st.info("Clearing cache... Please wait.")
            success = clear_cache_only()
            if success:
                # Reset flag and force refresh
                st.session_state.operation_in_progress = False
                st.rerun()

        # Show operation status
        if st.session_state.operation_in_progress:
            st.warning("⏳ Operation in progress... Buttons are disabled.")

        # Add a divider 
        st.markdown("---")

        # Initialize pdf_data in session state if it doesn't exist
        if 'pdf_data' not in st.session_state:
            st.session_state.pdf_data = None

        # Initialize report generation state if it doesn't exist
        if 'report_generating' not in st.session_state:
            st.session_state.report_generating = False

        # Create a placeholder for status messages
        status_placeholder = st.empty()

        # Show report generation status if it's in progress
        if st.session_state.report_generating:
            status_placeholder.info("⏳ Generating report... Please wait. This might take a minute.")

        # Use a regular button instead of download_button for initiating the process
        # Disable the button when report is generating
        report_button = st.button(
            "Generate Report", 
            disabled=st.session_state.report_generating,
            key="generate_report_button"
        )

        if report_button:
            # Set the report generating flag to disable the button on next rerun
            st.session_state.report_generating = True
            st.rerun()  # Force a rerun to update the button state

        # Handle actual report generation when flag is set but pdf_data is not
        if st.session_state.report_generating and st.session_state.pdf_data is None:
            try:
                # Show status message
                status_placeholder.info("Generating report... Please wait, this may take a minute.")
                
                # Generate the report
                pdf_data = generate_and_get_report()
                
                # Store the result
                st.session_state.pdf_data = pdf_data
                
                # Update status message
                if pdf_data:
                    status_placeholder.success("Report generated successfully!")
                else:
                    status_placeholder.error("Failed to generate report.")
                
            except Exception as e:
                # Show error
                status_placeholder.error(f"Error generating report: {str(e)}")
            
            # Reset the flag
            st.session_state.report_generating = False
            st.rerun()  # Rerun to update UI

        # Show download button only if we have PDF data
        if st.session_state.pdf_data:
            st.download_button(
                label="Download Report",
                data=st.session_state.pdf_data,
                file_name="questionnaire_report.pdf",
                mime="application/pdf"
            )

    # Auto-load data on page load
    with st.spinner("Analysing..."):
        try:
            # Set operation in progress flag
            st.session_state.operation_in_progress = True
            st.session_state.status = 'loading'
            
            # Fetch data from Database - now unpacks both values
            results, count_result = fetch_data()

            # Store last fetch time
            if 'last_fetch_time' not in st.session_state:
                st.session_state.last_fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Update session state with results
            st.session_state.data = results

            # Set status to ready
            st.session_state.status = 'ready'
            
            # Reset operation in progress flag
            st.session_state.operation_in_progress = False
            
            # Display data - pass both results and count_result
            display_vizualizations(results, count_result)

        except Exception as e:
            st.session_state.status = 'error'
            st.error(f"An error occurred: {e}")
            # Reset operation in progress flag even on error
            st.session_state.operation_in_progress = False

# Call the main function
if __name__ == "__main__":
    main()