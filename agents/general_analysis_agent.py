from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
import os
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import pandas as pd
from io import StringIO
import dotenv
from typing import Iterator, Optional, Dict, List, Any

# .env variables
dotenv.load_dotenv()

# AI endpoint
AI_ENDPOINT = os.getenv("AI_ENDPOINT")

class AnalysisResult(BaseModel):
    """Model for analysis results of questionnaire data"""
    category: str  # e.g. 'age', 'gender', 'location'
    question_id: Optional[str] = None  # None for general analysis
    segments: Dict[str, int]  # e.g. {'18-24': 45, '25-34': 32, ...}
    insights: str

class GeneralAnalysisWorkflow(Workflow):

    data_analysis_agent: Agent = Agent(
        model=Ollama(id="llama3.1:8b", host=AI_ENDPOINT),
        instructions=[
            "You are an agent responsible for analyzing questionnaire data by demographic categories in Kenya.",
            "You will be provided with questionnaire responses organized by user demographics.",
            "Analyze trends and patterns in the data based on age groups, gender, and locations.",
            "For each question, identify significant differences in responses across demographic segments.",
            "Provide clear, concise insights about what the data reveals about different demographic groups.",
            "Use statistical reasoning where appropriate to support your analysis.",
            "Focus on meaningful patterns rather than random variations in the data.",
            "",
            "IMPORTANT: The demographic data is available in every record as gender, location and date_of_birth.",
            "",
            "For age analysis, look for date_of_birth field with segments typically including:",
            "- under_18",
            "- 18-35",
            "- 36-65",
            "- 66+",
            "",
            "For gender analysis, look for gender field with segments typically including:",
            "- male",
            "- female",
            "- other",
            "",
            "For location analysis, look for the location field with segments typically including:",
            "- Nairobi",
            "- Mombasa",
            "- Kisumu",
            "- Nakuru",
            "- Eldoret",
            "- Other",
        ],
        add_history_to_messages=True,
        markdown=True,
        debug_mode=False,
    )
    
    # Agent for generating insights and recommendations
    insights_agent: Agent = Agent(
        model=Ollama(id="llama3.1:8b", host=AI_ENDPOINT),
        instructions=[
            "You are an agent responsible for generating insights from analyzed questionnaire data in Kenya.",
            "You will be provided with analysis results broken down by demographic categories.",
            "You MUST follow this exact format for your report:",
            "",
            "# Analysis",
            "## Key Insights from Questionnaire Analysis",
            "## Demographic Analysis",
            "### Age Distribution Analysis",
            "- [List bullet points about age distribution insights]",
            "### Gender Distribution Analysis",
            "- [List bullet points about gender distribution insights]",
            "### Location Distribution Analysis",
            "- [List bullet points about location distribution insights]",
            "## Recommendations",
            "1. [First recommendation]",
            "2. [Second recommendation]",
            "## Sentiment Analysis",
            "- [List bullet points about sentiment analysis insights]",
            "- [Give a summary on whether the sentiment is positive, negative or neutral]",
            "## Limitations",
            "[Paragraph about limitations]",
            "",
            "IMPORTANT: The demographic data is available in every record as gender, location and date_of_birth.",
            "",
            "When analyzing patterns in the data, consider:",
            "1. Age differences from date_of_birth",
            "2. Gender differences from gender field",
            "3. Geographic differences from location field",
            "4. How these demographics might influence people's responses and opinions"
        ],
        add_history_to_messages=True,
        markdown=True,
        debug_mode=False,
    )

    report_generation: Agent = Agent(
        model=Ollama(id="llama3.1:8b", host=AI_ENDPOINT),
        name="Report Generation",
        description="An agent that can generate a report based on the analysis",
        add_history_to_messages=True,
        markdown=True,
        debug_mode=False,
    )

    def __init__(self, **kwargs):
        """Initialize the workflow with optional parameters"""
        # Call the parent's __init__ without arguments
        super().__init__()
        
        # Set attributes from kwargs with defaults
        self.name = kwargs.get('name', "General Analysis")
        self.description = kwargs.get('description', "An agent that can analyze data and provide insights")
        self.session_id = kwargs.get('session_id', f"polls-analysis-{datetime.now().strftime('%Y%m%d%H%M%S')}")
        self.debug_mode = kwargs.get('debug_mode', False)
        self.submodule_id = kwargs.get('submodule_id', None)

    def fetch_data(self):
        # Create database URL from secrets
        db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)

        # Get all the data from the database
        with engine.connect() as conn:
            # Add filter for submodule_id if provided
            if self.submodule_id:
                # Modify your query to filter by submodule_id
                query = text("SELECT * FROM questionnaire_responses WHERE sub_module_id = :submodule_id")
                results = conn.execute(query, {"submodule_id": self.submodule_id}).fetchall()
            else:
                # Original query without filter
                query = text("SELECT * FROM questionnaire_responses")
                results = conn.execute(query).fetchall()

        return results
    
    def format_data(self, data):
        # Create a list of dictionaries from row objects
        formatted_data = []
        for row in data:
            # Access row elements as a tuple or by column name depending on SQLAlchemy version
            try:
                # Try to convert row to a dictionary using ._asdict() for RowProxy objects
                row_dict = dict(row._asdict())
            except (AttributeError, TypeError):
                # If that fails, try to access row as a tuple with column keys
                row_dict = {}
                for i, column in enumerate(row.keys() if hasattr(row, 'keys') else []):
                    row_dict[column] = row[i]
            
            formatted_data.append(row_dict)
        
        # Convert to JSON string
        data_str = json.dumps(formatted_data, default=str)

        return data_str
    
    def categorize_by_age(self, responses_data: str) -> Dict[str, List[Dict]]:
        """Categorize responses by age groups using direct date_of_birth field"""
        # Convert JSON string to list of dictionaries if needed
        if isinstance(responses_data, str):
            responses = json.loads(responses_data)
        else:
            responses = responses_data
        
        age_groups = {
            "under_18": [],
            "18-35": [],
            "36-65": [],
            "66+": [],
            "unknown": []
        }
        
        for response in responses:
            # Check if there's a date_of_birth directly in the response
            if response.get('date_of_birth'):
                # Calculate age from date_of_birth
                try:
                    dob_str = response.get('date_of_birth')
                    # Handle different date formats
                    if 'T' in dob_str and ('Z' in dob_str or '+' in dob_str):
                        # This is an ISO format with timezone info
                        from datetime import timezone
                        
                        # Replace Z with +00:00 for proper ISO parsing
                        if 'Z' in dob_str:
                            dob_str = dob_str.replace('Z', '+00:00')
                        
                        # Parse as timezone-aware datetime
                        dob = datetime.fromisoformat(dob_str)
                        
                        # Make sure we compare with timezone-aware datetime.now()
                        now = datetime.now(timezone.utc)
                    elif 'T' in dob_str:
                        # ISO format without timezone - treat as UTC
                        dob = datetime.fromisoformat(dob_str)
                        # Make current time naive for consistency
                        now = datetime.now()
                    else:
                        # Try different formats for non-ISO dates
                        formats = ['%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d']
                        for fmt in formats:
                            try:
                                dob = datetime.strptime(dob_str, fmt)
                                break
                            except ValueError:
                                continue
                        else:
                            # If no format works, put in unknown
                            age_groups["unknown"].append(response)
                            continue
                        
                        # Use naive datetime for non-ISO formats
                        now = datetime.now()
                    
                    # Calculate age
                    age = (now - dob).days // 365
                    
                    # Categorize by age
                    if age < 18:
                        age_groups["under_18"].append(response)
                    elif age < 36:
                        age_groups["18-35"].append(response)
                    elif age < 66:
                        age_groups["36-65"].append(response)
                    else:
                        age_groups["66+"].append(response)
                except Exception as e:
                    print(f"Error parsing date of birth '{dob_str}': {str(e)}")
                    age_groups["unknown"].append(response)
            else:
                # If no DOB in response, put in unknown
                age_groups["unknown"].append(response)
        
        return age_groups
    
    def categorize_by_gender(self, responses_data: str) -> Dict[str, List[Dict]]:
        """Categorize responses by gender using direct gender field"""
        # Convert JSON string to list of dictionaries if needed
        if isinstance(responses_data, str):
            responses = json.loads(responses_data)
        else:
            responses = responses_data
        
        gender_groups = {
            "male": [],
            "female": [],
            "other": [],
            "unknown": []
        }
        
        for response in responses:
            # Get gender directly from the response
            if response.get('gender'):
                gender = response.get('gender', '').lower()
                if gender == 'male':
                    gender_groups["male"].append(response)
                elif gender in ['female', 'femal']: # handle possible typo
                    gender_groups["female"].append(response)
                else:
                    gender_groups["other"].append(response)
            else:
                # If no gender info, add to unknown
                gender_groups["unknown"].append(response)
        
        return gender_groups
    
    def categorize_by_location(self, responses_data: str) -> Dict[str, List[Dict]]:
        """Categorize responses by location using direct location field"""
        # Convert JSON string to list of dictionaries if needed
        if isinstance(responses_data, str):
            responses = json.loads(responses_data)
        else:
            responses = responses_data
        
        location_groups = {}
        
        for response in responses:
            # Get location directly from the response
            location = response.get('location')
            
            # If no location found, use 'Unknown'
            if not location:
                location = "Unknown"
            
            # Add to appropriate group
            if location not in location_groups:
                location_groups[location] = []
            
            location_groups[location].append(response)
        
        return location_groups
    
    def generate_final_insights(self, data) -> str:
        """Generate final insights based on all analysis results"""
        
        agent_input = {
            "analysis_results": [result.dict() for result in self.analysis_results],
            "total_responses": len(data),
            "required_format": """
                # Analysis
                ## Key Insights from Questionnaire Analysis
                ## Demographic Analysis
                ### Age Distribution Analysis
                - [List bullet points about age distribution insights]
                ### Gender Distribution Analysis
                - [List bullet points about gender distribution insights]
                ### Location Distribution Analysis
                - [List bullet points about location distribution insights]
                ## Recommendations
                1. [First recommendation]
                2. [Second recommendation]
                ## Limitations
                [Paragraph about limitations]
            """
        }
        
        response = self.insights_agent.run(
            json.dumps(agent_input, indent=2)
        )
        
        return response.content
    
    def generate_report_pdf(self, final_response, data=None):
        # Import necessary libraries
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib import colors
        import matplotlib.pyplot as plt
        import numpy as np
        import tempfile
        import pandas as pd
        from io import StringIO
        
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Create reports directory at the project root
        reports_dir = os.path.join(project_root, "reports")
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        # Create a PDF file path
        pdf_file = os.path.join(reports_dir, f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
        
        # Create the PDF document
        doc = SimpleDocTemplate(pdf_file, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create a list of flowables for the PDF
        flowables = []
        
        # Keep track of temporary files to delete after PDF is built
        temp_files = []
        
        # Extract questionnaire name from the first record if available
        questionnaire_title = "Questionnaire Analysis Report"  # Default title
        
        if data:
            try:
                # Convert data to DataFrame if it's not already
                if isinstance(data, str):
                    data_parsed = json.loads(data)
                else:
                    data_parsed = data
                    
                df = pd.DataFrame(data_parsed)
                
                # Check if questionnaire_name exists in the data
                if 'questionnaire_name' in df.columns:
                    # Get the first non-null value
                    questionnaire_names = df['questionnaire_name'].dropna().unique()
                    if len(questionnaire_names) > 0:
                        questionnaire_title = f"{questionnaire_names[0]} Analysis Report"
                        print(f"Using questionnaire name from data: {questionnaire_names[0]}")
                elif 'formData' in df.columns and isinstance(df['formData'].iloc[0], dict):
                    # Try to get questionnaire name from formData if it exists
                    form_data = df['formData'].iloc[0]
                    if 'questionnaire_name' in form_data:
                        questionnaire_title = f"{form_data['questionnaire_name']} Analysis Report"
                        print(f"Using questionnaire name from formData: {form_data['questionnaire_name']}")
                
            except Exception as e:
                print(f"Error extracting questionnaire name: {e}")
        
        # Add the title with the questionnaire name
        title = Paragraph(questionnaire_title, styles['Title'])
        flowables.append(title)
        flowables.append(Spacer(1, 12))
        
        # Add date
        date_text = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal'])
        flowables.append(date_text)
        flowables.append(Spacer(1, 24))
        
        # Process data if available
        if data:
            try:
                # Convert JSON string to dictionary/list if it's not already
                if isinstance(data, str):
                    data = json.loads(data)
                
                # Convert to pandas DataFrame for easier manipulation
                df = pd.DataFrame(data)
                
                # Create a unique respondents dataframe to prevent duplication
                if 'id_no' in df.columns:
                    # Use the first occurrence of each respondent's ID
                    unique_respondents_df = df.drop_duplicates(subset=['id_no'])
                    print(f"Created unique respondents dataframe with {len(unique_respondents_df)} entries from {len(df)} total records")
                else:
                    unique_respondents_df = df
                    print("No id_no column found, using all records as unique")
                
                # Use location field directly instead of district_of_birth
                if 'location' in unique_respondents_df.columns:
                    print(f"Using location field directly for visualizations. Sample values: {unique_respondents_df['location'].head(3).tolist()}")
                
                # For debugging: Print column names in the DataFrame
                print(f"Available columns in the data: {unique_respondents_df.columns.tolist()}")
                
                # Calculate age from date_of_birth if available - working with unique respondents only
                if 'date_of_birth' in unique_respondents_df.columns:
                    try:
                        # Print sample values before processing
                        print(f"Date of birth samples (raw): {unique_respondents_df['date_of_birth'].head(5).tolist()}")
                        
                        # Create a direct calculation instead of relying on automatic detection
                        # Set fixed ages based on common birth years
                        # First create a simple mapping from birth year to age
                        current_year = datetime.now().year
                        
                        # Function to extract year from ISO date string like '1989-07-15T21:00:00.000Z'
                        def extract_year_from_iso(date_str):
                            try:
                                if isinstance(date_str, str) and len(date_str) >= 4:
                                    # Extract the first 4 characters which should be the year
                                    year_str = date_str[:4]
                                    if year_str.isdigit():
                                        year = int(year_str)
                                        # Basic sanity check for reasonable years
                                        if 1923 <= year <= current_year:
                                            return year
                            except Exception as e:
                                print(f"Error extracting year from {date_str}: {e}")
                            return None
                        
                        # Apply the function to extract years
                        unique_respondents_df['birth_year'] = unique_respondents_df['date_of_birth'].apply(extract_year_from_iso)
                        
                        # Calculate age directly from birth year
                        def calculate_age_from_year(year):
                            if year is None:
                                return None
                            age = current_year - year
                            # Basic sanity check
                            return age if 0 <= age <= 120 else None
                        
                        # Apply the function to calculate ages
                        unique_respondents_df['age'] = unique_respondents_df['birth_year'].apply(calculate_age_from_year)
                        
                        # Print detailed debugging information
                        print("\nDETAILED AGE CALCULATION:")
                        print("---------------------------")
                        print(f"Current year: {current_year}")
                        for i in range(min(10, len(unique_respondents_df))):
                            date_str = unique_respondents_df['date_of_birth'].iloc[i]
                            year = unique_respondents_df['birth_year'].iloc[i]
                            age = unique_respondents_df['age'].iloc[i]
                            print(f"Row {i}: Date '{date_str}' -> Year {year} -> Age {age}")
                        
                        # Convert to numeric and handle any NaN values
                        unique_respondents_df['age'] = pd.to_numeric(unique_respondents_df['age'], errors='coerce')
                        
                        # Debug age distribution
                        valid_ages = unique_respondents_df['age'].dropna()
                        print(f"\nCalculated {len(valid_ages)} valid ages")
                        if len(valid_ages) > 0:
                            print(f"Age distribution: {valid_ages.value_counts().sort_index().to_dict()}")
                            print(f"Age statistics: min={valid_ages.min()}, max={valid_ages.max()}, mean={valid_ages.mean():.1f}")
                            
                    except Exception as e:
                        print(f"Error in age calculation: {e}")
                        import traceback
                        traceback.print_exc()
                        
                    # If ages are still in a narrow range (33-41), it could be that the ages are correct
                    # but we can manually adjust them for demonstration purposes
                    valid_ages = unique_respondents_df['age'].dropna()
                    age_range = valid_ages.max() - valid_ages.min()
                    
                    if len(valid_ages) > 0 and age_range < 10:
                        print("\nDetected narrow age range. Adjusting for better visualization...")
                        
                        # Option 1: Keep the actual ages for reporting accuracy
                        # but create a wider distribution for visuals
                        original_ages = unique_respondents_df['age'].copy()
                        
                        # Store original age statistics for the summary table
                        if len(valid_ages) > 0:
                            age_stats = {
                                'original_mean': valid_ages.mean(),
                                'original_min': valid_ages.min(),
                                'original_max': valid_ages.max(),
                            }
                            print(f"Original age statistics: {age_stats}")
                        
                        # Create a wider distribution for visualization purposes
                        np.random.seed(42)  # For reproducibility
                        # Expand the age range while maintaining the same mean
                        mean_age = valid_ages.mean()
                        min_age = max(18, int(mean_age - 20))  # At least 18
                        max_age = int(mean_age + 20)  # Extend upward
                        
                        # Generate a wider distribution
                        expanded_ages = np.random.normal(mean_age, 10, size=len(unique_respondents_df))
                        expanded_ages = np.clip(expanded_ages, min_age, max_age).round().astype(int)
                        
                        # For demonstration only: replace ages with expanded distribution
                        # Uncomment the line below to use expanded ages for visualization
                        # unique_respondents_df['age'] = expanded_ages
                        
                        print(f"Note: Using actual ages from data. Age range is narrow ({valid_ages.min()}-{valid_ages.max()}).")
                
                # Add data summary section
                summary_heading = Paragraph("Data Summary", styles['Heading2'])
                flowables.append(summary_heading)
                flowables.append(Spacer(1, 12))
                
                # Create a summary table with basic statistics
                summary_data = []
                
                # Show total respondents
                summary_data.append(["Total Respondents", str(len(unique_respondents_df))])
                
                # Add gender counts (specifically male and female)
                if 'gender' in unique_respondents_df.columns:
                    # Count genders among unique respondents
                    gender_counts = unique_respondents_df['gender'].str.lower().value_counts()
                    male_count = gender_counts.get('male', 0)
                    female_count = gender_counts.get('female', 0)
                    
                    # Calculate gender percentages based on unique respondents
                    total_with_gender = unique_respondents_df['gender'].count()
                    male_percent = (male_count / total_with_gender * 100) if total_with_gender > 0 else 0
                    female_percent = (female_count / total_with_gender * 100) if total_with_gender > 0 else 0
                    
                    summary_data.append(["Male Respondents", f"{male_count} ({male_percent:.1f}%)"])
                    summary_data.append(["Female Respondents", f"{female_count} ({female_percent:.1f}%)"])
                    
                    # Add other genders if present (beyond male/female binary)
                    other_genders = set(gender_counts.index) - {'male', 'female'}
                    if other_genders:
                        other_count = sum(gender_counts.get(g, 0) for g in other_genders)
                        other_percent = (other_count / total_with_gender * 100) if total_with_gender > 0 else 0
                        summary_data.append(["Other Gender(s)", f"{other_count} ({other_percent:.1f}%)"])
                
                # Add age statistics if available - use unique_respondents_df instead of df
                if 'age' in unique_respondents_df.columns and pd.api.types.is_numeric_dtype(unique_respondents_df['age']):
                    valid_ages = unique_respondents_df['age'].dropna()
                    if len(valid_ages) > 0:
                        summary_data.append(["Average Age", f"{valid_ages.mean():.1f} years"])
                        summary_data.append(["Age Range", f"{valid_ages.min()} - {valid_ages.max()} years"])
                        print(f"Added age statistics from {len(valid_ages)} valid ages")
                
                # Add location information from unique respondents
                if 'location' in unique_respondents_df.columns:
                    # Count unique locations
                    # unique_locations = unique_respondents_df['location'].nunique()
                    # summary_data.append(["Unique Locations", str(unique_locations)])
                    
                    # Top 3 locations by number of respondents
                    top_locations = unique_respondents_df['location'].value_counts().head(3)
                    location_info = ", ".join([f"{loc} ({count})" for loc, count in top_locations.items()])
                    summary_data.append(["Top Locations", location_info])
                
                # Create and add the summary table
                summary_table = Table(summary_data, colWidths=[200, 250])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
                ]))
                
                flowables.append(summary_table)
                
                # Add a page break after the summary section
                flowables.append(PageBreak())
                
                # Generate demographic visualizations
                visualizations_heading = Paragraph("Demographic Visualizations", styles['Heading2'])
                flowables.append(visualizations_heading)
                flowables.append(Spacer(1, 12))
                
                # Create visualizations for gender, age, and location if they exist
                demo_cols = ['gender', 'age', 'location']
                
                # Debug message about available demographic columns
                print(f"Demographics to visualize: {demo_cols}")
                print(f"Available columns in unique_respondents_df: {unique_respondents_df.columns.tolist()}")
                for col in demo_cols:
                    if col in unique_respondents_df.columns:
                        print(f"Column {col} exists with dtype: {unique_respondents_df[col].dtype}")
                        print(f"Sample values: {unique_respondents_df[col].head(3).tolist()}")
                        print(f"Non-null count: {unique_respondents_df[col].notna().sum()} / {len(unique_respondents_df)}")
                    else:
                        print(f"Column {col} does not exist in the dataframe")
                
                # 1. Categorical demographics (likely gender and location)
                for col in demo_cols:
                    # Check in unique_respondents_df instead of df
                    if col in unique_respondents_df.columns and unique_respondents_df[col].dtype.name == 'object':
                        if unique_respondents_df[col].nunique() < 20:  # Reasonable number of categories
                            plt.figure(figsize=(8, 6))
                            
                            # Always use unique_respondents_df for consistent visualization of unique respondents
                            value_counts = unique_respondents_df[col].value_counts()
                            count_title = "Unique Respondents"
                            
                            # Print debug information
                            print(f"Creating visualization for {col} with {len(value_counts)} unique values")
                            print(f"Top values: {value_counts.head(3).to_dict()}")
                            
                            # Limit to top 6 categories for better pie chart readability
                            # For categories beyond top 6, group them as "Others"
                            if len(value_counts) > 6:
                                top_values = value_counts.iloc[:6]
                                others_sum = value_counts.iloc[6:].sum()
                                if others_sum > 0:
                                    top_values = pd.concat([top_values, pd.Series([others_sum], index=['Others'])])
                                pie_data = top_values
                            else:
                                pie_data = value_counts
                            
                            # Calculate percentages for labels
                            total = pie_data.sum()
                            labels = [f'{idx}\n{val} ({val/total:.1%})' for idx, val in zip(pie_data.index, pie_data.values)]
                            
                            # Create pie chart
                            plt.pie(pie_data, 
                                   labels=None,  # We'll use a legend instead of direct labels
                                   autopct='%1.1f%%', 
                                   startangle=90, 
                                   shadow=False, 
                                   explode=[0.05] * len(pie_data),  # Slight separation for all slices
                                   colors=plt.cm.tab10.colors,
                                   wedgeprops={'edgecolor': 'white', 'linewidth': 1})
                            
                            # Add legend with both category names and counts
                            plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))
                            
                            plt.title(f'Distribution of {col.title()} (Unique Respondents)')
                            plt.axis('equal')  # Equal aspect ratio ensures pie is circular
                            
                            # Save the plot to a temporary file - use a non-deleting file
                            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_file_name = temp_file.name
                            temp_files.append(temp_file_name)
                            plt.savefig(temp_file_name, format='png', dpi=150, bbox_inches='tight')
                            plt.close()
                            temp_file.close()  # Close but don't delete
                            
                            # Add the image to the PDF
                            flowables.append(Paragraph(f"Distribution of {col.title()}", styles['Heading3']))
                            img = Image(temp_file_name, width=450, height=320)
                            flowables.append(img)
                            
                            # Add a page break after each visualization
                            flowables.append(PageBreak())
                
                # 2. Age distribution (assuming numeric)
                if 'age' in unique_respondents_df.columns and pd.api.types.is_numeric_dtype(unique_respondents_df['age']) and unique_respondents_df['age'].notna().sum() > 0:
                    # Only create visualizations if we have actual age data
                    age_data = unique_respondents_df['age'].dropna()
                    print(f"Creating age visualizations using {len(age_data)} valid age values from unique respondents")
                    
                    # Create age group visualization
                    plt.figure(figsize=(7, 5))
                    
                    # Create age groups
                    bins = [18, 35, 65, 100]
                    labels = ['Youth', 'Middle Aged', 'Seniors']
                    unique_respondents_df['age_group'] = pd.cut(unique_respondents_df['age'], bins=bins, labels=labels, right=False)
                    
                    # Plot age groups
                    age_counts = unique_respondents_df['age_group'].value_counts().sort_index()
                    bars = plt.bar(age_counts.index, age_counts.values, color='lightgreen')
                    plt.xlabel('Age Group')
                    plt.ylabel('Count of Unique Respondents')
                    plt.title('Distribution by Age Group')
                    plt.xticks(rotation=0)  # No need for rotation with only 3 categories
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{height}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    
                    # Save the plot to a temporary file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_file_name = temp_file.name
                    temp_files.append(temp_file_name)
                    plt.savefig(temp_file_name, format='png', dpi=150)
                    plt.close()
                    temp_file.close()  # Close but don't delete
                    
                    # Add the image to the PDF
                    flowables.append(Paragraph("Distribution by Age Group", styles['Heading3']))
                    img = Image(temp_file_name, width=450, height=320)
                    flowables.append(img)
                    
                    # Add a page break after the age visualization
                    flowables.append(PageBreak())
                
                # 3. Questions Analysis Section - new addition for question_id based structure
                # First group data by question_id to analyze questions
                if 'question_id' in df.columns and 'answer' in df.columns:
                    # Group data by question_id
                    questions_data = df.groupby('question_id')
                    
                    # Add a section heading for Questions Analysis
                    questions_heading = Paragraph("Questions Analysis", styles['Heading2'])
                    flowables.append(questions_heading)
                    flowables.append(Spacer(1, 12))
                    
                    # Get list of unique question IDs
                    question_ids = df['question_id'].unique()
                    
                    # Process each question
                    for q_id in question_ids[:15]:  # Limit to first 15 questions
                        # Get data for this question
                        q_data = df[df['question_id'] == q_id]
                        
                        # Skip if too few responses
                        if len(q_data) < 5:
                            continue
                            
                        # Get answers for this question
                        answers = q_data['answer']
                        
                        # Handle list answers - if answers are in string representation of lists
                        # First check if answers appear to be lists
                        list_answers = False
                        try:
                            if answers.iloc[0] and (answers.iloc[0].startswith('[') or answers.iloc[0].startswith('{')):
                                list_answers = True
                                # Flatten list answers
                                flattened_answers = []
                                for ans in answers:
                                    if ans and isinstance(ans, str):
                                        try:
                                            # Try to parse as JSON list
                                            ans_list = json.loads(ans.replace("'", "\""))
                                            if isinstance(ans_list, list):
                                                flattened_answers.extend(ans_list)
                                            elif isinstance(ans_list, dict):
                                                flattened_answers.extend(ans_list.values())
                                            else:
                                                flattened_answers.append(str(ans))
                                        except:
                                            # If not valid JSON, add as is
                                            flattened_answers.append(str(ans))
                                    elif ans:
                                        flattened_answers.append(str(ans))
                                # Convert to Series for value_counts
                                answers = pd.Series(flattened_answers)
                        except (IndexError, AttributeError):
                            # If there's an error, just use the answers as they are
                            pass
                        
                        # Get response counts
                        response_counts = answers.value_counts().nlargest(5)  # Top 5 answers
                        
                        # Skip if no valid responses
                        if len(response_counts) == 0:
                            continue
                        
                        # Get question title from the data if available
                        # Try to extract the actual question text from the question_id
                        # Many question IDs follow formats like "TOPIC: ACTUAL QUESTION" or similar
                        q_title = q_id
                        
                        # Extract the actual question part if it follows common formats
                        if ' - ' in q_id:
                            # Format: "TOPIC - QUESTION"
                            q_title = q_id.split(' - ', 1)[1]
                        elif ':' in q_id:
                            # Format: "TOPIC: QUESTION"
                            q_title = q_id.split(':', 1)[1].strip()
                        
                        # Add question heading - just the question without any prefix
                        q_heading = Paragraph(f"Question: {q_title}", styles['Heading3'])
                        flowables.append(q_heading)
                        
                        # Create a mapping for answers to keep chart labels clean
                        answer_map = {answer: f"Option {chr(65+i)}" for i, answer in enumerate(response_counts.index)}
                        
                        # 1. Create overall response chart - one chart per page
                        plt.figure(figsize=(8, 6))
                        
                        # Map answers to labels for cleaner display
                        chart_labels = [answer_map[answer] for answer in response_counts.index]
                        
                        # Create horizontal bar chart
                        bars = plt.barh(chart_labels, response_counts.values, color='skyblue')
                        plt.xlabel('Number of Responses')
                        plt.ylabel('Response Option')
                        plt.title(f'Top Responses')
                        
                        # Add value labels
                        for bar in bars:
                            width = bar.get_width()
                            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                                    f'{width}', ha='left', va='center')
                        
                        plt.tight_layout()
                        
                        # Save chart to temp file
                        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        temp_file_name = temp_file.name
                        temp_files.append(temp_file_name)
                        plt.savefig(temp_file_name, format='png', dpi=150)
                        plt.close()
                        temp_file.close()
                        
                        # Add chart to PDF with its own heading and table on the same page
                        flowables.append(Paragraph("Overall Responses", styles['Heading4']))
                        img = Image(temp_file_name, width=450, height=300)
                        flowables.append(img)
                        flowables.append(Spacer(1, 12))
                        
                        # Create and add the legend table with clear formatting
                        legend_data = [["Label", "Answer"]]
                        for answer, label in answer_map.items():
                            legend_data.append([label, str(answer)])
                        
                        # Fixed width columns for consistent appearance
                        legend_table = Table(legend_data, colWidths=[80, 400])
                        legend_table.setStyle(TableStyle([
                            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                            ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left align label column
                            ('ALIGN', (1, 0), (1, -1), 'LEFT'),  # Left align answer column
                            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                            ('WORDWRAP', (1, 1), (1, -1), True)  # Enable word wrapping for answers
                        ]))
                        flowables.append(legend_table)
                        
                        # Add page break after this chart and its legend, only if we have more sections coming
                        has_gender_section = 'gender' in df.columns and df['gender'].notna().sum() > 10
                        has_age_section = 'age_group' in df.columns and df['age_group'].notna().sum() > 10
                        has_location_section = 'location' in df.columns and df['location'].notna().sum() > 10
                        
                        if has_gender_section or has_age_section or has_location_section:
                            flowables.append(PageBreak())
                        
                        # 2. By Gender (if gender data exists)
                        if 'gender' in df.columns and df['gender'].notna().sum() > 10:
                            try:
                                # Get top gender categories
                                top_genders = df['gender'].value_counts().nlargest(3).index
                                
                                # Get data for this question filtered by top genders
                                gender_q_data = q_data[q_data['gender'].isin(top_genders)]
                                
                                # Prepare data for cross-tabulation
                                # Create a temporary DataFrame with gender and answer
                                gender_chart_data = []
                                
                                for _, row in gender_q_data.iterrows():
                                    gender = row['gender']
                                    answer = row['answer']
                                    
                                    # Handle list answers if needed
                                    if list_answers and answer and isinstance(answer, str) and (answer.startswith('[') or answer.startswith('{')):
                                        try:
                                            ans_list = json.loads(answer.replace("'", "\""))
                                            if isinstance(ans_list, list):
                                                for a in ans_list:
                                                    if a in response_counts.index:  # Only include top 5 answers
                                                        gender_chart_data.append({'gender': gender, 'answer': a})
                                            elif isinstance(ans_list, dict):
                                                for a in ans_list.values():
                                                    if a in response_counts.index:
                                                        gender_chart_data.append({'gender': gender, 'answer': a})
                                            else:
                                                if answer in response_counts.index:
                                                    gender_chart_data.append({'gender': gender, 'answer': answer})
                                        except:
                                            if answer in response_counts.index:
                                                gender_chart_data.append({'gender': gender, 'answer': answer})
                                    elif answer in response_counts.index:
                                        gender_chart_data.append({'gender': gender, 'answer': answer})
                                
                                # Convert to DataFrame
                                gender_df = pd.DataFrame(gender_chart_data)
                                
                                # Skip if no data after filtering
                                if len(gender_df) == 0:
                                    raise ValueError("No gender data for this question after filtering")
                                
                                # Create cross-tabulation
                                gender_cross = pd.crosstab(gender_df['gender'], gender_df['answer'])
                                
                                # Create grouped bar chart
                                plt.figure(figsize=(10, 6))
                                
                                # Prepare data with mapped labels
                                x = np.arange(len(top_genders))
                                width = 0.15
                                
                                # Plot bars for each answer (up to 5)
                                for i, answer in enumerate(response_counts.index):
                                    if answer in gender_cross.columns:
                                        gender_values = []
                                        for gender in top_genders:
                                            gender_values.append(gender_cross.at[gender, answer] if gender in gender_cross.index and answer in gender_cross.columns else 0)
                                        
                                        offset = width * (i - len(response_counts.index)/2 + 0.5)
                                        plt.bar(x + offset, gender_values, width, 
                                               label=answer_map[answer], 
                                               color=plt.cm.tab10.colors[i % 10])
                                
                                plt.xlabel('Gender')
                                plt.ylabel('Count')
                                plt.title('Responses by Gender')
                                plt.xticks(x, top_genders)
                                plt.legend(title="Response Options")
                                
                                plt.tight_layout()
                                
                                # Save chart to temp file
                                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                                temp_file_name = temp_file.name
                                temp_files.append(temp_file_name)
                                plt.savefig(temp_file_name, format='png', dpi=150)
                                plt.close()
                                temp_file.close()
                                
                                # Add chart to PDF with its own heading and legend on the same page
                                flowables.append(Paragraph("Responses by Gender", styles['Heading4']))
                                img = Image(temp_file_name, width=450, height=300)
                                flowables.append(img)
                                flowables.append(Spacer(1, 12))
                                
                                # Create and add the legend table with the same format as above
                                legend_table = Table(legend_data, colWidths=[80, 400])
                                legend_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left align label column
                                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),  # Left align answer column
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                    ('WORDWRAP', (1, 1), (1, -1), True)
                                ]))
                                flowables.append(legend_table)
                                
                                # Add page break after this chart and its legend, only if we have more sections coming
                                if has_age_section or has_location_section:
                                    flowables.append(PageBreak())
                                
                            except Exception as e:
                                print(f"Error creating gender chart for question {q_id}: {e}")
                        
                        # 3. By Age Group (if age_group exists and has enough data)
                        if 'age_group' in df.columns and df['age_group'].notna().sum() > 10:
                            try:
                                # Get age groups with sufficient data
                                valid_age_groups = df['age_group'].value_counts()
                                valid_age_groups = valid_age_groups[valid_age_groups > 5].index
                                
                                # Same approach as gender analysis but for age groups
                                age_chart_data = []
                                age_q_data = q_data[q_data['age_group'].isin(valid_age_groups)]
                                
                                for _, row in age_q_data.iterrows():
                                    age_group = row['age_group']
                                    answer = row['answer']
                                    
                                    # Handle list answers if needed
                                    if list_answers and answer and isinstance(answer, str) and (answer.startswith('[') or answer.startswith('{')):
                                        try:
                                            ans_list = json.loads(answer.replace("'", "\""))
                                            if isinstance(ans_list, list):
                                                for a in ans_list:
                                                    if a in response_counts.index:
                                                        age_chart_data.append({'age_group': age_group, 'answer': a})
                                            elif isinstance(ans_list, dict):
                                                for a in ans_list.values():
                                                    if a in response_counts.index:
                                                        age_chart_data.append({'age_group': age_group, 'answer': a})
                                            else:
                                                if answer in response_counts.index:
                                                    age_chart_data.append({'age_group': age_group, 'answer': answer})
                                        except:
                                            if answer in response_counts.index:
                                                age_chart_data.append({'age_group': age_group, 'answer': answer})
                                    elif answer in response_counts.index:
                                        age_chart_data.append({'age_group': age_group, 'answer': answer})
                                
                                # Convert to DataFrame
                                age_df = pd.DataFrame(age_chart_data)
                                
                                # Skip if no data after filtering
                                if len(age_df) == 0:
                                    raise ValueError("No age data for this question after filtering")
                                
                                # Create cross-tabulation
                                age_cross = pd.crosstab(age_df['age_group'], age_df['answer'])
                                
                                # Create grouped bar chart
                                plt.figure(figsize=(10, 6))
                                
                                # Prepare data with mapped labels
                                x = np.arange(len(valid_age_groups))
                                width = 0.15
                                
                                # Plot bars for each answer (up to 5)
                                for i, answer in enumerate(response_counts.index):
                                    if answer in age_cross.columns:
                                        age_values = []
                                        for age_group in valid_age_groups:
                                            age_values.append(age_cross.at[age_group, answer] if age_group in age_cross.index and answer in age_cross.columns else 0)
                                        
                                        offset = width * (i - len(response_counts.index)/2 + 0.5)
                                        plt.bar(x + offset, age_values, width, 
                                               label=answer_map[answer], 
                                               color=plt.cm.tab10.colors[i % 10])
                                
                                plt.xlabel('Age Group')
                                plt.ylabel('Count')
                                plt.title('Responses by Age Group')
                                plt.xticks(x, valid_age_groups)
                                plt.legend(title="Response Options")
                                
                                plt.tight_layout()
                                
                                # Save chart to temp file
                                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                                temp_file_name = temp_file.name
                                temp_files.append(temp_file_name)
                                plt.savefig(temp_file_name, format='png', dpi=150)
                                plt.close()
                                temp_file.close()
                                
                                # Add chart to PDF with its own heading and legend on the same page
                                flowables.append(Paragraph("Responses by Age Group", styles['Heading4']))
                                img = Image(temp_file_name, width=450, height=300)
                                flowables.append(img)
                                flowables.append(Spacer(1, 12))
                                
                                # Create and add the legend table with the same format as above
                                legend_table = Table(legend_data, colWidths=[80, 400])
                                legend_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left align label column
                                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),  # Left align answer column
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                    ('WORDWRAP', (1, 1), (1, -1), True)
                                ]))
                                flowables.append(legend_table)
                                
                                # Add page break after this chart and its legend, only if we have more sections coming
                                if has_location_section:
                                    flowables.append(PageBreak())
                                
                            except Exception as e:
                                print(f"Error creating age group chart for question {q_id}: {e}")
                        
                        # 4. By Location (if location data exists)
                        if 'location' in df.columns and df['location'].notna().sum() > 10:
                            try:
                                # Get top locations
                                top_locations = df['location'].value_counts().nlargest(5).index
                                
                                # Same approach as gender analysis but for locations
                                location_chart_data = []
                                location_q_data = q_data[q_data['location'].isin(top_locations)]
                                
                                for _, row in location_q_data.iterrows():
                                    location = row['location']
                                    answer = row['answer']
                                    
                                    # Handle list answers if needed
                                    if list_answers and answer and isinstance(answer, str) and (answer.startswith('[') or answer.startswith('{')):
                                        try:
                                            ans_list = json.loads(answer.replace("'", "\""))
                                            if isinstance(ans_list, list):
                                                for a in ans_list:
                                                    if a in response_counts.index:
                                                        location_chart_data.append({'location': location, 'answer': a})
                                            elif isinstance(ans_list, dict):
                                                for a in ans_list.values():
                                                    if a in response_counts.index:
                                                        location_chart_data.append({'location': location, 'answer': a})
                                            else:
                                                if answer in response_counts.index:
                                                    location_chart_data.append({'location': location, 'answer': answer})
                                        except:
                                            if answer in response_counts.index:
                                                location_chart_data.append({'location': location, 'answer': answer})
                                    elif answer in response_counts.index:
                                        location_chart_data.append({'location': location, 'answer': answer})
                                
                                # Convert to DataFrame
                                location_df = pd.DataFrame(location_chart_data)
                                
                                # Skip if no data after filtering
                                if len(location_df) == 0:
                                    raise ValueError("No location data for this question after filtering")
                                
                                # Create cross-tabulation
                                location_cross = pd.crosstab(location_df['location'], location_df['answer'])
                                
                                # Create grouped bar chart
                                plt.figure(figsize=(12, 7))
                                
                                # Prepare data with mapped labels
                                x = np.arange(len(top_locations))
                                width = 0.15
                                
                                # Plot bars for each answer (up to 5)
                                for i, answer in enumerate(response_counts.index):
                                    if answer in location_cross.columns:
                                        location_values = []
                                        for location in top_locations:
                                            location_values.append(location_cross.at[location, answer] if location in location_cross.index and answer in location_cross.columns else 0)
                                        
                                        offset = width * (i - len(response_counts.index)/2 + 0.5)
                                        plt.bar(x + offset, location_values, width, 
                                               label=answer_map[answer], 
                                               color=plt.cm.tab10.colors[i % 10])
                                
                                plt.xlabel('Location')
                                plt.ylabel('Count')
                                plt.title('Responses by Location')
                                plt.xticks(x, top_locations, rotation=45, ha='right')
                                plt.legend(title="Response Options")
                                
                                plt.tight_layout()
                                
                                # Save chart to temp file
                                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                                temp_file_name = temp_file.name
                                temp_files.append(temp_file_name)
                                plt.savefig(temp_file_name, format='png', dpi=150, bbox_inches='tight')
                                plt.close()
                                temp_file.close()
                                
                                # Add chart to PDF with its own heading and legend on the same page
                                flowables.append(Paragraph("Responses by Location", styles['Heading4']))
                                img = Image(temp_file_name, width=450, height=300)
                                flowables.append(img)
                                flowables.append(Spacer(1, 12))
                                
                                # Create and add the legend table with the same format as above
                                legend_table = Table(legend_data, colWidths=[80, 400])
                                legend_table.setStyle(TableStyle([
                                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                                    ('ALIGN', (0, 0), (0, -1), 'LEFT'),  # Left align label column
                                    ('ALIGN', (1, 0), (1, -1), 'LEFT'),  # Left align answer column
                                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                    ('WORDWRAP', (1, 1), (1, -1), True)
                                ]))
                                flowables.append(legend_table)
                                
                                # Don't add page break after the last chart's legend
                                # (will add page break after the entire question instead)
                                
                            except Exception as e:
                                print(f"Error creating location chart for question {q_id}: {e}")
                        
                        # Add page break to move to the next question
                        flowables.append(PageBreak())
                  
            except Exception as e:
                # If anything goes wrong with data processing, print the error but continue
                print(f"Error generating demographic visualizations: {e}")
                error_msg = Paragraph(f"Error generating demographic visualizations: {str(e)}", styles['Normal'])
                flowables.append(error_msg)
                flowables.append(Spacer(1, 12))
                flowables.append(PageBreak())
        
        # Process the analysis content - ensure proper headings
        if isinstance(final_response, dict) and 'final_insights' in final_response:
            content_text = final_response.get('final_insights', '')
        else:
            content_text = str(final_response)
        
        # If content doesn't start with "Analysis" heading, add it
        if not content_text.startswith("# Analysis"):
            content_text = "# Analysis\n" + content_text
        
        # Ensure all required sections are present
        required_sections = [
            "# Analysis",
            "## Key Insights from Questionnaire Analysis",
            "## Demographic Analysis",
            "### Age Distribution Analysis",
            "### Gender Distribution Analysis", 
            "### Location Distribution Analysis",
            "## Recommendations",
            "## Limitations"
        ]
        
        # Check content for required sections and add any missing ones
        content_lines = content_text.split('\n')
        sections_present = {section: False for section in required_sections}
        
        for line in content_lines:
            for section in required_sections:
                if line.strip() == section:
                    sections_present[section] = True
        
        # Add any missing sections at the end
        for section, present in sections_present.items():
            if not present:
                content_text += f"\n{section}\n"
        
        # Split the content into paragraphs
        content_parts = content_text.split('\n')
        
        # Add the content
        for part in content_parts:
            if part.strip():  # Skip empty lines
                if part.startswith('# '):
                    p = Paragraph(part[2:], styles['Heading1'])
                elif part.startswith('## '):
                    p = Paragraph(part[3:], styles['Heading2'])
                elif part.startswith('### '):
                    p = Paragraph(part[4:], styles['Heading3'])
                else:
                    p = Paragraph(part, styles['Normal'])
                flowables.append(p)
                flowables.append(Spacer(1, 6))
        
        # Build the PDF
        doc.build(flowables)
        
        # Now that the PDF is built, clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {temp_file}: {e}")
        
        print(f"Report saved to: {pdf_file}")
        return pdf_file

    def run(self) -> Iterator[RunResponse]:
        try:
            # Get all the data from the database
            data = self.fetch_data()
            
            # Convert data to a string representation
            # Create a list of dictionaries from row objects
            formatted_data = self.format_data(data)

            # Initialize results list
            self.analysis_results = []
            
            # Step 1: Categorize responses by demographics
            print("  Categorizing responses by age...")
            age_groups = self.categorize_by_age(formatted_data)
            print("  Categorizing responses by gender...")
            gender_groups = self.categorize_by_gender(formatted_data)
            print("  Categorizing responses by location...")
            location_groups = self.categorize_by_location(formatted_data)
            
            # Step 2: Analyze demographic distribution
            print("  Analyzing demographic distribution...")
            
            # Age distribution
            age_distribution = AnalysisResult(
                category="age",
                question_id=None,
                segments={group: len(responses) for group, responses in age_groups.items()},
                insights=self.data_analysis_agent.run(
                    f"Analyze the age distribution of respondents: {json.dumps({group: len(responses) for group, responses in age_groups.items()})}"
                ).content
            )
            self.analysis_results.append(age_distribution)
            
            # Gender distribution
            gender_distribution = AnalysisResult(
                category="gender",
                question_id=None,
                segments={group: len(responses) for group, responses in gender_groups.items()},
                insights=self.data_analysis_agent.run(
                    f"Analyze the gender distribution of respondents: {json.dumps({group: len(responses) for group, responses in gender_groups.items()})}"
                ).content
            )
            self.analysis_results.append(gender_distribution)
            
            # Location distribution
            location_distribution = AnalysisResult(
                category="location",
                question_id=None,
                segments={group: len(responses) for group, responses in location_groups.items()},
                insights=self.data_analysis_agent.run(
                    f"Analyze the location distribution of respondents: {json.dumps({group: len(responses) for group, responses in location_groups.items()})}"
                ).content
            )
            self.analysis_results.append(location_distribution)

           
            # Data analysis
            data_analysis_response = self.data_analysis_agent.run(formatted_data)
            # Extract just the content from the RunResponse
            data_analysis_content = data_analysis_response.content

            # Data Insights
            insights_response = self.insights_agent.run(data_analysis_content)
            # Extract just the content from the RunResponse
            insights_content = insights_response.content

            print("Compiling the final report...")
            final_insights = self.generate_final_insights(data)

            # Prepare results
            result_summary = {
                "total_responses": len(data),
                # "questions": sorted(list(questions.keys())),  # Include the list of questions
                "demographic_analysis": {
                    "age": self.analysis_results[0].segments if len(self.analysis_results) > 0 else {},
                    "gender": self.analysis_results[1].segments if len(self.analysis_results) > 1 else {},
                    "location": self.analysis_results[2].segments if len(self.analysis_results) > 2 else {}
                },
                # "question_analysis_count": 0,  # Start with 0 since we're doing on-demand analysis
                "final_insights": final_insights
            }

            # Generate the report in PDF format
            self.generate_report_pdf(result_summary, data)

            yield RunResponse(
                content=result_summary, event=RunEvent.workflow_completed
            )

        except Exception as e:
            print(f"Error: {e}")
            raise e
        

def main():
    general_analysis = GeneralAnalysisWorkflow(
        name="General Analysis",
        description="An agent that can analyze data and provide insights",
        session_id=f"polls-analysis-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        debug_mode=True,
    )
    
    # Run the workflow
    for response in general_analysis.run():
        # Print the content of the response
        print("\n--- RESULTS ---\n")
        print(response.content)
        print("\n--- END OF RESULTS ---\n")

if __name__ == "__main__":
    main()


