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

# .env variables
dotenv.load_dotenv()

# AI endpoint
AI_ENDPOINT = os.getenv("AI_ENDPOINT")

class GeneralAnalysis(BaseModel):
    sentiment: str = Field(description="The sentiment of the data")
    analysis: str = Field(description="The analysis of the data")
    conclusion: str = Field(description="The conclusion of the analysis")

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
            "IMPORTANT: The IPRS (Integrated Population Registration System) data contains official biographical",
            "information like date of birth, gender, and location information. This data is stored in the 'iprs'",
            "property within the 'user' object. Always prioritize IPRS data when available as it's the most",
            "reliable source of demographic information.",
            "",
            "For age analysis, look for date_of_birth in IPRS data, with segments typically including:",
            "- under_18",
            "- 18-24",
            "- 25-34",
            "- 35-44",
            "- 45-54",
            "- 55+",
            "",
            "For gender analysis, look for gender in IPRS data, with segments typically including:",
            "- male",
            "- female",
            "- other",
            "",
            "For location analysis, look for these fields in IPRS data (in order of priority):",
            "- county_of_birth",
            "- district_of_birth",
            "- division_of_birth",
            "- location_of_birth",
            "- nationality"
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
            "Synthesize the analysis into clear, actionable insights.",
            "Identify the most significant trends and patterns across all demographic segments.",
            "Highlight unexpected or counterintuitive findings that merit further investigation.",
            "Present your insights in a structured, easy-to-understand format.",
            "Ensure your insights are supported by the data and avoid overinterpreting minor variations.",
            "",
            "IMPORTANT: The IPRS (Integrated Population Registration System) data contains official biographical",
            "information like date of birth, gender, and location information. This data is stored in the 'iprs'",
            "property within the 'user' object. This is the most reliable source of demographic information.",
            "",
            "When analyzing patterns in the data, consider:",
            "1. Age differences from IPRS date_of_birth",
            "2. Gender differences from IPRS gender field",
            "3. Geographic differences from IPRS location fields (county_of_birth, district_of_birth)",
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

    def fetch_data(self):
        # Create database URL from secrets
        db_url = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)

        # Get all the data from the database
        with engine.connect() as conn:
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
                if len(unique_respondents_df) != len(df):
                    summary_data.append(["Total Records", str(len(df))])
                
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
                    unique_locations = unique_respondents_df['location'].nunique()
                    summary_data.append(["Unique Locations", str(unique_locations)])
                    
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
                else:
                    if 'age' in df.columns:
                        print(f"Age column exists but contains no valid data: {df['age'].describe()}")
                    else:
                        print("No age column was created from date_of_birth")
                
            except Exception as e:
                # If anything goes wrong with data processing, print the error but continue
                print(f"Error generating demographic visualizations: {e}")
                error_msg = Paragraph(f"Error generating demographic visualizations: {str(e)}", styles['Normal'])
                flowables.append(error_msg)
                flowables.append(Spacer(1, 12))
                flowables.append(PageBreak())
        
        # Add the AI analysis section
        analysis_heading = Paragraph("Analysis", styles['Heading2'])
        flowables.append(analysis_heading)
        flowables.append(Spacer(1, 12))
        
        # Split the content into paragraphs
        content_parts = final_response.content.split('\n')
        
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

    def run(self):
        try:
            # Get all the data from the database
            data = self.fetch_data()
            
            # Convert data to a string representation
            # Create a list of dictionaries from row objects
            formatted_data = self.format_data(data)
           
            # Data analysis
            data_analysis_response = self.data_analysis_agent.run(formatted_data)
            # Extract just the content from the RunResponse
            data_analysis_content = data_analysis_response.content

            # Data Insights
            insights_response = self.insights_agent.run(data_analysis_content)
            # Extract just the content from the RunResponse
            insights_content = insights_response.content


            # Compile the final report
            final_response: RunResponse = self.report_generation.run(
                json.dumps(
                    {
                        "data_analysis": data_analysis_content,
                        "data_insights": insights_content,
                    },
                    indent=4,
                )
            )

            # Generate the report in PDF format
            self.generate_report_pdf(final_response, data)

            yield RunResponse(
                content=final_response.content, event=RunEvent.workflow_completed
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


