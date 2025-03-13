import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from typing import Dict, List, Any
import dotenv  # Add this package (pip install python-dotenv)
import matplotlib.pyplot as plt
import numpy as np

# Load environment variables from .env file
dotenv.load_dotenv()

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Questionnaire Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add the parent directory to the Python path to make imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the polls analysis workflow
from inc_polls_analysis_agent import PollsAnalysisWorkflow, AnalysisResult

# Get API endpoint from environment variable with fallback
DEFAULT_API_ENDPOINT = "https://incapidev.reflow.co.ke/api/v2/fetchsurveydata/16"
API_ENDPOINT = os.environ.get("POLLS_API_ENDPOINT", DEFAULT_API_ENDPOINT)

# Define cache timeout (4 hours in seconds)
CACHE_TIMEOUT = 4 * 60 * 60

# Add CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .category-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #424242;
        margin-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache the workflow creation separately using cache_resource (for non-serializable objects)
@st.cache_resource(ttl=CACHE_TIMEOUT)
def create_workflow(api_endpoint):
    """Create and return a workflow instance (cached as a resource)"""
    print(f"Creating new workflow instance for {api_endpoint}")
    return PollsAnalysisWorkflow(
        api_endpoint=api_endpoint,
        description="Polls Analysis Workflow",
        session_id=f"polls-analysis-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        debug_mode=True,
    )

# Cache the analysis results using cache_data (for serializable data)
@st.cache_data(ttl=CACHE_TIMEOUT)
def fetch_and_analyze_data(api_endpoint):
    """Fetch and analyze data with caching"""
    print(f"Fetching fresh data from {api_endpoint}")
    
    # Get the workflow from the resource cache
    workflow = create_workflow(api_endpoint)
    
    # Run the workflow and get the results
    results = list(workflow.run())
    
    if results and results[0].content:
        try:
            summary_results = json.loads(results[0].content)
            return summary_results, workflow.analysis_results, workflow.responses
        except Exception as e:
            print(f"Error processing results: {str(e)}")
            return None, [], []
    
    print("No results returned from analysis")
    return None, [], []

def display_visualizations(results, analysis_results, responses):
    """Display visualizations based on analysis results"""
    
    # Overview metrics
    st.markdown('<div class="sub-header">Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Responses", results.get("total_responses", 0))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        # Safely get the question_analysis_count or calculate it
        if "question_analysis_count" in results:
            question_count = results["question_analysis_count"]
        else:
            # Calculate it based on analysis_results
            # Count unique question_ids that aren't None
            question_ids = set()
            for result in analysis_results:
                if result.question_id is not None:
                    question_ids.add(result.question_id)
            question_count = len(question_ids)
            
        st.metric("Questions Analyzed", question_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        # Count demographic categories
        demographic_categories = set()
        for result in analysis_results:
            if result.category:
                demographic_categories.add(result.category)
        
        st.metric("Demographic Categories", len(demographic_categories))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # General demographic visualizations
    st.markdown('<div class="sub-header">Demographic Distribution</div>', unsafe_allow_html=True)
    
    demo_col1, demo_col2 = st.columns(2)
    
    with demo_col1:
        # Age distribution
        age_data = results["demographic_analysis"]["age"]
        if age_data:
            age_df = pd.DataFrame({
                "Age Group": list(age_data.keys()),
                "Count": list(age_data.values())
            })
            
            # Create pie chart
            fig = px.pie(
                age_df, 
                values="Count", 
                names="Age Group",
                title="Distribution by Age Group",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Find the general age analysis
            for result in analysis_results:
                if result.category == "age" and result.question_id is None:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown('<div class="category-title">Age Group Insights</div>', unsafe_allow_html=True)
                    st.markdown(result.insights)
                    st.markdown('</div>', unsafe_allow_html=True)
                    break
    
    with demo_col2:
        # Gender distribution
        gender_data = results["demographic_analysis"]["gender"]
        if gender_data:
            gender_df = pd.DataFrame({
                "Gender": list(gender_data.keys()),
                "Count": list(gender_data.values())
            })
            
            # Create pie chart
            fig = px.pie(
                gender_df, 
                values="Count", 
                names="Gender",
                title="Distribution by Gender",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Find the general gender analysis
            for result in analysis_results:
                if result.category == "gender" and result.question_id is None:
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown('<div class="category-title">Gender Insights</div>', unsafe_allow_html=True)
                    st.markdown(result.insights)
                    st.markdown('</div>', unsafe_allow_html=True)
                    break
    
    # Location distribution
    st.markdown('<div class="sub-header">Location Distribution</div>', unsafe_allow_html=True)
    location_data = results["demographic_analysis"]["location"]
    if location_data:
        location_df = pd.DataFrame({
            "Location": list(location_data.keys()),
            "Count": list(location_data.values())
        })
        
        # Create bar chart
        fig = px.bar(
            location_df, 
            x="Location", 
            y="Count",
            title="Distribution by Location",
            color="Count",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Find the general location analysis
        for result in analysis_results:
            if result.category == "location" and result.question_id is None:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown('<div class="category-title">Location Insights</div>', unsafe_allow_html=True)
                st.markdown(result.insights)
                st.markdown('</div>', unsafe_allow_html=True)
                break
    
    # Question Analysis
    st.markdown('<div class="sub-header">Question Analysis</div>', unsafe_allow_html=True)
    
    # Get unique questions
    question_ids = set()
    for result in analysis_results:
        if result.question_id is not None:
            question_ids.add(result.question_id)
    
    # Create dropdown for selecting question
    selected_question = st.selectbox(
        "Select a question for detailed analysis:",
        options=list(question_ids)
    )
    
    if selected_question:
        st.markdown(f"### Analysis for Question: {selected_question}")
        
        # Create tabs for different demographic breakdowns
        tabs = st.tabs(["Age Analysis", "Gender Analysis", "Location Analysis"])
        
        # Age tab
        with tabs[0]:
            for result in analysis_results:
                if result.category == "age" and result.question_id == selected_question:
                    # Visualize the segments
                    segments_df = pd.DataFrame({
                        "Age Group": list(result.segments.keys()),
                        "Response Count": list(result.segments.values())
                    })
                    
                    fig = px.bar(
                        segments_df,
                        x="Age Group",
                        y="Response Count",
                        title=f"Responses by Age Group for Question {selected_question}",
                        color="Response Count",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display insights
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown('<div class="category-title">Age Group Insights for this Question</div>', unsafe_allow_html=True)
                    st.markdown(result.insights)
                    st.markdown('</div>', unsafe_allow_html=True)
                    break
        
        # Gender tab
        with tabs[1]:
            for result in analysis_results:
                if result.category == "gender" and result.question_id == selected_question:
                    # Visualize the segments
                    segments_df = pd.DataFrame({
                        "Gender": list(result.segments.keys()),
                        "Response Count": list(result.segments.values())
                    })
                    
                    fig = px.bar(
                        segments_df,
                        x="Gender",
                        y="Response Count",
                        title=f"Responses by Gender for Question {selected_question}",
                        color="Response Count",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display insights
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown('<div class="category-title">Gender Insights for this Question</div>', unsafe_allow_html=True)
                    st.markdown(result.insights)
                    st.markdown('</div>', unsafe_allow_html=True)
                    break
        
        # Location tab
        with tabs[2]:
            for result in analysis_results:
                if result.category == "location" and result.question_id == selected_question:
                    # Visualize the segments
                    segments_df = pd.DataFrame({
                        "Location": list(result.segments.keys()),
                        "Response Count": list(result.segments.values())
                    })
                    
                    fig = px.bar(
                        segments_df,
                        x="Location",
                        y="Response Count",
                        title=f"Responses by Location for Question {selected_question}",
                        color="Response Count",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display insights
                    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                    st.markdown('<div class="category-title">Location Insights for this Question</div>', unsafe_allow_html=True)
                    st.markdown(result.insights)
                    st.markdown('</div>', unsafe_allow_html=True)
                    break
    
    # Final Insights
    st.markdown('<div class="sub-header">Overall Analysis Insights</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(results["final_insights"])
    st.markdown('</div>', unsafe_allow_html=True)

def display_question_analysis(analysis_results, responses, workflow=None):
    """Display detailed analysis for a selected question"""
    st.header("Question-Level Analysis", divider="blue")
    
    # Initialize session state for storing analysis results
    if 'analyzed_questions' not in st.session_state:
        st.session_state.analyzed_questions = {}
    
    # Check if we have responses
    if not responses or len(responses) == 0:
        st.warning("No responses available for analysis.")
        return
    
    # Extract questions from formData
    all_questions = []
    question_counts = {}
    
    for response in responses:
        if isinstance(response, dict) and 'formData' in response and isinstance(response['formData'], dict):
            for key in response['formData'].keys():
                if key not in all_questions:
                    all_questions.append(key)
                    question_counts[key] = 1
                else:
                    question_counts[key] = question_counts.get(key, 0) + 1
    
    # Sort questions by frequency (most common first)
    all_questions = sorted(all_questions, key=lambda q: question_counts.get(q, 0), reverse=True)
    
    if not all_questions:
        st.warning("No questions were extracted from the responses.")
        return
    
    # Display dropdown for question selection
    selected_question = st.selectbox(
        "Select a question for detailed analysis",
        options=all_questions,
        index=0,
        key="question_selector"
    )
    
    if selected_question:
        st.subheader(f"Analysis of: {selected_question}")
        
        # Count all answers for this question
        answers = []
        age_data = {}
        gender_data = {}
        location_data = {}
        
        for response in responses:
            if isinstance(response, dict) and 'formData' in response:
                form_data = response['formData']
                if isinstance(form_data, dict) and selected_question in form_data:
                    answer = str(form_data[selected_question])
                    answers.append(answer)
                    
                    # Extract demographic info for visualization
                    user = response.get('user', {})
                    iprs = user.get('iprs', {})
                    
                    # Age categorization
                    age_group = 'unknown'
                    if iprs and iprs.get('date_of_birth'):
                        try:
                            dob = iprs['date_of_birth']
                            # Very simplified age calculation - just for UI
                            age = 2024 - int(dob.split('-')[0]) 
                            if age < 18: age_group = 'under_18'
                            elif age < 25: age_group = '18-24'
                            elif age < 35: age_group = '25-34'
                            elif age < 45: age_group = '35-44'
                            elif age < 55: age_group = '45-54'
                            else: age_group = '55+'
                        except:
                            pass
                    
                    if age_group not in age_data:
                        age_data[age_group] = []
                    age_data[age_group].append(answer)
                    
                    # Gender categorization
                    gender = 'unknown'
                    if iprs and iprs.get('gender'):
                        gender = iprs['gender'].lower()
                        if gender not in ['male', 'female']:
                            gender = 'other'
                    
                    if gender not in gender_data:
                        gender_data[gender] = []
                    gender_data[gender].append(answer)
                    
                    # Location categorization
                    location = 'unknown'
                    if iprs:
                        if iprs.get('county_of_birth'):
                            location = iprs['county_of_birth']
                        elif iprs.get('district_of_birth'):
                            location = iprs['district_of_birth']
                        elif iprs.get('nationality'):
                            location = iprs['nationality']
                    
                    if location not in location_data:
                        location_data[location] = []
                    location_data[location].append(answer)
        
        # Show answers distribution
        if not answers:
            st.warning("No answers found for this question.")
            return
            
        st.write(f"Total responses to this question: {len(answers)}")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Overall Distribution", "Demographic Analysis"])
        
        with tab1:
            # Create basic answer distribution
            from collections import Counter
            answer_counts = Counter(answers)
            
            # Convert to DataFrame for visualization
            if answer_counts:
                df = pd.DataFrame({
                    'Answer': list(answer_counts.keys()),
                    'Count': list(answer_counts.values())
                })
                
                # Sort by count descending
                df = df.sort_values('Count', ascending=False)
                
                # Show table
                st.dataframe(df)
                
                # Create chart - only if there aren't too many categories
                if len(df) <= 10:
                    st.bar_chart(df.set_index('Answer'))
                else:
                    # Take top 10 for chart
                    st.bar_chart(df.head(10).set_index('Answer'))
                    st.info("Chart shows top 10 responses only")
            else:
                st.warning("No answers found for this question")
        
        with tab2:
            st.subheader("Demographic Analysis")
            
            # Show any existing analysis results
            existing_results = [
                result for result in analysis_results 
                if hasattr(result, 'question_id') and result.question_id == selected_question
            ]
            
            # Also check session state
            if selected_question in st.session_state.analyzed_questions:
                existing_results.extend(st.session_state.analyzed_questions[selected_question])
            
            # Group results by category for better organization
            categorized_results = {
                "age": [],
                "gender": [],
                "location": []
            }
            
            for result in existing_results:
                if hasattr(result, 'category') and result.category in categorized_results:
                    categorized_results[result.category].append(result)
            
            # Add analysis option selector
            analysis_type = st.selectbox(
                "Analysis type:",
                options=["By Age", "By Gender", "By Location"],
                index=0,
                key=f"analysis_type_{selected_question}"
            )
            
            # Display the appropriate demographic distribution based on selected type
            if analysis_type == "By Age":
                # Age distribution visualization
                st.subheader("Response Distribution by Age Group")
                
                # Show age group distribution
                age_counts = {age: len(answers) for age, answers in age_data.items() if answers}
                if age_counts:
                    age_df = pd.DataFrame({
                        'Age Group': list(age_counts.keys()),
                        'Count': list(age_counts.values())
                    })
                    age_df = age_df.sort_values('Count', ascending=False)
                    st.bar_chart(age_df.set_index('Age Group'))
                    
                    # Show popular answers by age
                    st.subheader("Top Answer by Age Group")
                    for age, answers_list in age_data.items():
                        if answers_list:
                            age_counter = Counter(answers_list)
                            top_answer = age_counter.most_common(1)[0][0]
                            percentage = (age_counter[top_answer] / len(answers_list)) * 100
                            st.write(f"**{age}**: {top_answer} ({percentage:.1f}%)")
                else:
                    st.info("No age data available for analysis")
                
                # Show AI analysis for this category if available
                if categorized_results["age"]:
                    st.subheader("AI Analysis by Age")
                    for result in categorized_results["age"]:
                        with st.expander(f"Age Analysis ({result.category})"):
                            st.markdown(result.insights)
                            
                            # Create visualization from the segments data if available
                            if hasattr(result, 'segments') and result.segments:
                                segments_df = pd.DataFrame({
                                    'Age Group': list(result.segments.keys()),
                                    'Count': list(result.segments.values())
                                })
                                st.bar_chart(segments_df.set_index('Age Group'))
                
            elif analysis_type == "By Gender":
                # Gender distribution visualization
                st.subheader("Response Distribution by Gender")
                
                # Show gender distribution
                gender_counts = {gender: len(answers) for gender, answers in gender_data.items() if answers}
                if gender_counts:
                    gender_df = pd.DataFrame({
                        'Gender': list(gender_counts.keys()),
                        'Count': list(gender_counts.values())
                    })
                    gender_df = gender_df.sort_values('Count', ascending=False)
                    st.bar_chart(gender_df.set_index('Gender'))
                    
                    # Create pie chart for gender distribution
                    fig, ax = plt.subplots()
                    ax.pie(gender_counts.values(), labels=gender_counts.keys(), autopct='%1.1f%%')
                    ax.set_title("Responses by Gender")
                    st.pyplot(fig)
                    
                    # Show popular answers by gender
                    st.subheader("Top Answer by Gender")
                    for gender, answers_list in gender_data.items():
                        if answers_list:
                            gender_counter = Counter(answers_list)
                            top_answer = gender_counter.most_common(1)[0][0]
                            percentage = (gender_counter[top_answer] / len(answers_list)) * 100
                            st.write(f"**{gender}**: {top_answer} ({percentage:.1f}%)")
                else:
                    st.info("No gender data available for analysis")
                
                # Show AI analysis for this category if available
                if categorized_results["gender"]:
                    st.subheader("AI Analysis by Gender")
                    for result in categorized_results["gender"]:
                        with st.expander(f"Gender Analysis ({result.category})"):
                            st.markdown(result.insights)
                            
                            # Create visualization from the segments data if available
                            if hasattr(result, 'segments') and result.segments:
                                segments_df = pd.DataFrame({
                                    'Gender': list(result.segments.keys()),
                                    'Count': list(result.segments.values())
                                })
                                st.bar_chart(segments_df.set_index('Gender'))
                
            else:  # By Location
                # Location distribution visualization
                st.subheader("Response Distribution by Location")
                
                # Show location distribution (top 10 locations)
                location_counts = {loc: len(answers) for loc, answers in location_data.items() if answers}
                if location_counts:
                    # Convert to DataFrame and sort
                    location_df = pd.DataFrame({
                        'Location': list(location_counts.keys()),
                        'Count': list(location_counts.values())
                    })
                    location_df = location_df.sort_values('Count', ascending=False)
                    
                    # If too many locations, show only top 10
                    if len(location_df) > 10:
                        st.write("Top 10 Locations:")
                        st.bar_chart(location_df.head(10).set_index('Location'))
                    else:
                        st.bar_chart(location_df.set_index('Location'))
                    
                    # Show popular answers by top 5 locations
                    st.subheader("Top Answers by Location")
                    top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    for location, _ in top_locations:
                        answers_list = location_data[location]
                        if answers_list:
                            location_counter = Counter(answers_list)
                            top_answer = location_counter.most_common(1)[0][0]
                            percentage = (location_counter[top_answer] / len(answers_list)) * 100
                            st.write(f"**{location}**: {top_answer} ({percentage:.1f}%)")
                else:
                    st.info("No location data available for analysis")
                
                # Show AI analysis for this category if available
                if categorized_results["location"]:
                    st.subheader("AI Analysis by Location")
                    for result in categorized_results["location"]:
                        with st.expander(f"Location Analysis ({result.category})"):
                            st.markdown(result.insights)
                            
                            # Create visualization from the segments data if available
                            if hasattr(result, 'segments') and result.segments:
                                # Get top 10 locations by response count
                                top_segments = dict(sorted(
                                    result.segments.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True
                                )[:10])
                                
                                segments_df = pd.DataFrame({
                                    'Location': list(top_segments.keys()),
                                    'Count': list(top_segments.values())
                                })
                                st.bar_chart(segments_df.set_index('Location'))
            
            # Add button to request new analysis
            if workflow:
                col1, col2 = st.columns([5, 1])
                with col2:
                    if st.button("Analyze", key=f"analyze_btn_{selected_question}"):
                        with st.spinner(f"Analyzing '{selected_question}' by {analysis_type.lower()}..."):
                            try:
                                # Get demographic category from analysis type
                                category = analysis_type.lower().split(" ")[1]  # "By Age" -> "age"
                                
                                if category == "age":
                                    groups = workflow.categorize_by_age(workflow.responses)
                                elif category == "gender":
                                    groups = workflow.categorize_by_gender(workflow.responses)
                                else:  # location
                                    groups = workflow.categorize_by_location(workflow.responses)
                                
                                # Run the analysis
                                new_result = workflow.analyze_responses_by_question(
                                    question_id=selected_question,
                                    demographic_category=category,
                                    segmented_responses=groups
                                )
                                
                                # Store the new result
                                if selected_question not in st.session_state.analyzed_questions:
                                    st.session_state.analyzed_questions[selected_question] = []
                                
                                st.session_state.analyzed_questions[selected_question].append(new_result)
                                
                                # Force a rerun to show the new analysis
                                st.rerun()
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                with col1:
                    st.info(f"Click 'Analyze' to generate AI insights about this question by {analysis_type.lower()}")
            else:
                st.warning("Analysis workflow not available. Results shown are static only.")

def extract_questions_from_responses(responses):
    """Extract all unique questions from the responses"""
    all_questions = set()
    
    # Extract questions from formData
    for response in responses:
        if 'formData' in response and isinstance(response['formData'], dict):
            all_questions.update(response['formData'].keys())
    
    # Extract questions from sub_module fields if available
    for response in responses:
        if 'sub_module' in response and 'fields' in response['sub_module']:
            fields = response['sub_module']['fields']
            if isinstance(fields, list):
                for field in fields:
                    if 'name' in field and field['name']:
                        all_questions.add(field['name'])
    
    return sorted(list(all_questions))

def display_demographic_insights(analysis_results):
    """Display demographic insights from the analysis"""
    # ... existing code ...

def display_final_insights(results):
    """Display final insights from the analysis"""
    # ... existing code ...

def main():
    st.title("Questionnaire Analysis Dashboard")
    
    # Initialize session state
    if 'analyzed_questions' not in st.session_state:
        st.session_state.analyzed_questions = {}
        
    # Simple sidebar with info about the data source
    with st.sidebar:
        st.title("Data Source")
        
        # Add refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.last_fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.rerun()
        
        # Add environment info
        st.markdown("---")
        st.subheader("Environment")
        st.text(f"Cache timeout: {CACHE_TIMEOUT // 3600} hours")
        
        # Show last data fetch time
        if 'last_fetch_time' in st.session_state:
            st.text(f"Last fetch: {st.session_state.last_fetch_time}")
    
    # Auto-load data on startup using cached functions
    with st.spinner("Loading data..."):
        try:
            # Get data from cache or fetch new data
            results, analysis_results, responses = fetch_and_analyze_data(API_ENDPOINT)
            
            # Get the workflow object separately from cache
            workflow = create_workflow(API_ENDPOINT)
            
            # Store last fetch time if not set
            if 'last_fetch_time' not in st.session_state:
                st.session_state.last_fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if results is None:
                st.error("No data could be retrieved from the API.")
                return
            
            # Display the dashboard sections
            display_visualizations(results, analysis_results, responses)
            display_demographic_insights(analysis_results)
            display_question_analysis(analysis_results, responses, workflow)
            display_final_insights(results)
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

# Call the main function
if __name__ == "__main__":
    main()