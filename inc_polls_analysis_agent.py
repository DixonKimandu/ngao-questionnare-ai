import json
import random
import os
import sys
from datetime import datetime
from typing import Iterator, Optional, List, Dict, Any, Tuple
from collections import Counter, defaultdict

# Add the parent directory to the Python path to make imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agno.agent import Agent
from agno.models.ollama import Ollama  # You can replace with OpenAI or other models
from agno.storage.workflow.sqlite import SqliteWorkflowStorage
from agno.utils.log import logger
from agno.workflow import RunEvent, RunResponse, Workflow
from pydantic import BaseModel, Field
import requests

# Import questionnaire models
from data.questionnaire_model import User, QuestionDependency, QuestionField, SubModule, Response, QuestionnaireData, SubModuleResponse

class AnalysisResult(BaseModel):
    """Model for analysis results of questionnaire data"""
    category: str  # e.g. 'age', 'gender', 'location'
    question_id: Optional[str] = None  # None for general analysis
    segments: Dict[str, int]  # e.g. {'18-24': 45, '25-34': 32, ...}
    insights: str
    
class PollsAnalysisWorkflow(Workflow):
    """Workflow to analyze questionnaire data by demographic categories"""
    
    # Agent for retrieving and processing data
    data_retrieval_agent: Agent = Agent(
        model=Ollama(id="llama3.1:8b", host="http://localhost:11434"),
        instructions=[
            "You are an agent responsible for retrieving and processing questionnaire data in Kenya.",
            "You will fetch data from an API endpoint and organize it for analysis.",
            "Extract all responses and ensure data is properly structured for demographic analysis.",
            "Handle any data inconsistencies or missing values appropriately.",
            "Provide clear summaries of the data retrieved."
        ],
        add_history_to_messages=True,
        markdown=True,
        debug_mode=False,
    )
    
    # Agent for analyzing data by demographics
    data_analysis_agent: Agent = Agent(
        model=Ollama(id="llama3.1:8b", host="http://localhost:11434"),
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
        model=Ollama(id="llama3.1:8b", host="http://localhost:11434"),
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
    
    def __init__(self, api_endpoint: str, **kwargs):
        super().__init__(**kwargs)
        self.api_endpoint = api_endpoint
        self.responses = []
        self.analysis_results = []
    
    def fetch_questionnaire_data(self) -> List[Dict]:
        """Fetch questionnaire data from the API endpoint or use local JSON file"""
        print(f"📊 Fetching questionnaire data from {self.api_endpoint}...")
        
        # First try the API
        responses = self._fetch_from_api()
        
        # If API fails, try to use the local JSON file
        if not responses:
            print("⚠️ API fetch failed. Trying to use local JSON file...")
            responses = self._fetch_from_local_file()
        
        print(f"✅ Total responses retrieved: {len(responses)}")
        return responses

    def _fetch_from_api(self) -> List[Dict]:
        """Fetch data from the API endpoint"""
        try:
            # First, authenticate and get the bearer token
            base_url = self.api_endpoint.split('/api/')[0]
            auth_url = f"{base_url}/api/v1/user/login"
            
            print(f"🔑 Authenticating with {auth_url}...")
            
            auth_payload = {
                "telephone": "0797275002",  # Replace with actual credentials
                "password": "1234"          # Replace with actual credentials
            }
            
            auth_headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            auth_response = requests.post(auth_url, json=auth_payload, headers=auth_headers)
            
            print(f"🔄 Auth response status: {auth_response.status_code}")
            
            if auth_response.status_code != 200:
                print(f"❌ Authentication failed: {auth_response.text}")
                return []
            
            # Extract the bearer token from the authentication response
            auth_data = auth_response.json()
            bearer_token = auth_data.get('data', {}).get('token')
            
            if not bearer_token:
                print("❌ No token in authentication response")
                return []
                
            print(f"✅ Authentication successful, got token")
            
            # Fetch questionnaire data with the bearer token
            headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {bearer_token}"
            }
            
            print(f"🔄 Fetching data from {self.api_endpoint}...")
            response = requests.get(self.api_endpoint, headers=headers)
            
            print(f"🔄 Data response status: {response.status_code}")
            
            if response.status_code == 200:
                # Parse the response JSON
                raw_data = response.json()
                
                # Check if the expected structure exists
                if not isinstance(raw_data, dict) or 'data' not in raw_data:
                    print(f"❌ Unexpected API response structure")
                    return []
                
                # Get the array of response objects
                response_items = raw_data['data']
                
                if not isinstance(response_items, list):
                    print(f"❌ Data is not a list, it's a {type(response_items)}")
                    return []
                    
                print(f"✅ Found {len(response_items)} responses in API data")
                
                return response_items
            else:
                print(f"❌ Failed to fetch data: {response.text}")
                return []
                
        except Exception as e:
            print(f"❌ Error in API fetch: {str(e)}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            return []

    def _fetch_from_local_file(self) -> List[Dict]:
        """Fetch data from the local JSON file"""
        try:
            # Path to the local JSON file
            file_path = "data/questionnaire_data.json"
            print(f"📂 Reading from local file: {file_path}")
            
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
            
            # Check if the expected structure exists
            if not isinstance(raw_data, dict) or 'data' not in raw_data:
                print(f"❌ Unexpected file structure")
                return []
            
            # Get the array of response objects
            response_items = raw_data['data']
            
            if not isinstance(response_items, list):
                print(f"❌ Data is not a list, it's a {type(response_items)}")
                return []
                
            print(f"✅ Found {len(response_items)} responses in local file")
            
            return response_items
        except Exception as e:
            print(f"❌ Error reading local file: {str(e)}")
            return []
    
    def categorize_by_age(self, responses: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize responses by age groups using IPRS data"""
        age_groups = {
            "under_18": [],
            "18-24": [],
            "25-34": [],
            "35-44": [],
            "45-54": [],
            "55+": [],
            "unknown": []
        }
        
        for response in responses:
            # Get user data and IPRS information
            user = response.get('user', {})
            iprs = user.get('iprs', {})
            
            # If IPRS is null or empty, fallback to other fields
            if not iprs:
                is_adult = user.get('is_adult', '')
                if is_adult == 'Minor':
                    age_groups["under_18"].append(response)
                elif is_adult == 'Adult':
                    # Default adult to 25-34 category
                    age_groups["25-34"].append(response)
                else:
                    age_groups["unknown"].append(response)
                continue
            
            # Check if there's a date_of_birth in IPRS data
            if iprs and isinstance(iprs, dict) and iprs.get('date_of_birth'):
                # Calculate age from date_of_birth
                try:
                    dob_str = iprs.get('date_of_birth')
                    # Handle different date formats
                    if 'T' in dob_str:
                        dob = datetime.fromisoformat(dob_str.replace('Z', '+00:00'))
                    else:
                        # Try different formats
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
                    
                    # Calculate age
                    age = (datetime.now() - dob).days // 365
                    
                    # Categorize by age
                    if age < 18:
                        age_groups["under_18"].append(response)
                    elif age < 25:
                        age_groups["18-24"].append(response)
                    elif age < 35:
                        age_groups["25-34"].append(response)
                    elif age < 45:
                        age_groups["35-44"].append(response)
                    elif age < 55:
                        age_groups["45-54"].append(response)
                    else:
                        age_groups["55+"].append(response)
                except Exception as e:
                    print(f"Error parsing date of birth: {str(e)}")
                    age_groups["unknown"].append(response)
            else:
                # If no DOB in IPRS, check is_adult field
                is_adult = user.get('is_adult', '')
                if is_adult == 'Minor':
                    age_groups["under_18"].append(response)
                elif is_adult == 'Adult':
                    # Default adult to 25-34 category
                    age_groups["25-34"].append(response)
                else:
                    age_groups["unknown"].append(response)
        
        return age_groups
    
    def categorize_by_gender(self, responses: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize responses by gender using IPRS data"""
        gender_groups = {
            "male": [],
            "female": [],
            "other": [],
            "unknown": []
        }
        
        for response in responses:
            # Get user data and IPRS information
            user = response.get('user', {})
            iprs = user.get('iprs', {})
            
            # If IPRS has gender information, use it
            if iprs and isinstance(iprs, dict) and iprs.get('gender'):
                gender = iprs.get('gender', '').lower()
                if gender == 'male':
                    gender_groups["male"].append(response)
                elif gender in ['female', 'femal']: # handle possible typo
                    gender_groups["female"].append(response)
                else:
                    gender_groups["other"].append(response)
            else:
                # Try to infer gender from name if available (simplified approach)
                first_name = user.get('name', '').split(' ')[0].lower() if user.get('name') else ''
                # This is a very basic approach and not reliable for all names
                if first_name in ['john', 'david', 'michael', 'paul', 'james', 'robert', 'william']:
                    gender_groups["male"].append(response)
                elif first_name in ['mary', 'patricia', 'linda', 'barbara', 'elizabeth', 'susan', 'jennifer']:
                    gender_groups["female"].append(response)
                else:
                    gender_groups["unknown"].append(response)
        
        return gender_groups
    
    def categorize_by_location(self, responses: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize responses by location using IPRS data"""
        location_groups = {}
        
        for response in responses:
            # Get user data and IPRS information
            user = response.get('user', {})
            iprs = user.get('iprs', {})
            
            # Try to get location from IPRS
            location = None
            
            if iprs and isinstance(iprs, dict):
                # Try different location fields in priority order
                for field in ['county_of_birth', 'district_of_birth', 'division_of_birth', 'location_of_birth']:
                    if iprs.get(field):
                        location = iprs.get(field)
                        break
            
            # If no location from IPRS, try locationId or nationality
            if not location and iprs and iprs.get('nationality'):
                location = iprs.get('nationality')
            elif not location and user.get('locationId'):
                location = f"Location ID: {user.get('locationId')}"
            
            # If still no location, check if there's a phone number with country code
            if not location and user.get('telephone'):
                phone = user.get('telephone')
                if phone.startswith('+254') or phone.startswith('254') or phone.startswith('07'):
                    location = "Kenya"
            
            # If still no location, use 'unknown'
            if not location:
                location = "Unknown"
            
            # Add to appropriate group
            if location not in location_groups:
                location_groups[location] = []
            
            location_groups[location].append(response)
        
        return location_groups
    
    def analyze_responses_by_question(self, 
                                    question_id: str, 
                                    demographic_category: str,
                                    segmented_responses: Dict[str, List[Dict]]) -> AnalysisResult:
        """Analyze responses for a specific question across demographic segments"""
        
        results = {}
        total_responses = sum(len(responses) for responses in segmented_responses.values())
        
        print(f"Analyzing question: '{question_id}' by {demographic_category}")
        print(f"Total responses to analyze: {total_responses}")
        
        for segment, responses in segmented_responses.items():
            if not responses:
                continue
                
            # Count responses for this question in this segment
            answer_counts = Counter()
            
            for response in responses:
                form_data = response.get('formData', {})
                if form_data and question_id in form_data:
                    answer = form_data[question_id]
                    # Handle both string and list answers
                    if isinstance(answer, list):
                        for item in answer:
                            answer_counts[str(item)] += 1
                    else:
                        answer_counts[str(answer)] += 1
            
            # Calculate percentage of segment that responded to this question
            segment_size = len(responses)
            response_count = sum(answer_counts.values())
            response_rate = response_count / segment_size if segment_size > 0 else 0
            
            # Store results for this segment
            results[segment] = {
                "size": segment_size,
                "response_count": response_count,
                "response_rate": response_rate,
                "answers": dict(answer_counts)
            }
        
        # Print some debug info
        segments_with_answers = sum(1 for segment_data in results.values() if segment_data["response_count"] > 0)
        print(f"Found answers in {segments_with_answers} demographic segments")
        
        # Prepare input for analysis agent
        agent_input = {
            "question_id": question_id,
            "demographic_category": demographic_category,
            "results": results,
            "total_responses": total_responses
        }
        
        # Get insights from analysis agent
        response = self.data_analysis_agent.run(
            json.dumps(agent_input, indent=2)
        )
        
        # Create segment summary for the result
        segment_summary = {segment: data["response_count"] for segment, data in results.items()}
        
        return AnalysisResult(
            category=demographic_category,
            question_id=question_id,
            segments=segment_summary,
            insights=response.content
        )
    
    def analyze_general_demographics(self, 
                                    demographic_category: str,
                                    segmented_responses: Dict[str, List[Response]]) -> AnalysisResult:
        """Analyze general demographic distribution"""
        
        segment_counts = {segment: len(responses) for segment, responses in segmented_responses.items()}
        total_responses = sum(segment_counts.values())
        
        # Prepare input for analysis agent
        agent_input = {
            "demographic_category": demographic_category,
            "segment_counts": segment_counts,
            "total_responses": total_responses
        }
        
        # Get insights from analysis agent
        response = self.data_analysis_agent.run(
            json.dumps(agent_input, indent=2)
        )
        
        return AnalysisResult(
            category=demographic_category,
            question_id=None,  # None indicates general demographic analysis
            segments=segment_counts,
            insights=response.content
        )
    
    def generate_final_insights(self) -> str:
        """Generate final insights based on all analysis results"""
        
        agent_input = {
            "analysis_results": [result.dict() for result in self.analysis_results],
            "total_responses": len(self.responses)
        }
        
        response = self.insights_agent.run(
            json.dumps(agent_input, indent=2)
        )
        
        return response.content
    
    def extract_questions(self, responses: List[Response]) -> Dict[str, str]:
        """Extract all unique questions from the responses"""
        questions = {}
        
        for response in responses:
            if hasattr(response, 'formData') and isinstance(response.formData, dict):
                for question_id in response.formData.keys():
                    if question_id not in questions:
                        questions[question_id] = question_id  # In real implementation, you'd get the question text
        
        return questions
    
    def run(self) -> Iterator[RunResponse]:
        """Run the polls analysis workflow"""
        print("\n🔍 Starting polls analysis workflow")
        
        try:
            # Fetch questionnaire data
            print(f"  Fetching data from {self.api_endpoint}...")
            self.responses = self.fetch_questionnaire_data()
            
            if not self.responses:
                yield RunResponse(
                    content=json.dumps({
                        "error": "Failed to fetch questionnaire data",
                        "total_responses": 0,
                        "question_analysis_count": 0,
                        "demographic_analysis": {
                            "age": {},
                            "gender": {},
                            "location": {}
                        },
                        "final_insights": "No data was retrieved from the API."
                    }),
                    event=RunEvent.workflow_completed
                )
                return
            
            print(f"  Processing {len(self.responses)} responses...")
            
            # Initialize results list
            self.analysis_results = []
            
            # Step 1: Categorize responses by demographics
            print("  Categorizing responses by age...")
            age_groups = self.categorize_by_age(self.responses)
            print("  Categorizing responses by gender...")
            gender_groups = self.categorize_by_gender(self.responses)
            print("  Categorizing responses by location...")
            location_groups = self.categorize_by_location(self.responses)
            
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
            
            # Step 3: Extract all questions, but DO NOT analyze them automatically
            print("  Extracting questions from responses...")
            questions = {}
            
            # First try to get questions from formData
            for response in self.responses:
                if 'formData' in response and isinstance(response['formData'], dict):
                    for question_id, answer in response['formData'].items():
                        if question_id not in questions:
                            questions[question_id] = {"count": 0, "answers": set()}
                        questions[question_id]["count"] += 1
                        questions[question_id]["answers"].add(str(answer))
            
            print(f"  Found {len(questions)} unique questions in responses")
            
            # Generate final insights
            print("  Generating final insights...")
            final_insights = self.generate_final_insights()
            
            # Prepare results
            result_summary = {
                "total_responses": len(self.responses),
                "questions": sorted(list(questions.keys())),  # Include the list of questions
                "demographic_analysis": {
                    "age": self.analysis_results[0].segments if len(self.analysis_results) > 0 else {},
                    "gender": self.analysis_results[1].segments if len(self.analysis_results) > 1 else {},
                    "location": self.analysis_results[2].segments if len(self.analysis_results) > 2 else {}
                },
                "question_analysis_count": 0,  # Start with 0 since we're doing on-demand analysis
                "final_insights": final_insights
            }
            
            # Ensure we're returning valid JSON
            yield RunResponse(
                content=json.dumps(result_summary, indent=2),
                event=RunEvent.workflow_completed
            )
        except Exception as e:
            # Return an error message as valid JSON
            error_message = str(e)
            print(f"Error in workflow: {error_message}")
            yield RunResponse(
                content=json.dumps({
                    "error": f"Analysis workflow failed: {error_message}",
                    "total_responses": 0,
                    "question_analysis_count": 0,
                    "demographic_analysis": {
                        "age": {},
                        "gender": {},
                        "location": {}
                    },
                    "final_insights": f"Analysis failed with error: {error_message}"
                }),
                event=RunEvent.workflow_completed
            )

# Function to run the polls analysis
def run_polls_analysis(api_endpoint: str):
    """Function to run the polls analysis workflow"""
    # Print header
    print("\n" + "="*80)
    print(f"POLLS ANALYSIS WORKFLOW - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    workflow = PollsAnalysisWorkflow(
        api_endpoint=api_endpoint,
        description="Polls Analysis Workflow",
        session_id=f"polls-analysis-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        storage=SqliteWorkflowStorage(
            table_name="polls_analysis_workflow",
            db_file="tmp/workflows.db",
        ),
        debug_mode=True,
    )
    
    print("\n🔄 Starting workflow execution...")
    
    # Run the workflow and get the results
    results = list(workflow.run())
    
    # Print the results
    print("\n📊 Analysis Results:")
    for i, result in enumerate(results, 1):
        print(f"  Result {i}/{len(results)}:")
        # Pretty print the JSON
        try:
            result_json = json.loads(result.content)
            print(json.dumps(result_json, indent=2))
        except json.JSONDecodeError:
            print(result.content)
    
    print("\n" + "="*80)
    print(f"WORKFLOW COMPLETED - Generated {len(results)} results")
    print("="*80 + "\n")
    
    return results

# Add this at the end of the file
if __name__ == "__main__":
    # Define the API endpoint where questionnaire data will be retrieved
    QUESTIONNAIRE_API_ENDPOINT = "https://inc-citizen.cabinex.co.ke/api/v1/sub_module_data/16"
    
    # Run the polls analysis workflow
    run_polls_analysis(QUESTIONNAIRE_API_ENDPOINT)