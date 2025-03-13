from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union, Any
from datetime import datetime

class User(BaseModel):
    id: int
    name: str
    email: str
    telephone: str
    is_adult: str
    password: str
    iPRS_PersonId: Optional[Union[str, int]] = None
    profile_pic: Optional[str] = None
    locationId: Optional[int] = None
    created_at: datetime
    iprs: Optional[Any] = None

class QuestionDependency(BaseModel):
    value: str
    question: str

class QuestionField(BaseModel):
    list: List[Any] = []
    name: str
    type: str = Field(..., description="Type of question (single-choice, multi-select, text, etc.)")
    options: List[str] = []
    dependency: QuestionDependency
    input_fields: List[Any] = []

class SubModule(BaseModel):
    id: int
    name: str
    description: str
    bf: bool
    repetition: bool
    fields: List[QuestionField]
    createdAt: datetime
    modulesId: int

class Response(BaseModel):
    id: int
    sub_moduleId: int
    submissionDate: datetime
    attachments: List[Any] = []
    formData: Dict[str, Union[str, List[str]]]
    userId: int
    user: User
    sub_module: SubModule

class SubModuleResponse(BaseModel):
    """Model representing the actual API response structure"""
    message: str
    data: Dict[str, Any]  # Changed from List[Dict] to Dict to match actual API structure

class QuestionnaireData(BaseModel):
    """Model for processed questionnaire data"""
    message: str
    data: List[Response]  # This is for after you've processed the raw API data