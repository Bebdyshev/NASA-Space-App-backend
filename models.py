from pydantic import BaseModel, Field
from bson import ObjectId
from typing import Optional, Any

class UserInDB(BaseModel):
    id: Optional[str] = Field(default_factory=str)
    company_name: str
    email: str
    hashed_password: str

class CandidateInDB(BaseModel):
    id: Optional[str] = Field(default_factory=str)
    user_id: str
    data: Any

class UserCreate(BaseModel):
    company_name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str