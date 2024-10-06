from pydantic import BaseModel, Field
from bson import ObjectId
from typing import Optional, Any

class UserInDB(BaseModel):
    id: Optional[str] = Field(default_factory=str)
    company_name: str
    email: str
    hashed_password: str

class FieldInDB(BaseModel):
    id: Optional[str] = Field(default_factory=str)
    user_id: str
    bbox: str
    field_name: str

class FieldCreate(BaseModel):
    bbox: str
    field_name: str

class UserCreate(BaseModel):
    company_name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str

class PlantRequest(BaseModel):
    url: str