from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId
from auth_utils import hash_password, verify_password, create_access_token, verify_access_token
from models import *
from config import get_db
from datetime import timedelta
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

@app.get("/")
async def root():
    return {"message": "Hello world!"}

@app.post("/register/", response_model=dict)
async def register_user(user: UserCreate, db = Depends(get_db)) -> dict:
    user_collection = db["users"]
    existing_user = await user_collection.find_one({"email": user.email})
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = hash_password(user.password)
    new_user = UserInDB(
        company_name=user.company_name,
        email=user.email,
        hashed_password=hashed_pw
    )
    await user_collection.insert_one(new_user.dict())
    
    return {"msg": "User created successfully"}

@app.post("/login/", response_model=Token)
async def login_for_access_token(user: UserLogin, db = Depends(get_db)) -> Token:
    user_collection = db["users"]
    db_user = await user_collection.find_one({"email": user.email})
    
    if not db_user or not verify_password(user.password, db_user['hashed_password']):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/map/satellite_photo")
async def get_satellite_photo(
    token: str = Depends(oauth2_scheme), 
    db = Depends(get_db),
    bbox: str = Body(...), 
    layers: str = Body(...), 
    time: str = Body(...)
    ):

    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    url = os.getenv("WMS_URL")

    return {
        "satellite_data": url+"?SERVICE=WMS&REQUEST=GetMap&BBOX="+bbox+"&LAYERS="+layers+"&FORMAT=image/jpeg&WIDTH=512&HEIGHT=512&CRS=EPSG:4326&TIME="+time
    }

@app.get("/users/me/")
async def read_users_me(token: str = Depends(oauth2_scheme), db = Depends(get_db)):
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    user_email = payload.get("sub")
    user_collection = db["users"]
    user = await user_collection.find_one({"email": user_email})
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user