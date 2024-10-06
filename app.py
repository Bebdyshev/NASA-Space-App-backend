from fastapi import FastAPI, Depends, HTTPException, Body, WebSocket, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId
from fastapi.responses import JSONResponse
from auth_utils import hash_password, verify_password, create_access_token, verify_access_token
from models import *
from config import get_db
from datetime import timedelta
import os
from dotenv import load_dotenv
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from partialjson.json_parser import JSONParser
from pinecone import Pinecone
import re
import requests
import base64
import datetime
import sklearn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


# Load environment variables
load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000"],  # Change this to the origin of your React app
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
client = OpenAI(api_key="sk-None-GzNN8dNxc0JTOVCjZ4fOT3BlbkFJQAee2MsEogfhOZL62Cpm")
pc = Pinecone(api_key="552fa38d-c865-4c95-a0af-a7c65f1302f0")
index = pc.Index("nasa")
parser = JSONParser(strict=False)

def url_to_base64(url):
    response = requests.get(url)
    if response.status_code == 200:
        image_data = base64.b64encode(response.content).decode('utf-8')
        return f"data:image/jpeg;base64,{image_data}"  # Change 'jpeg' to the correct image type if necessary
    else:
        raise ValueError("Could not retrieve image from URL")

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
    
    result = await user_collection.insert_one(new_user.dict())
    new_user_id = result.inserted_id
    
    await user_collection.update_one(
        {"_id": new_user_id}, 
        {"$set": {"id": str(new_user_id)}}
    )
    
    return {"msg": "User created successfully"}

@app.post("/login/", response_model=Token)
async def login_for_access_token(user: UserLogin, db = Depends(get_db)) -> Token:
    user_collection = db["users"]
    db_user = await user_collection.find_one({"email": user.email})
    
    if not db_user or not verify_password(user.password, db_user['hashed_password']):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.email, "user_id": str(db_user["_id"])}, expires_delta=access_token_expires
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

@app.post("/map/fields/", response_model=dict)
async def create_field(
    field: FieldCreate, 
    token: str = Depends(oauth2_scheme), 
    db = Depends(get_db)
) -> dict:
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_email = payload.get("sub")
    user_collection = db["users"]
    user = await user_collection.find_one({"email": user_email})
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    new_field = FieldInDB(
        user_id=str(user["_id"]),
        bbox=field.bbox,
        field_name=field.field_name
    )
    
    fields_collection = db["fields"]
    await fields_collection.insert_one(new_field.dict())
    
    return {"msg": "Field created successfully"}


@app.get("/map/fields/", response_model=List[FieldInDB])
async def get_fields_by_user(
    token: str = Depends(oauth2_scheme), 
    db = Depends(get_db)
) -> List[FieldInDB]:
    payload = verify_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user_id = str(payload.get("user_id"))

    fields_collection = db["fields"]

    fields = await fields_collection.find({"user_id": user_id}).to_list(length=None)
    
    if not fields:
        raise HTTPException(status_code=404, detail="No fields found for the user")

    return fields

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

# Plant identifier
@app.post("/plant")
async def plant_identifier(file: UploadFile = File(...)):
    image_data = await file.read()
    # Encode the image data to base64
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    # Prepare the message for the chat model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s plant/seed in this image? Return - type, disease, caringTutor as text, NOT AS JSON"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    completion = response.choices[0].message.content
    print(completion)
    return {completion}

# Price suggester
@app.post("/price")
async def price_suggester(product: str = Body(...)):
    response = client.embeddings.create(
        input=product,
        model="text-embedding-ada-002"
    )

    embedding = response.data[0].embedding
    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True,
    )

    prices = [extract_price(match['metadata']['price']) for match in results['matches']]
    if prices:
        avg_price = sum(prices) / len(prices)
        prompt = f"I am selling {product}. The average price of similar products is {avg_price}. Should I sell at a higher or lower price? What price do you recommend and why? In answer return current avg price, dont talk basics"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            max_tokens=100
        )
        
        return response.choices[0].message.content
    else:
        raise HTTPException(status_code=404, detail="No valid prices found.")
    
@app.websocket("/chatgpt")
async def chatgpt_stream(websocket: WebSocket):
    print("Connection started")
    await websocket.accept()
    
    try:
        while True:
            prompt = await websocket.receive_text()

            # Start GPT-4 completion with streaming enabled
            response_stream = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """
                        You are an AI assistant that generates a roadmap for agriculture tasks for a season.
                        Return only JSON in the following structure:
                        [
                            {
                                "title": "string",
                                "description": "string",
                                "timestamp": "string",
                                "childs": [
                                    {
                                        "title": "string",
                                        "description": "string",
                                        "timestamp": "string"
                                    }
                                ]
                            }
                        ].
                        The parent items should be broader tasks (e.g. "preparing area"), and the "childs" array should contain the steps involved in those tasks (e.g. "soil preparation", "fertilizing").
                        Do not return any text outside the JSON format.
                    """},
                    {"role": "user", "content": prompt}
                ],
                stream=True,
            )

            # Variables to track when JSON is complete
            res = ''
            open_braces = 0  # Count of open '{' braces
            close_braces = 0  # Count of closed '}' braces
            is_json_complete = False

            for chunk in response_stream:
                msg = chunk.choices[0].delta.content
                if msg:
                    res += msg
                    await websocket.send_json(parser.parse(res))
                    # open_braces += msg.count('{')
                    # close_braces += msg.count('}')

                    # # Check if the JSON structure is complete
                    # if open_braces == close_braces and open_braces > 0:
                    #     is_json_complete = True

                    # # Send chunks to client as they arrive
                    # await websocket.send_text(msg)

                    # # Parse and send the final JSON when complete
                    # if is_json_complete:
                    #     await websocket.send_json(parser.parse(res))
                    #     break  # Exit the loop once the JSON is complete
    except Exception as e:
        await websocket.close()
        print(f"Error: {e}")

# Utility function to extract price
def extract_price(price_str):
    match = re.search(r'(\d+)', price_str)
    if match:
        return int(match.group(1)) 
    return 0





