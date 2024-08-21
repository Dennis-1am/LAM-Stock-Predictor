from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from stock import Stock
import json
import requests
import os


app = FastAPI()
stock_instance = Stock()

@app.get("/stock/")
async def root():
    return {"message": "Hello World"}

@app.get("/stock/{ticker}")
async def stock(ticker: str):
    try:
        data = stock_instance.get_stock_data(ticker, 'max')
        data_json = jsonable_encoder(data.to_dict())
        return JSONResponse(content=data_json)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))