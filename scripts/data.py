import os
import sys
import pandas as pd
import datetime as dt
from dataretrieval import nwis
from dataretrieval import waterdata
import numpy as np
import requests
import io

def discharge(ID, start, end): #getting discharge data, utilized previously defined "Helper" functions 
    url = "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items"
    params = {
        "f": "json",
        "monitoring_location_id": f'USGS-{ID}',
        "parameter_code": "00060",
        "statistic_id": "00003",
        "time": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 10000,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    payload = r.json()
    rows = []
    for feat in payload.get("features", []):
        props = feat.get("properties", {})
        rows.append({
            "Date": pd.to_datetime(props.get("time")).normalize(),
            "Discharge_cfs": pd.to_numeric(props.get("value"), errors="coerce"),
        })
    df = pd.DataFrame(rows).sort_values("Date")
    
    #fill missing values with interpolate, it was easier to do here
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    range1 = pd.date_range(start=start,end=end,freq="D")
    df = df.reindex(range1)
    df.index.name = 'Date'
    df = df.reset_index()
    df = df.interpolate(method='linear', limit_direction='both').ffill().bfill()
    df.to_csv(f'./Data Files/Streamflow/{ID}_discharge.csv', index=True)
    return df

def daymet(ID,lat, lon, years): #get rain and temp max/min, same source as streamflow
    url = "https://daymet.ornl.gov/single-pixel/api/data"
    params = {
        "lat": lat,
        "lon": lon,
        "vars": "prcp,tmax,tmin",
        "years":years
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    text = r.text.splitlines()
    start_idx = None
    for i, line in enumerate(text):
        if line.lower().startswith("year,"):
            start_idx = i
            break
    if start_idx is None:
        data = pd.read_csv(io.StringIO(r.text))
    else:
        data = pd.read_csv(io.StringIO("\n".join(text[start_idx:])))
    data.to_csv(f'./Data Files/Rain_Temp/{ID}_daymet.csv', index=False) #save (not normalized though)
    #note, because the model is training on patterns and not actually using physics informed losses, units were not changed 
    return data

def normalize_daymet(df): #fit for years, so it can be aligned with SWE later
    cols = {c.lower(): c for c in df.columns}
    out = df.copy()
    out["Date"] = pd.to_datetime(out[cols["year"]].astype(int).astype(str)) + pd.to_timedelta(out[cols["yday"]].astype(int) - 1, unit="D")
    rename_map = {}
    for key in ["prcp (mm/day)", "tmax (deg c)", "tmin (deg c)"]:
        if key in cols:
            rename_map[cols[key]] = key
    out = out.rename(columns=rename_map)
    keep = ["Date"] + [c for c in ["prcp (mm/day)", "tmax (deg c)", "tmin (deg c)"] if c in out.columns]
    
    return out[keep].sort_values("Date").reset_index(drop=True)

def swe_set(stations,ID): 
    #Get swe, the CSVs were directly downloaded from my Homework2 functions, there were too many to include in this homework so it is linked on main page
    full_set = pd.DataFrame()
    for site in stations:
        df = pd.read_csv(f'./Data Files/SWE/df_{site}_UT_SNTL_UT_SNTL.csv',usecols=['Date','Snow Water Equivalent (m) Start of Day Values'])
        df = df.rename(columns={"Snow Water Equivalent (m) Start of Day Values":f'{site}_SWE (m)'})
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index('Date',inplace=True)
        full_set = pd.concat([full_set,df],join = 'outer',axis=1)
    full_set.to_csv(f'./Data Files/SWE/{ID}_full_SWE.csv', index=False)#save
    return full_set #all stations together

def full_data(daymet, swe, ID): #combine for full parameters datafram to use in training
    daymet['Date'] = pd.to_datetime(daymet['Date'])
    daymet.set_index('Date', inplace=True)

    training = pd.concat([daymet,swe],join='outer',ignore_index=False,axis=1)
    training = training.reset_index(drop=False)
    training.to_csv(f'./Data Files/{ID}_combined.csv', index=True)#save 
    return training
