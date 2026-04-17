import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch 

def train_validation_split(site_1,site_2,site_3,training_end,validate_start,validate_end,filetype):
    #get years column, tehy are the same but still doing it separately for good practice 
    site_1['year'] = pd.to_datetime(site_1['Date']).dt.year
    site_2['year'] = pd.to_datetime(site_2['Date']).dt.year
    site_3['year'] = pd.to_datetime(site_3['Date']).dt.year

    #produce traning/validation set. 
    #History of split: doing 2016-2022 then 2023-2025 has 30% validation, and doing 2016-2023 then 2024-2025 has 20% validation. I felt more training data was better, so this is 80/20 
    
    #---------Site 1-------------
    t1 = site_1[site_1['year']<= training_end].copy()
    v1 = site_1[(site_1['year'] >= validate_start) & (site_1['year'] <=validate_end)].copy()
    #save just in case
    t1.to_csv(f'./LSTM Files/Train-Validate/{filetype}_1_training.csv', index=False)
    v1.to_csv(f'./LSTM Files/Train-Validate/{filetype}_1_validation.csv', index=False)

    #---------Site 2-------------                               testing set location falls between these
    t2 = site_2[site_2['year']<= training_end].copy()
    v2 = site_2[(site_2['year'] >= validate_start) & (site_2['year'] <=validate_end)].copy()
    #save just in case
    t2.to_csv(f'./LSTM Files/Train-Validate/{filetype}_2_training.csv', index=False)
    v2.to_csv(f'./LSTM Files/Train-Validate/{filetype}_2_validation.csv', index=False)

    #---------Site 3-------------
    t3 = site_3[site_3['year']<= training_end].copy()
    v3 = site_3[(site_3['year'] >= validate_start) & (site_3['year'] <=validate_end)].copy()
    #save just in case
    t3.to_csv(f'./LSTM Files/Train-Validate/{filetype}_3_training.csv', index=False)
    v3.to_csv(f'./LSTM Files/Train-Validate/{filetype}_3_validation.csv', index=False)    

    #set all dataframes to only have parameters (drop the dates/years)
    sets = [t1,t2,t3,v1,v2,v3]
    sets = [df.drop(columns=["Date","year"]).reset_index(drop=True) for df in sets]
    t1,t2,t3,v1,v2,v3 = sets

    return t1,t2,t3,v1,v2,v3

def masking(df): #one gage site didn't have the same number of stations, now everything has to be masked 
    mask = ~np.isnan(df) #boolean map         
    df_filled = np.nan_to_num(df) #change NAN to zeros (NAN breaks LSTM models)
    return df_filled, mask.astype(np.float32)

def minmax(t1,t2,t3,v1,v2,v3,parameter,): #normalize the data with MinMaxScalar in scikit
    minmax = MinMaxScaler()

    if parameter == 'x': #if there is x-data, aka inputs
        normalize = np.vstack([t1, t2, t3])
        minmax.fit(normalize)
        #training
        t1 = minmax.transform(t1)
        t2 = minmax.transform(t2)
        t3 = minmax.transform(t3)
        #validation
        v1 = minmax.transform(v1)
        v2 = minmax.transform(v2)
        v3 = minmax.transform(v3)

    if parameter == 'y': #if there is y-data, aka targets
        normalize = np.concatenate([np.array(t1),np.array(t2),np.array(t3)]).reshape(-1, 1) #the data sets have to be in 2D, so handled differently than X
        minmax.fit(normalize)
        #training
        t1 = minmax.transform(np.array(t1).reshape(-1, 1))
        t2 = minmax.transform(np.array(t2).reshape(-1, 1))
        t3 = minmax.transform(np.array(t3).reshape(-1, 1))
        #validation
        v1 = minmax.transform(np.array(v1).reshape(-1, 1))
        v2 = minmax.transform(np.array(v2).reshape(-1, 1))
        v3 = minmax.transform(np.array(v3).reshape(-1, 1))

    return t1,t2,t3,v1,v2,v3,minmax

def sequencing(x,mask,y):
    step = 30       #for 30-day analysis
    xset, yset,maskset = [],[],[]       #for tracking

    mask = np.asarray(mask) #only item I forgot to make an array earlier, could fix later if there is time? -no

    #This is the sequencing method we did in Machine learning with masking, felt more appropriate for my dataset 
    for i in range(len(x) - step):
        xset.append(x[i:i+step])
        maskset.append(mask[i:i+step])
        yset.append(y[i+step])

    #ordered set with (input, mask, target)
    xset = torch.tensor(xset, dtype=torch.float32)
    maskset = torch.tensor(maskset, dtype=torch.float32)
    yset = torch.tensor(yset, dtype=torch.float32)
    
    return xset,maskset,yset

def build_full_sets(x_list, mask_list,y_list): #combine data for training 
    X,M,Y = [],[],[] #input, mask, target

    for i in range(len(x_list)): #make a set of X's, M's and Y's, as they are fed differently into the model 
        x,m,y = sequencing(x_list[i],mask_list[i],y_list[i])
        X.append(x)
        M.append(m)
        Y.append(y)

    #combine all sites into one input, mask, and target sets (main document has validation/training separated still)
    Input = torch.cat(X,dim=0)
    Mask = torch.cat(M,dim=0)
    Target = torch.cat(Y,dim=0)

    return Input,Mask,Target


