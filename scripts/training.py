import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch 

def train_validation_split(site_1,site_2,site_3,training_end,validate_start,validate_end,filetype):
    site_1['year'] = pd.to_datetime(site_1['Date']).dt.year
    site_2['year'] = pd.to_datetime(site_2['Date']).dt.year
    site_3['year'] = pd.to_datetime(site_3['Date']).dt.year

    t1 = site_1[site_1['year']<= training_end].copy()
    v1 = site_1[(site_1['year'] >= validate_start) & (site_1['year'] <=validate_end)].copy()
    #save just in case
    t1.to_csv(f'./LSTM Files/Train-Validate/{filetype}_1_training.csv', index=False)
    v1.to_csv(f'./LSTM Files/Train-Validate/{filetype}_1_validation.csv', index=False)

    t2 = site_2[site_2['year']<= training_end].copy()
    v2 = site_2[(site_2['year'] >= validate_start) & (site_2['year'] <=validate_end)].copy()
    #save just in case
    t2.to_csv(f'./LSTM Files/Train-Validate/{filetype}_2_training.csv', index=False)
    v2.to_csv(f'./LSTM Files/Train-Validate/{filetype}_2_validation.csv', index=False)


    t3 = site_3[site_3['year']<= training_end].copy()
    v3 = site_3[(site_3['year'] >= validate_start) & (site_3['year'] <=validate_end)].copy()
    #save just in case
    t3.to_csv(f'./LSTM Files/Train-Validate/{filetype}_3_training.csv', index=False)
    v3.to_csv(f'./LSTM Files/Train-Validate/{filetype}_3_validation.csv', index=False)

    print(f"Training Rows: Site 1 - {len(t1)}, Site 2 = {len(t2)}, Site 3 - {len(t3)}")
    print(f"Validation Rows: Site 1 - {len(v1)}, Site 2 = {len(v2)}, Site 3 - {len(v3)}")
   
    if (len({len(t1), len(t2), len(t3)}) == 1) and (len({len(v1), len(v2), len(v3)}) == 1):
        print(f'Percent Validation: {(len(v1)/(len(t1)+len(v1)) * 100):.0f}%')    

    #set all dataframes to only have parameters
    sets = [t1,t2,t3,v1,v2,v3]
    sets = [df.drop(columns=["Date","year"]).reset_index(drop=True) for df in sets]

    t1,t2,t3,v1,v2,v3 = sets

    return t1,t2,t3,v1,v2,v3

def masking(df):
    mask = ~np.isnan(df) #boolean map         
    df_filled = np.nan_to_num(df) #change NAN to zeros (NAN breaks LSTM models)

    return df_filled, mask.astype(np.float32)

def minmax(t1,t2,t3,v1,v2,v3,parameter):
    minmax = MinMaxScaler()

    if parameter == 'x':
        normalize = np.vstack([t1, t2, t3])
        minmax.fit(normalize)

        t1 = minmax.transform(t1)
        t2 = minmax.transform(t2)
        t3 = minmax.transform(t3)

        v1 = minmax.transform(v1)
        v2 = minmax.transform(v2)
        v3 = minmax.transform(v3)

    if parameter == 'y':
        normalize = np.concatenate([np.array(t1),np.array(t2),np.array(t3)]).reshape(-1, 1)
        minmax.fit(normalize)

        t1 = minmax.transform(np.array(t1).reshape(-1, 1))
        t2 = minmax.transform(np.array(t2).reshape(-1, 1))
        t3 = minmax.transform(np.array(t3).reshape(-1, 1))

        v1 = minmax.transform(np.array(v1).reshape(-1, 1))
        v2 = minmax.transform(np.array(v2).reshape(-1, 1))
        v3 = minmax.transform(np.array(v3).reshape(-1, 1))

    return t1,t2,t3,v1,v2,v3

def sequencing(x,mask,y):
    step = 30 #for 30-day analysis
    xset, yset,maskset = [],[],[]

    mask = np.asarray(mask) #only item I forgot to make an array earlier

    for i in range(len(x) - step):
        xset.append(x[i:i+step])
        maskset.append(mask[i:i+step])
        yset.append(y[i+step])

    xset = torch.tensor(xset, dtype=torch.float32)
    maskset = torch.tensor(maskset, dtype=torch.float32)
    yset = torch.tensor(yset, dtype=torch.float32)
    
    return xset,maskset,yset

def build_full_sets(x_list, mask_list,y_list):
    X,M,Y = [],[],[]
    for i in range(len(x_list)):
        x,m,y = sequencing(x_list[i],mask_list[i],y_list[i])
        X.append(x)
        M.append(m)
        Y.append(y)

    Input = torch.cat(X,dim=0)
    Mask = torch.cat(M,dim=0)
    Target = torch.cat(Y,dim=0)

    return Input,Mask,Target


