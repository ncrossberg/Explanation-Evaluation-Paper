import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as plt
import regex as re

#Import dataframe
df = pd.read_csv('New User Study UCC_August 31, 2026_17.03.csv')

df = df.drop(['StartDate', 'EndDate', 'Status', 'IPAddress', 'RecipientLastName', 'RecipientFirstName',
            'RecipientEmail', 'ExternalReference', 'LocationLatitude'], axis = 1)

df = df.drop(df[df['Consent'] == 'No'].index)
df = df.drop(df[df['Q31'] == 'A2 - Pre-Intermediate'].index)
df = df.drop(df[df['Q31'] == 'A1 - Elementary'].index)
#Lose one row here due to low English Level

#Generate Attention Check Dictionary Here
#Isolate Columns ending in _10 and first row
attention_columns = df.loc[:, df.iloc[0].astype(str).str.contains('Attention')].columns.tolist()
df_attention = df[attention_columns]

#Generate dictionary with preferred response for each AC
import re

pattern = r"Please select the '([^']+)'"

attention_dict = {}
for col in df_attention.columns:
    text = df_attention.loc[0, col]
    match = re.search(pattern, str(text))
    if match:
        attention_dict[col] = match.group(1)
    else:
        attention_dict[col] = None  # or some placeholder if no match found

attention_dict = {re.sub(r'_\d+$', 'AC', k): v for k, v in attention_dict.items()}

#Contains the Features and their corresponding attention check
attention_dict

#Drop Indexing Rows
df = df.drop([0, 1], axis = 0)

#Check Number of remaining responses
df.shape #119 responses total eligible so far

#Now check response times
df['Duration (in seconds)'] = pd.to_numeric(df['Duration (in seconds)'], errors='coerce')
df['Duration (in seconds)'].describe()/60


#Remove unnecessary columns
df = df.drop(['Progress', 'Duration (in seconds)',
            'Finished', 'RecordedDate', 'ResponseId',
            'LocationLongitude', 'DistributionChannel', 'UserLanguage',
            'Last Seen Flow Element ID', 'Last Seen Question IDs', 'Consent', 'Q31', 'Q79'], axis = 1)

#First break dataframe into row and column with one row per feature / participant making sure to maintain participant ID
column_list = list(set([c[:-4] for c in df.columns.tolist()]))

#Create participant column
df['Participant'] = df.index

#Rename Attention Check Columns
rename_map = {col: re.sub(r'_\d+$', 'AC', col) for col in attention_columns}
df = df.rename(columns=rename_map)

#Dictionary for re-coding
value_dict = {'Strongly Disagree' : 1, 'Disagree' : 2, 'Neutral' : 3, 'Agree' : 4, 'Strongly Agree' : 5}

final_frame = pd.DataFrame(columns=['Item_1', 'Item_2', 'Item_3', 'Item_4', 'Item_5', 'Item_6', 'Item_7', 'Item_8', 'Item_9', 'Attention Check', 'Participant', 'Feature', 'Dataset', 'Factor 1', 'Factor 2', 'Total Score', 'Average Score'])
final_frame

df_working = df.copy().drop('Participant', axis = 1)

for idx in range(43):
    #Identify Subset
    ac_idx = [i for i, c in enumerate(df_working.columns) if c.endswith('AC')][0]
    
    #Cut subset
    sub_frame = df_working.iloc[:, :ac_idx+1]
    sub_frame['Participant'] = df['Participant']
    sub_columns = sub_frame.columns

    #Drop subset from dataframe
    df_working.drop(sub_frame.columns.tolist()[:-1], axis = 1, inplace=True)
    
    #Drop Rows that failed attention check or are empty
    attention = attention_dict[str(sub_columns[-2])]
    sub_frame = sub_frame[sub_frame[str(sub_columns[-2])] == attention]
    
    sub_frame.replace(value_dict, inplace=True)
    
    #Add Feature Column
    sub_frame['Feature'] = [sub_columns[-2][:-3] for x in range(len(sub_frame))]
    
    #Add Dataset Column
    sub_frame['Dataset'] = [sub_columns[-2][-3:-2] for x in range(len(sub_frame))]
    
    #Rename Items
    ac_idx = [i for i, c in enumerate(sub_frame.columns) if c.endswith('AC')][0]

    sub_frame.columns = [f'Item_{i+1}' for i in range(ac_idx)] + ['Attention Check'] + sub_frame.columns[ac_idx+1:].tolist()
    
    #Add Factor Variable
    if ac_idx == 9:
        sub_frame['Factor 1'] = sub_frame.loc[:, 'Item_1':'Item_6'].sum(axis=1)
        sub_frame['Factor 2'] = sub_frame.loc[:, 'Item_7':'Item_9'].sum(axis=1)
        sub_frame['Total Score'] = sub_frame.loc[:, 'Item_1':'Item_9'].sum(axis=1)
        sub_frame['Average Score'] = [x/ac_idx for x in sub_frame['Total Score']]
    elif ac_idx == 8: #Need to check that this is working; also need to add empty Item 9 column to make sure dataframe stacks
        sub_frame['Factor 1'] = sub_frame.loc[:, 'Item_1':'Item_5'].sum(axis=1)
        sub_frame['Factor 2'] = sub_frame.loc[:, 'Item_5':'Item_8'].sum(axis=1)
        sub_frame['Total Score'] = sub_frame.loc[:, 'Item_1':'Item_8'].sum(axis=1)
        sub_frame['Average Score'] = [x/ac_idx for x in sub_frame['Total Score']]
        
        #Add Item_9 column and fill with NA to allow stacking
        sub_frame.insert(8, 'Item_9', [np.nan for x in range(len(sub_frame))])
    
    final_frame = pd.concat([sub_frame, final_frame], axis=0)

final_frame.to_csv('Understandability_Scores_Processed.csv')

final_frame = pd.read_csv('Understandability_Scores_Processed.csv')

final_frame.groupby(['Dataset', 'Feature'])['Total Score'].describe()