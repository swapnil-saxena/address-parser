import spacy
from spacy.tokens import DocBin
import pandas as pd
import re
pd.set_option('display.max_colwidth', -1)

def massage_data(address):
    '''Pre process address string to remove new line characters, add comma punctuations etc.'''
    cleansed_address1=re.sub(r'(,)(?!\s)',', ',address)
    cleansed_address2=re.sub(r'(\\n)',', ',cleansed_address1)
    cleansed_address3=re.sub(r'(?!\s)(-)(?!\s)',' - ',cleansed_address2)
    cleansed_address=re.sub(r'\.','',cleansed_address3)
    return cleansed_address

def get_address_span(address=None,address_component=None,label=None):
    '''Search for specified address component and get the span.
    Eg: get_address_span(address="221 B, Baker Street, London",address_component="221",label="BUILDING_NO") would return (0,2,"BUILDING_NO")'''

    if pd.isna(address_component) or str(address_component)=='nan':
        pass
    else:
        address_component1=re.sub('\.','',address_component)
        address_component2=re.sub(r'(?!\s)(-)(?!\s)',' - ',address_component1)
        span=re.search('\\b(?:'+address_component2+')\\b',address)
        #print(span)
        return (span.start(),span.end(),label)

def extend_list(entity_list,entity):
    if pd.isna(entity):
        return entity_list
    else:
        entity_list.append(entity)
        return entity_list

def create_entity_spans(df,tag_list):
    '''Create entity spans for training/test datasets'''
    df['Address']=df['Address'].apply(lambda x: massage_data(x))
    df["BuildingTag"]=df.apply(lambda row:get_address_span(address=row['Address'],address_component=row['Building_Name'],label='BUILDING_NAME'),axis=1)
    df["BuildingNoTag"]=df.apply(lambda row:get_address_span(address=row['Address'],address_component=row['Building_Number'],label='BUILDING_NO'),axis=1)
    df["RecipientTag"]=df.apply(lambda row:get_address_span(address=row['Address'],address_component=row['Recipient'],label='RECIPIENT'),axis=1)
    df["StreetNameTag"]=df.apply(lambda row:get_address_span(address=row['Address'],address_component=row['Street_Name'],label='STREET_NAME'),axis=1)
    df["ZipCodeTag"]=df.apply(lambda row:get_address_span(address=row['Address'],address_component=row['Zip_Code'],label='ZIP_CODE'),axis=1)
    df["CityTag"]=df.apply(lambda row:get_address_span(address=row['Address'],address_component=row['City'],label='CITY'),axis=1)
    df["CountryTag"]=df.apply(lambda row:get_address_span(address=row['Address'],address_component=row['Country'],label='COUNTRY'),axis=1)
    df["StateTag"]=df.apply(lambda row:get_address_span(address=row['Address'],address_component=row['State'],label='STATE'),axis=1)
    df['EmptySpan']=df.apply(lambda x: [], axis=1)

    for i in tag_list:
        df['EntitySpans']=df.apply(lambda row: extend_list(row['EmptySpan'],row[i]),axis=1)
        df['EntitySpans']=df[['EntitySpans','Address']].apply(lambda x: (x[1], x[0]),axis=1)
    return df['EntitySpans']

def get_doc_bin(training_data,nlp):
    '''Create DocBin object for building training/test corpus'''
    # the DocBin will store the example documents
    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text) #Construct a Doc object
        ents = []
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db

#Load blank English model. This is needed for initializing a Document object for our training/test set.
nlp = spacy.blank("en")

#Define custom entity tag list
tag_list=["BuildingTag","BuildingNoTag","RecipientTag","StreetNameTag","ZipCodeTag","CityTag","StateTag","CountryTag"]

###### Training dataset prep ###########
# Read the training dataset into pandas
df_train=pd.read_csv(filepath_or_buffer="./corpus/dataset/us-train-dataset.csv",sep=",",dtype=str)

# Get entity spans
df_entity_spans= create_entity_spans(df_train.astype(str),tag_list)
training_data= df_entity_spans.values.tolist()

# Get & Persist DocBin to disk
doc_bin_train= get_doc_bin(training_data,nlp)
doc_bin_train.to_disk("./corpus/spacy-docbins/train.spacy")
######################################


###### Validation dataset prep ###########
# Read the validation dataset into pandas
df_test=pd.read_csv(filepath_or_buffer="./corpus/dataset/us-test-dataset.csv",sep=",",dtype=str)

# Get entity spans
df_entity_spans= create_entity_spans(df_test.astype(str),tag_list)
validation_data= df_entity_spans.values.tolist()

# Get & Persist DocBin to disk
doc_bin_test= get_doc_bin(validation_data,nlp)
doc_bin_test.to_disk("./corpus/spacy-docbins/test.spacy")
##########################################