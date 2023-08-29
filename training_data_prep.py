import spacy
from spacy.tokens import DocBin
import pandas as pd
import re
# pd.set_option('display.max_colwidth', -1)

def massage_data(address):
    '''Pre process address string to remove new line characters, add comma punctuations etc.'''
    cleansed_address = re.sub(r'(,)(?!\s)', ', ', address)  # Add comma after comma without space
    cleansed_address = re.sub(r'(\\n)', ', ', cleansed_address)  # Replace newline characters with comma and space
    cleansed_address = re.sub(r'(?!\s)(-)(?!\s)', ' - ', cleansed_address)  # Add space around hyphens
    cleansed_address = re.sub(r'\.', '', cleansed_address)  # Remove periods
    cleansed_address = re.sub(r'\'', '', cleansed_address)  # Remove single quotes
    cleansed_address = re.sub(r'[\(\)]', '', cleansed_address)  # Remove parentheses
    print("After cleanse address: " + cleansed_address)
    return cleansed_address

def get_address_span(address=None,address_component=None,label=None):
    '''Search for specified address component and get the span.
    Eg: get_address_span(address="221 B, Baker Street, London",address_component="221",label="BUILDING_NO") would return (0,2,"BUILDING_NO")'''

    if pd.isna(address_component) or str(address_component)=='nan':
        pass
    else:
        # print(f'- full address - {address}')
        address_component = re.sub(r'(?!\s)(-)(?!\s)',' - ', address_component) # Put space before and after the hyphen
        address_component = re.sub(r'\.', '', address_component)  # Remove periods from address_component
        address_component = re.sub(r'\'', '', address_component)  # Remove single quotes from address_component
        address_component = re.sub(r'[\(\)]', '', address_component)  # Remove parentheses from address_component
        # print(address_component3)
        if label == 'CITY' or label == 'ADDITION':
            print(f'looking for {address_component} into {address}')
            pattern = rf'\b(?:{address_component})\b'
            matches = re.finditer(pattern, address)
            span = None
            for match in matches:
                print(f'match found: {match}')
                span = match
            print(f'city or addition - found: {span}')
        else:
            print(f'looking for {address_component} into {address}')
            span = re.search('\\b(?:' + address_component + ')\\b', address)
            print(f'else - found: {span}')
        return (span.start(),span.end(),label)

def extend_list(entity_list,entity):
    if pd.isna(entity):
        return entity_list
    else:
        entity_list.append(entity)
        return entity_list

def create_entity_spans(df,tag_list):
    '''Create entity spans for training/test datasets'''
    df['full_address']=df['full_address'].apply(lambda x: massage_data(x))
    # df['full_address']=df['full_address']
    df["straatTag"]=df.apply(lambda row:get_address_span(address=row['full_address'],address_component=row['straat'],label='STREET'),axis=1)
    df["huisnummerTag"]=df.apply(lambda row:get_address_span(address=row['full_address'],address_component=row['huisnummer'],label='HOUSE_NUMBER'),axis=1)
    df["huisletterTag"]=df.apply(lambda row:get_address_span(address=row['full_address'],address_component=row['huisletter'],label='HOUSE_LETTER'),axis=1)
    df["huistoevoegingTag"]=df.apply(lambda row:get_address_span(address=row['full_address'],address_component=row['huistoevoeging'],label='ADDITION'),axis=1)
    df["postcodeTag"]=df.apply(lambda row:get_address_span(address=row['full_address'],address_component=row['postcode'],label='POSTAL_CODE'),axis=1)
    df["cityTag"]=df.apply(lambda row:get_address_span(address=row['full_address'],address_component=row['woonplaats'],label='CITY'),axis=1)
    df['EmptySpan']=df.apply(lambda x: [], axis=1)

    for i in tag_list:
        df['EntitySpans']=df.apply(lambda row: extend_list(row['EmptySpan'],row[i]),axis=1)
        df['EntitySpans']=df[['EntitySpans','full_address']].apply(lambda x: (x[1], x[0]),axis=1)
    return df['EntitySpans']

def get_doc_bin(training_data,nlp):
    '''Create DocBin object for building training/test corpus'''
    # the DocBin will store the example documents
    db = DocBin()
    for text, annotations in training_data:
        doc = nlp(text) #Construct a Doc object
        ents = []
        print(f'{text} - Annotations: {annotations}')
        for start, end, label in annotations:
            span = doc.char_span(start, end, label=label)
            print(f'span: {span}')
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db

#Load blank Dutch model. This is needed for initializing a Document object for our training/test set.
nlp = spacy.blank("nl")

#Define custom entity tag list
tag_list=["straatTag","huisnummerTag","huisletterTag","huistoevoegingTag","postcodeTag","cityTag"]

###### Training dataset prep ###########
# Read the training dataset into pandas
# df_train=pd.read_csv(filepath_or_buffer="./corpus/dataset/us-train-dataset.csv",sep=",",dtype=str)
df_train = pd.read_csv(r'corpus/dataset/distinct_addresses.csv',sep=",",dtype=str)
df_train = df_train.sample(n=300, replace = False)
# print(df_train)

# Get entity spans
df_entity_spans= create_entity_spans(df_train.astype(str),tag_list)
training_data= df_entity_spans.values.tolist()

# Get & Persist DocBin to disk
doc_bin_train= get_doc_bin(training_data,nlp)
doc_bin_train.to_disk("./corpus/spacy-docbins/train.spacy")
######################################


###### Validation dataset prep ###########
# Read the validation dataset into pandas
# df_test=pd.read_csv(filepath_or_buffer="./corpus/dataset/us-test-dataset.csv",sep=",",dtype=str)
df_test = pd.read_csv(r'corpus/dataset/distinct_addresses.csv',sep=",",dtype=str)
df_test = df_test.sample(n=50, replace = False)
# print(df_test)

# Get entity spans
df_entity_spans= create_entity_spans(df_test.astype(str),tag_list)
validation_data= df_entity_spans.values.tolist()

# Get & Persist DocBin to disk
doc_bin_test= get_doc_bin(validation_data,nlp)
doc_bin_test.to_disk("./corpus/spacy-docbins/test.spacy")
##########################################