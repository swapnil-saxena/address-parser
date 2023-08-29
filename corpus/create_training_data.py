import pandas as pd
import random

df = pd.read_csv(r'corpus/dataset/pcodes.csv', sep=',', dtype=str)
df.fillna('', inplace=True)
samples = df.sample(n=500, replace = False)
# print(f'before {samples}')
for idx, sample in samples.iterrows():
    # print(sample)
    separate_format = random.choice([True, False])

    # Customize the postcode format
    if sample['postcode'] == '':
        postcode = ''
    else:
        postcode = sample['postcode']
        if separate_format:
            postcode = f"{postcode[:4]} {postcode[4:]}"
        else:
            postcode = f"{postcode[:4]}{postcode[4:]}"
    # Create full_address with customized postcode format
    full_address = f"{sample['straat']} {sample['huisnummer']} {sample['huisletter']} {sample['huistoevoeging']} {postcode} {sample['woonplaats']}"
    
    samples.loc[idx, 'postcode'] = postcode
    samples.loc[idx, "full_address"] = full_address
# print(f'sample + new col: {samples}')
samples.to_csv(r'corpus/dataset/distinct_addresses.csv', index=False)