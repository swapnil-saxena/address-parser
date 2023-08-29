import spacy

nlp=spacy.load("output/models/model-best")

address_list = [
    "Pieter de Hoochstraat 9 A 1071ED Amsterdam",
    "Googweg 8 A 1399ER Muiderberg",
    "Bloemenhof 51 1906XV Limmen",
    "Prinses Alexialaan 5 * 2135VH Hoofddorp",
    "Kempen 57 2036EK Haarlem",
    "Aurikelstraat 55 1032AR Amsterdam",
    "Prattenburg 403 2036SM Haarlem",
    "Nieuwstraat 80 . 1381XV Weesp",
    "Vissersdijk 52 1601LR Enkhuizen",
    "Kanaalstraat 61 RD. 1975BB IJmuiden",
    "Goudplevier 21 * 1191VP Ouderkerk aan de Amstel",
    "J. Bijhouwersstraat 71 1992JV Velserbroek",
    "Sint-Jorisveld 11 2023GD Haarlem",
    "Krijn Taconiskade 469 1087HW Amsterdam 1000€",
    "Amsteldijk 59 D 1074HX Amsterdam",
    "Weldam 8 1081HN Amsterdam",
    "Peltenburgstraat 53 2033ES Haarlem",
    "Vierwindenstraat 78 1 1013LA Amsterdam",
    "Revaleiland 442 + PP 1014ZG Amsterdam",
    "'s-Gravelandseweg 6 1381HH Weesp",
    "Bella Vistastraat 298 1096GM Amsterdam",
    "IJdok 109 1013 MM Amsterdam",
    "Huigsloterdijk 381 huur 2158 LR Buitenkaag",
    "Cattenhagestraat 21 1411 CR Naarden",
    "Balboaplein 84 1057 VS Amsterdam",
    "Stroet 115 1744 GM Sint Maarten",
    "Kolksteeg 3 2 1012 PT Amsterdam",
    "Bijlmerplein 858 S7 1102 ME Amsterdam",
    "Bijlmerplein 858 U6 1102 ME Amsterdam",
    "Laurens Reaellaan 8 2 2024 BE Haarlem",
    "Witte Herenstraat 25 2011 NT Haarlem",
    "Pieter Goosstraat 10 1018 LA Amsterdam",
    "Herengracht 14 1382 AE Weesp",
    "Struikheidelaan 69 1213 WZ Hilversum",
    "Prins Hendriklaan 70 1261AJ Blaricum",
    "Prins Hendriklaan 74 1261 AJ Blaricum",
    "Prins Hendriklaan 64 1261 AJ Blaricum",
    "Prins Hendriklaan 78 1261 AJ Blaricum",
    "Prins Hendriklaan 76 1261 AJ Blaricum",
    "Prins Hendriklaan 68 1261 AJ Blaricum",
    "Prins Hendriklaan 58 1261 AJ Blaricum",
    "Prins Hendriklaan 54 1261 AJ Blaricum",
    "Prins Hendriklaan 48 1261 AJ Blaricum",
    "Prins Hendriklaan 66 1261 AJ Blaricum",
    "Prins Hendriklaan 40 1261 AJ Blaricum",
    "Prins Hendriklaan 30 1261 AJ Blaricum",
    "VOC-kade 314 1018 LG 100€. Amsterdam",
    "Noordereinde 12 1243 JG 's-Graveland",
    "van Beekstraat 124 1121 NT Landsmeer"
]

print("Model best")
# Checking predictions for the NER model
for address in address_list:
    doc=nlp(address)
    ent_list=[(ent.text, ent.label_) for ent in doc.ents]
    print("Address string -> "+address)
    print("Parsed address -> "+str(ent_list))
    print("******")