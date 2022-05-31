import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(df):

    scaler = StandardScaler()
    encoder = LabelEncoder()

    df.drop(
        columns=["user_id", "REGION", "MRG", "TOP_PACK", "ZONE1", "ZONE2"], inplace=True
    )

    num_cols = [
        "MONTANT",
        "FREQUENCE_RECH",
        "REVENUE",
        "ARPU_SEGMENT",
        "FREQUENCE",
        "DATA_VOLUME",
        "ON_NET",
        "ORANGE",
        "TIGO",
        "REGULARITY",
        "FREQ_TOP_PACK",
    ]

    for feature in num_cols:
        df[feature].fillna((df[feature].mean()), inplace=True)

    df[num_cols] = scaler.fit_transform(df[num_cols])

    df["TENURE"] = encoder.fit_transform(df["TENURE"])

    return df


######  Testing It Out   ######
# import joblib

# model = joblib.load("./XGBClassifier.pkl")

# test = pd.read_csv("../../../Data/Test.csv")

# data = preprocess(test)
# print()
# print()
# print(data.select_dtypes(include=["int64", "float64"]).describe().T)
# print()
# print(data.info())

# data["CHURN"] = model.predict_proba(test)[:, 1]
# print(data.head())
