import fastapi
import pandas
import io
import fastapi.responses
import joblib

app = fastapi.FastAPI()

model = joblib.load("./RidgeCVModel.pkl")


@app.get("/")
async def home():
    return "This is a Machine Learning Application to predict burn area"


@app.post("/predict")
async def upload_file(
    file: fastapi.UploadFile = fastapi.File(
        description="Upload a CSV file containing client data to predict whether the client will churn or not"
    ),
):
    # Read the data file.
    print(file.filename)
    data = await file.read()
    # print(data)
    # print(type(data))

    # Convert the data file into pandas dataframe.
    df = pandas.read_csv(io.BytesIO(data))
    df["burn_area"] = 200
    df["another_column"] = 34
    print(df)
    print(type(df))

    # response = fastapi.responses.FileResponse(path=df, filename='download', media_type='text/csv')
    response = fastapi.responses.StreamingResponse(
        io.StringIO(df.to_csv(index=False)), media_type="text/csv"
    )

    return response
