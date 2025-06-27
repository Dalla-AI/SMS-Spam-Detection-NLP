import pandas as pd

df = pd.read_parquet (r"C:\Users\dalla\sms_spam\plain_text\train-00000-of-00001.parquet")
print(df.head())


df.to_csv("SMS_Spam.csv", index=False)
print("CSV file dowloaded! Thanks prof W")



