import json
import time
import pandas as pd
from kafka import KafkaProducer

online_df = pd.read_csv("online.csv")

online_df_x = online_df.drop(columns=["Diabetes_binary"])
online_df_y = online_df["Diabetes_binary"]

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = "health_data"

for index, row in online_df_x.iterrows():
    record = row.to_dict()

    producer.send(topic, value=record)
    print(f"Sent record: {record}")

    time.sleep(3)

producer.flush()
producer.close()
