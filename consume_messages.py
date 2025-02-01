from kafka import KafkaConsumer

# Kafka configuration
bootstrap_servers = 'localhost:9092'
topic = 'health_data_predicted'

# Create a Kafka consumer
consumer = KafkaConsumer(topic,
                         bootstrap_servers=bootstrap_servers,
                         auto_offset_reset='latest',  # Start reading from the beginning if no offset is stored
                         enable_auto_commit=False)  # Disable auto-commit to manually control offsets

try:
	for message in consumer:
		print(f"Received message: {message.value.decode('utf-8')}")
except KeyboardInterrupt:
	pass
finally:
	# Close the consumer
	consumer.close()
