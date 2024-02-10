from kafka import KafkaConsumer, KafkaProducer, TopicPartition
import json
import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import base64
import schedule
import time

load_dotenv()

source_topic_name = os.getenv('SOURCE_TOPIC_NAME')
destination_topic_name = os.getenv('DESTINATION_TOPIC_NAME')
bootstrap_servers = [os.getenv('BROKER')]  
group_id = os.getenv('GROUP_ID')  

def train_model():

    consumer = KafkaConsumer(
        source_topic_name,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest', 
        enable_auto_commit=False,
        group_id=group_id,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))  
    )

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )


    consumer.subscribe([source_topic_name])
    partitions = consumer.partitions_for_topic(source_topic_name)
    tp_list = [TopicPartition(source_topic_name, p) for p in partitions]
    end_offsets = consumer.end_offsets(tp_list)

    print(f"Collecting dataset...")

    dataset = []
    try:
        for message in consumer:
            dataset.append(message.value)
            
            if message.offset >= end_offsets[TopicPartition(source_topic_name, message.partition)] - 1:
                break

            if len(dataset) >= 10000:
                break
    finally:
        consumer.close()

    print(f"Collected {len(dataset)} messages for training.")

    texts = [d['message'] for d in dataset]  

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    input_sequences = []
    target_sequences = []

    for seq in sequences:
        for i in range(1, len(seq)):
            input_seq = seq[:i]

            target_seq = seq[i]
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    num_classes = len(tokenizer.word_index) + 1  
    target_sequences = tf.keras.utils.to_categorical(target_sequences, num_classes=num_classes)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=num_classes, output_dim=64, input_length=max_sequence_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(input_sequences, target_sequences, epochs=100, verbose=1)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  
        tf.lite.OpsSet.SELECT_TF_OPS  
    ]
    converter._experimental_lower_tensor_list_ops = False  

    tflite_model = converter.convert()

    # Encode the TFLite model in base64
    model_base64 = base64.b64encode(tflite_model).decode('utf-8')

    # Serialize the tokenizer to JSON
    tokenizer_json = tokenizer.to_json()

    combined_payload = {
        "tokenizer": tokenizer_json,
        "model": model_base64
    }

    producer.send(destination_topic_name, value=combined_payload)
    producer.flush()
    print(f"Model Trained and Sent!")

schedule.every().day.at("00:00").do(train_model)

if __name__ == "__main__":
    train_model()
    while True:
        schedule.run_pending()
        time.sleep(1)