"""
-- Created by: Ashok Kumar Pant
-- Email: asokpant@gmail.com
-- Created on: 20/04/2025
"""
import torch
from datasets import load_dataset
# https://huggingface.co/datasets
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments


class SentimentClassifier:
    def __init__(self, model_name='bert-base-uncased', num_labels=2):
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.trainer = None

    def init_model(self):
        """Initialize the pretrained model."""
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)
        return self

    @classmethod
    def load_model(cls, checkpoint_path):
        """Load a trained model from checkpoint."""
        print(f"Loading model from {checkpoint_path}")
        instance = cls()
        instance.model = BertForSequenceClassification.from_pretrained(checkpoint_path)
        return instance

    def load_and_tokenize_data(self, dataset_name="imdb", train_size=2000, test_size=500):
        train_split = f"train[:{train_size}]"
        test_split = f"test[:{test_size}]"

        dataset = load_dataset(dataset_name, split={'train': train_split, 'test': test_split})

        def tokenize_fn(example):
            return self.tokenizer(example["text"], truncation=True, padding="max_length")

        tokenized_datasets = {
            "train": dataset["train"].map(tokenize_fn, batched=False),
            "test": dataset["test"].map(tokenize_fn, batched=False)
        }
        return tokenized_datasets["train"], tokenized_datasets["test"]

    def train(self, output_dir="./models", epochs=3, train_size=2000, test_size=500):
        self.init_model()
        train_data, eval_data = self.load_and_tokenize_data(train_size=train_size, test_size=test_size)

        training_args = TrainingArguments(output_dir=output_dir, eval_strategy="epoch",
                                          per_device_train_batch_size=64, per_device_eval_batch_size=64,
                                          num_train_epochs=epochs, learning_rate=2e-5,
                                          logging_dir=f"{output_dir}/logs", logging_steps=10, )

        self.trainer = Trainer(model=self.model, args=training_args, train_dataset=train_data, eval_dataset=eval_data)
        self.trainer.train()

    def evaluate(self):
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call train() first.")
        return self.trainer.evaluate()

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return predictions.tolist(), probs.tolist()


def training_example():
    classifier = SentimentClassifier()
    classifier.train()

    eval_results = classifier.evaluate()
    print("Evaluation Results:", eval_results)


def prediction_example():
    classifier = SentimentClassifier.load_model("./models/checkpoint-96")

    sample_text = "This movie was fantastic! The acting and story were top-notch."
    prediction, probability = classifier.predict(sample_text)
    print(f"Prediction: {prediction}, Probability: {probability}")


if __name__ == "__main__":
    training_example()
    # prediction_example()
