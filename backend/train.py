import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def train_model():
    print("Loading dataset...")
    # Load your dataset here (e.g., WELFake)
    # df = pd.read_csv('fake_news_dataset.csv')
    # For demonstration, creating a dummy dataset
    data = {
        'text': ["This is fake news", "This is a real factual report"],
        'label': [1, 0] # 1 for Fake, 0 for Real
    }
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )
    
    print("Starting training...")
    trainer.train()
    print("Saving model...")
    model.save_pretrained("./fine_tuned_fakenews_model")
    tokenizer.save_pretrained("./fine_tuned_fakenews_model")

if __name__ == "__main__":
    train_model()
