from datasets import load_dataset

dataset = load_dataset("Tuana/game-of-thrones", split="train")
print(f"Dataset size: {len(dataset)}")
print(f"Column names: {dataset.column_names}")
print(f"\nFirst record:")
print(dataset[0])
print(f"\nSecond record:")
print(dataset[1])
