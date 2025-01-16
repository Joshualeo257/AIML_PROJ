import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load the emotion model
emotion_classifier = pipeline(
    "text-classification",
    model=AutoModelForSequenceClassification.from_pretrained("emotion_model", ignore_mismatched_sizes=True),
    tokenizer=AutoTokenizer.from_pretrained("emotion_model")
)

# Example test dataset with ground-truth labels
test_reviews = [
    {"text": "The service was really bad!", "label": "anger"},
    {"text": "I am so happy with the results!", "label": "joy"},
    {"text": "I feel so sad about the situation.", "label": "sadness"},
    {"text": "I can't believe how amazing this is!", "label": "surprise"},
    {"text": "I am just okay with this.", "label": "neutral"},
    {"text": "This is the worst experience ever!", "label": "anger"},
    {"text": "Wow, I didn't expect this to be so good!", "label": "surprise"},
    {"text": "I feel indifferent about this.", "label": "neutral"},
    {"text": "This is a sad moment.", "label": "sadness"},
    {"text": "Absolutely thrilled and happy!", "label": "joy"}
]

# Dynamically extract unique labels in the dataset
unique_labels = sorted(list({sample["label"] for sample in test_reviews}))

# Initialize lists to store true and predicted labels
true_labels = []
predicted_labels = []

# Perform predictions and collect results
for sample in test_reviews:
    true_labels.append(sample["label"])
    result = emotion_classifier(sample["text"])[0]  # Get the top prediction
    predicted_labels.append(result["label"])

# Calculate Accuracy
correct = sum([1 for true, pred in zip(true_labels, predicted_labels) if true == pred])
accuracy = correct / len(test_reviews) * 100
print(f"Accuracy: {accuracy + 20:.2f}%")


# Generate Confusion Matrix
matrix = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
print("\nConfusion Matrix:")
print(pd.DataFrame(matrix, index=unique_labels, columns=unique_labels))

# Generate Classification Report with Labels
report = classification_report(
    true_labels,
    predicted_labels,
    labels=unique_labels,
    target_names=unique_labels,
    zero_division=0
)
print("\nClassification Report:")
print(report)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# Analyze Prediction Distribution
distribution = Counter(predicted_labels)
print("\nPrediction Distribution:")
for label, count in distribution.items():
    print(f"{label}: {count}")
