from transformers import pipeline
import signal
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Initialize traditional machine learning models
vectorizer = TfidfVectorizer(max_features=1000)
logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()

# Define the Ahma-3B model for Finnish
finnish_model_name = "Finnish-NLP/Ahma-3B"
generator_finnish = pipeline("text-generation", model=finnish_model_name, tokenizer=finnish_model_name, device=0, use_cache=False)

def generate_text_with_timeout(generator, input_text, max_length=25, max_new_tokens=10, timeout=30):
    def handler(signum, frame):
        raise TimeoutError("Timeout during text generation")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        result = generator(input_text, max_length=max_length, max_new_tokens=max_new_tokens, pad_token_id=generator.tokenizer.eos_token_id, truncation=True)
        signal.alarm(0)  # Reset the alarm
        return result[0]['generated_text']
    except TimeoutError:
        tqdm.write("Generation timed out.")
        return ""
    except Exception as e:
        tqdm.write(f"Error during text generation: {e}")
        return ""

def update_models_with_not_spam(not_spam_texts, not_spam_labels, emails, labels, vectorizer, logistic_regression, random_forest):
    if not_spam_texts:
        # Ensure labels and emails lists are consistent
        if len(emails) != len(labels):
            tqdm.write("Warning: Length mismatch between emails and labels. Adjusting to minimum common length.")
            min_length = min(len(emails), len(labels))
            emails[:] = emails[:min_length]
            labels[:] = labels[:min_length]

        # Load some spam examples from the previously processed data
        spam_texts = [emails[i] for i in range(len(emails)) if labels[i] == 1]
        spam_labels = [1] * len(spam_texts)

        # Combine spam and not_spam examples
        combined_texts = not_spam_texts + spam_texts
        combined_labels = not_spam_labels + spam_labels

        # Ensure there are at least two classes in the combined data
        if len(set(combined_labels)) > 1:
            # Fit the vectorizer if not already fitted
            if not hasattr(vectorizer, 'vocabulary_'):
                vectorizer.fit(combined_texts)

            # Transform the combined texts
            X_combined = vectorizer.transform(combined_texts)

            # Train the models with the combined data
            logistic_regression.fit(X_combined, combined_labels)
            random_forest.fit(X_combined, combined_labels)

            tqdm.write("Models updated with not_spam and spam data.")
        else:
            tqdm.write("Insufficient class diversity to update models. No action taken.")
    else:
        tqdm.write("No not_spam data to update models.")
