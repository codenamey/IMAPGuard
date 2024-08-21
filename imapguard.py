import imaplib
import email
from email.header import decode_header
from tqdm import tqdm
import pickle
import os
from dotenv import load_dotenv
from transformers import pipeline, MarianMTModel, MarianTokenizer, AutoModelForSequenceClassification, AutoTokenizer
import spacy
import torch
import gc
import signal
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from langdetect import detect, LangDetectException

# Load environment variables from .env file
load_dotenv()

# IMAP server settings from .env file
imap_server = os.getenv('IMAP_SERVER')
email_user = os.getenv('EMAIL_USER')
email_pass = os.getenv('EMAIL_PASS')

# File paths for saving data
emails_file = "emails.pkl"
progress_file = "progress.pkl"
replied_senders_file = "replied_senders.pkl"  # File to save replied senders

# White list of senders/domains that should not be marked as spam
white_list = ["trusted@example.com", "ficora.fi", "1702.fi", "traficom.fi", "tulli.fi", "posti.fi", "vero.fi", "kela.fi", "poliisi.fi", "vayla.fi", "vrk.fi", "sahko.fi", "prh"]

# Keywords that likely indicate a legitimate email
safe_keywords = ["invoice", "receipt", "order", "payment", "renewal", "alert", "notification"]

# Load spaCy models for grammatical analysis
try:
    nlp_finnish = spacy.load('fi_core_news_sm')  # Finnish model
except:
    spacy.cli.download('fi_core_news_sm')
    nlp_finnish = spacy.load('fi_core_news_sm')

try:
    nlp_english = spacy.load('en_core_web_sm')  # English model
except:
    spacy.cli.download('en_core_web_sm')
    nlp_english = spacy.load('en_core_web_sm')

# Load translation models for multilingual analysis
marian_fi_to_en_model_name = "Helsinki-NLP/opus-mt-fi-en"
marian_en_to_fi_model_name = "Helsinki-NLP/opus-mt-en-fi"
translator_fi_to_en = pipeline("translation", model=marian_fi_to_en_model_name)
translator_en_to_fi = pipeline("translation", model=marian_en_to_fi_model_name)

# Load FinBERT model for spam classification
finbert_model_name = "TurkuNLP/bert-base-finnish-cased-v1"
finbert_tokenizer = AutoTokenizer.from_pretrained(finbert_model_name)
finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_model_name)
spam_classifier = pipeline("text-classification", model=finbert_model, tokenizer=finbert_tokenizer)

# Load replied senders list if it exists
if os.path.exists(replied_senders_file):
    with open(replied_senders_file, "rb") as f:
        replied_senders = pickle.load(f)
else:
    replied_senders = set()

# Connect to the IMAP server and select the inbox
def connect_and_select_inbox():
    mail = imaplib.IMAP4_SSL(imap_server)
    mail.login(email_user, email_pass)
    mail.select('inbox')
    return mail

mail = connect_and_select_inbox()

# Load previous progress if it exists
if os.path.exists(progress_file):
    with open(progress_file, "rb") as f:
        processed_message_ids, last_processed_count = pickle.load(f)
else:
    processed_message_ids = set()
    last_processed_count = 0

# Fetch all email UIDs
status, messages = mail.uid('search', None, 'ALL')
email_uids = messages[0].split()

# Load previously collected emails and labels if they exist
if os.path.exists(emails_file):
    with open(emails_file, "rb") as f:
        emails, labels = pickle.load(f)
else:
    emails = []
    labels = []

# Define the Ahma-3B model for Finnish
finnish_model_name = "Finnish-NLP/Ahma-3B"
generator_finnish = pipeline("text-generation", model=finnish_model_name, tokenizer=finnish_model_name, device=-1, use_cache=False)

# Initialize traditional machine learning models
vectorizer = TfidfVectorizer(max_features=1000)
logistic_regression = LogisticRegression()
random_forest = RandomForestClassifier()

# Initialize counters for clean and spam emails
clean_emails = 0
spam_emails = 0

# Function to move emails to the Junk folder
def move_to_junk_folder(mail, uid, junk_folder='Junk'):
    result = mail.uid('COPY', uid, junk_folder)
    if result[0] == 'OK':
        mail.uid('STORE', uid, '+FLAGS', r'(\Deleted)')
        mail.expunge()

# Function to generate text based on input with timeout handling
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
        print("Generation timed out.")
        return ""
    except Exception as e:
        print(f"Error during text generation: {e}")
        return ""

# Function to preprocess email content
def preprocess_email_content(content, max_length=512):  # Further reduced max_length
    if len(content) > max_length:
        content = content[:max_length]
    return content

# Function to ensure IMAP connection is still active
def ensure_mail_connection(mail):
    try:
        mail.noop()  # Sends a no-op to keep the connection alive
    except imaplib.IMAP4.abort:
        mail = connect_and_select_inbox()
    return mail

# Function to save replied senders list
def save_replied_senders():
    with open(replied_senders_file, "wb") as f:
        pickle.dump(replied_senders, f)

# Function to detect language and translate if necessary
def translate_email_content(content):
    try:
        detected_lang = detect(content)
        if detected_lang == "fi":
            return content  # No translation needed
        elif detected_lang == "en":
            return content  # No translation needed
        elif detected_lang != "fi":
            translation = translator_fi_to_en(content)[0]['translation_text']
            return translation
        else:
            translation = translator_en_to_fi(content)[0]['translation_text']
            return translation
    except LangDetectException as e:
        print(f"Error detecting language: {e}")
        return content

# Print initial status
print("Processing emails...")

total_emails = len(email_uids)
batch_size = max(1, total_emails // 200)  # Increased batch size to reduce memory usage per batch
start_index = last_processed_count
end_index = min(start_index + batch_size, total_emails)

# Set up the batch progress bar
batch_progress_bar = tqdm(total=batch_size, unit="emails", leave=True, position=0)

try:
    while start_index < total_emails:
        # Reset and configure batch progress bar
        batch_progress_bar.n = 0
        batch_progress_bar.last_print_n = 0
        batch_progress_bar.total = min(batch_size, total_emails - start_index)
        batch_progress_bar.refresh()

        for i, uid in enumerate(email_uids[start_index:end_index]):
            try:
                mail = ensure_mail_connection(mail)
                status, msg_data = mail.uid('fetch', uid, '(RFC822)')
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])

                        # Get the Message-ID
                        message_id = msg.get("Message-ID")
                        if not message_id or message_id in processed_message_ids:
                            continue

                        # Decode the email subject
                        subject_header = msg["Subject"]
                        if subject_header is not None:
                            subject, encoding = decode_header(subject_header)[0]
                            try:
                                if isinstance(subject, bytes):
                                    subject = subject.decode(encoding if encoding else "utf-8", errors='replace')
                            except LookupError:
                                subject = subject.decode('utf-8', errors='replace')
                        else:
                            subject = "(No Subject)"

                        # Process the email body
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body = part.get_payload(decode=True).decode(errors='replace')
                        else:
                            body = msg.get_payload(decode=True).decode(errors='replace')

                        # Preprocess email content
                        body = preprocess_email_content(body)

                        # Translate email content if necessary
                        body = translate_email_content(body)

                        # Combine subject and body for analysis
                        full_text = f"{subject} {body}"

                        # Check if the sender is in the white list or replied senders
                        sender = msg.get("From")
                        if sender in replied_senders or any(whitelisted in sender for whitelisted in white_list):
                            tqdm.write(f"Email '{subject}' from '{sender}' is in the replied or white list, skipping spam check.")
                            clean_emails += 1
                            continue

                        # If the subject or body contains safe keywords, skip spam check
                        if any(keyword.lower() in full_text.lower() for keyword in safe_keywords):
                            tqdm.write(f"Email '{subject}' contains safe keywords, skipping spam check.")
                            clean_emails += 1
                            continue

                        # Generate text and classify with timeout using Ahma-3B model
                        try:
                            generated_text = generate_text_with_timeout(generator_finnish, body, max_length=25, max_new_tokens=10, timeout=30)
                            is_spam_ahma = 'spam' in generated_text.lower()
                        except Exception as e:
                            tqdm.write(f"Error during text generation: {e}")
                            is_spam_ahma = False

                        # Classify email using FinBERT model
                        try:
                            finbert_result = spam_classifier(full_text)[0]
                            is_spam_finbert = finbert_result['label'] == 'LABEL_1'  # Assuming LABEL_1 is spam
                        except Exception as e:
                            tqdm.write(f"Error during FinBERT classification: {e}")
                            is_spam_finbert = False

                        # Determine final spam status based on both classifiers
                        is_spam = is_spam_ahma or is_spam_finbert

                        if is_spam:
                            move_to_junk_folder(mail, uid, junk_folder='Junk')
                            tqdm.write(f"Email '{subject}' moved to Junk folder by classifier.")
                            spam_emails += 1
                        else:
                            tqdm.write(f"Email '{subject}' is not spam.")
                            clean_emails += 1
                            # Add sender to replied list
                            replied_senders.add(sender)
                            save_replied_senders()

                        # Update progress
                        processed_message_ids.add(message_id)
                        batch_progress_bar.update(1)

                        # Update the description with current statistics
                        current_progress = (start_index + i + 1) / total_emails * 100
                        tqdm.write(f"Progress ({current_progress:.2f}%) | Clean messages: {clean_emails} | Spam messages: {spam_emails}")

                        # Delay to avoid overwhelming the mail server
                        time.sleep(1)

            except Exception as e:
                tqdm.write(f"Error processing email UID {uid}: {e}")
            finally:
                # Ensure that progress is always updated
                batch_progress_bar.update(1)

        # Run garbage collection to free up memory
        gc.collect()

        # Move to the next batch
        start_index = end_index
        end_index = min(start_index + batch_size, total_emails)

finally:
    # Close the connection to the server
    mail.close()
    mail.logout()

    # Close the batch progress bar
    batch_progress_bar.close()

    # Final garbage collection
    gc.collect()
