import imaplib
import email
from email.header import decode_header
from tqdm import tqdm
import pickle
import os
from dotenv import load_dotenv
from transformers import pipeline
import spacy
import torch
import gc
import signal

# Load environment variables from .env file
load_dotenv()

# IMAP server settings from .env file
imap_server = os.getenv('IMAP_SERVER')
email_user = os.getenv('EMAIL_USER')
email_pass = os.getenv('EMAIL_PASS')

# File paths for saving data
emails_file = "emails.pkl"
progress_file = "progress.pkl"

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

# Connect to the IMAP server and fetch emails
mail = imaplib.IMAP4_SSL(imap_server)
mail.login(email_user, email_pass)
mail.select('inbox')

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

# Function for grammatical analysis using spaCy
def perform_grammatical_analysis(text, language='fi'):
    if language == 'fi':
        doc = nlp_finnish(text)
    else:
        doc = nlp_english(text)
    return doc

# Print initial status
print("Processing emails...")

total_emails = len(email_uids)
batch_size = max(1, total_emails // 200)  # Increased batch size to reduce memory usage per batch
start_index = last_processed_count
end_index = min(start_index + batch_size, total_emails)

# Set up the batch progress bar
batch_progress_bar = tqdm(total=batch_size, desc="Batch Processing", unit="emails", leave=True, position=0)

try:
    while start_index < total_emails:
        # Reset and configure batch progress bar
        batch_progress_bar.n = 0
        batch_progress_bar.last_print_n = 0
        batch_progress_bar.total = min(batch_size, total_emails - start_index)
        batch_progress_bar.refresh()

        for i, uid in enumerate(email_uids[start_index:end_index]):
            try:
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

                        # Determine language and perform grammatical analysis
                        if 'fi' in subject.lower() or 'fi' in body.lower():
                            doc = perform_grammatical_analysis(body, language='fi')
                        else:
                            doc = perform_grammatical_analysis(body, language='en')

                        # Combine subject and body for analysis
                        full_text = f"{subject} {body}"

                        # Check if the sender is in the white list
                        sender = msg.get("From")
                        if any(whitelisted in sender for whitelisted in white_list):
                            tqdm.write(f"Email '{subject}' from '{sender}' is in the white list, skipping spam check.")
                            continue

                        # If the subject or body contains safe keywords, skip spam check
                        if any(keyword.lower() in full_text.lower() for keyword in safe_keywords):
                            tqdm.write(f"Email '{subject}' contains safe keywords, skipping spam check.")
                            continue

                        # Generate text and classify with timeout
                        try:
                            generated_text = generate_text_with_timeout(generator_finnish, body, max_length=25, max_new_tokens=10, timeout=30)
                            is_spam = 'spam' in generated_text.lower()
                        except Exception as e:
                            tqdm.write(f"Error during text generation: {e}")
                            is_spam = False

                        if is_spam:
                            move_to_junk_folder(mail, uid, junk_folder='Junk')
                            tqdm.write(f"Email '{subject}' moved to Junk folder by classifier.")
                        else:
                            tqdm.write(f"Email '{subject}' is not spam.")

                        # Update progress
                        processed_message_ids.add(message_id)
                        batch_progress_bar.update(1)

                        # Update the progress file
                        with open(progress_file, "wb") as f:
                            pickle.dump((processed_message_ids, last_processed_count), f)

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
