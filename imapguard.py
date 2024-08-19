import imaplib
import email
from email.header import decode_header
from tqdm import tqdm
import pickle
import os
from dotenv import load_dotenv
from transformers import pipeline

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
white_list = ["trusted@example.com", "1702.fi", "traficom.fi", "tulli.fi", "posti.fi", "vero.fi", "kela.fi", "poliisi.fi", "vayla.fi", "vrk.fi", "sahko.fi", "prh"]

# Keywords that likely indicate a legitimate email
safe_keywords = ["invoice", "receipt", "order", "payment", "renewal", "alert", "notification"]

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
generator_finnish = pipeline("text-generation", model=finnish_model_name, tokenizer=finnish_model_name, device=-1)

# Function to move emails to the Junk folder
def move_to_junk_folder(mail, uid, junk_folder='Junk'):
    result = mail.uid('COPY', uid, junk_folder)
    if result[0] == 'OK':
        mail.uid('STORE', uid, '+FLAGS', r'(\Deleted)')
        mail.expunge()

# Function to generate text based on input
def generate_text(generator, input_text, max_length=50):
    try:
        result = generator(input_text, max_length=max_length, pad_token_id=generator.tokenizer.eos_token_id, truncation=True)
        return result[0]['generated_text']
    except Exception as e:
        print(f"Error during text generation: {e}")
        return ""

# Print initial status
print("Processing emails...")

total_emails = len(email_uids)
batch_size = max(1, total_emails // 100)
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

                        # Generate text and classify
                        try:
                            generated_text = generate_text(generator_finnish, body, max_length=50)
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

        # Move to the next batch
        start_index = end_index
        end_index = min(start_index + batch_size, total_emails)

finally:
    # Close the connection to the server
    mail.close()
    mail.logout()

    # Close the batch progress bar
    batch_progress_bar.close()
