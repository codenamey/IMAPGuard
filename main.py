import os  # Tämä varmistaa, että os-moduuli on käytössä
import pickle
from tqdm import tqdm
import gc
import time
import email
from tools.email_processing import connect_and_select_inbox, move_to_junk_folder, ensure_mail_connection, save_replied_senders
from tools.translation import translate_email_content
from tools.spam_detection import generate_text_with_timeout, update_models_with_not_spam
from conf.config import WHITE_LIST, SAFE_KEYWORDS, SUBJECT_SPAM_THRESHOLD

# Main program logic here
# Load previous progress if it exists
progress_file = "progress.pkl"
if os.path.exists(progress_file):
    with open(progress_file, "rb") as f:
        processed_message_ids, last_processed_count = pickle.load(f)
else:
    processed_message_ids = set()
    last_processed_count = 0

mail = connect_and_select_inbox()

# Fetch all email UIDs
status, messages = mail.uid('search', None, 'ALL')

# Tulosta status ja UID:t tarkistusta varten
print(f"Status: {status}")
print(f"Messages: {messages}")

# Tarkista, onko sähköposteja löytynyt
if status == 'OK' and messages[0]:
    email_uids = messages[0].split()
    print(f"Found {len(email_uids)} emails.")
else:
    print("No emails found or failed to fetch emails.")
    exit()

# Tarkistus silmukan aloituksessa
total_emails = len(email_uids)
batch_size = max(1, total_emails // 200)
start_index = last_processed_count
end_index = min(start_index + batch_size, total_emails)

print(f"Starting email processing from index {start_index} to {end_index}")

batch_progress_bar = tqdm(total=batch_size, unit="emails", leave=True)

try:
    while start_index < total_emails:
        batch_progress_bar.n = 0
        batch_progress_bar.last_print_n = 0
        batch_progress_bar.total = min(batch_size, total_emails - start_index)
        batch_progress_bar.refresh()

        # Initialize a dictionary to track subject counts
        subject_counter = {}

        for i, uid in enumerate(email_uids[start_index:end_index]):
            print(f"Processing email UID {uid} ({i + 1}/{len(email_uids[start_index:end_index])})")
            try:
                mail = ensure_mail_connection(mail)
                status, msg_data = mail.uid('fetch', uid, '(RFC822)')
                print(f"Fetch status: {status}")
                if status != 'OK':
                    print(f"Failed to fetch email UID {uid}")
                    continue

                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject = msg["Subject"] if msg["Subject"] else "(No Subject)"
                        print(f"Email subject: {subject}")

                        # Ota mukaan peruskäsittely, kuten spam-tarkistus
                        # (Tähän tulee käsittelylogiikka)

            except Exception as e:
                print(f"Error processing email UID {uid}: {e}")
            finally:
                print(f"Finished processing email UID {uid}")
                batch_progress_bar.update(1)

        start_index = end_index
        end_index = min(start_index + batch_size, total_emails)

finally:
    with open(progress_file, "wb") as f:
        pickle.dump((processed_message_ids, start_index), f)
    save_replied_senders()

    mail.close()
    mail.logout()

    batch_progress_bar.close()

    gc.collect()
