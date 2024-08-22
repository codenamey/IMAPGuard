import os
import pickle
from tqdm import tqdm
import gc
import time
import email
from tools.email_processing import (
    connect_and_select_inbox,
    move_to_junk_folder,
    ensure_mail_connection,
    save_replied_senders,
    preprocess_email_content
)
from tools.translation import translate_email_content
from tools.spam_detection import generate_text_with_timeout, update_models_with_not_spam
from conf.config import WHITE_LIST, SAFE_KEYWORDS, SUBJECT_SPAM_THRESHOLD

# Main program logic
progress_file = "progress.pkl"
if os.path.exists(progress_file):
    with open(progress_file, "rb") as f:
        processed_message_ids, last_processed_count = pickle.load(f)
else:
    processed_message_ids = set()
    last_processed_count = 0

mail = connect_and_select_inbox()

status, messages = mail.uid('search', None, 'ALL')
if status == 'OK' and messages[0]:
    email_uids = messages[0].split()
    print(f"Found {len(email_uids)} emails.")
else:
    print("No emails found or failed to fetch emails.")
    exit()

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

        subject_counter = {}

        for i, uid in enumerate(email_uids[start_index:end_index]):
            print(f"Processing email UID {uid} ({i + 1}/{len(email_uids[start_index:end_index])})")
            try:
                mail = ensure_mail_connection(mail)
                status, msg_data = mail.uid('fetch', uid, '(RFC822)')
                if status != 'OK':
                    print(f"Failed to fetch email UID {uid}")
                    continue

                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject = msg["Subject"] if msg["Subject"] else "(No Subject)"
                        print(f"Email subject: {subject}")

                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body = part.get_payload(decode=True).decode(errors='replace')
                        else:
                            body = msg.get_payload(decode=True).decode(errors='replace')

                        body = preprocess_email_content(body)
                        body = translate_email_content(body)
                        full_text = f"{subject} {body}"

                        sender = msg.get("From")
                        if any(whitelisted in sender for whitelisted in WHITE_LIST):
                            print(f"Email '{subject}' from '{sender}' is in the white list and passed header checks, skipping spam check.")
                            clean_emails += 1
                            continue

                        if any(keyword.lower() in full_text.lower() for keyword in SAFE_KEYWORDS):
                            print(f"Email '{subject}' contains safe keywords, skipping spam check.")
                            clean_emails += 1
                            continue

                        try:
                            generated_text = generate_text_with_timeout(generator_finnish, body, max_length=25, max_new_tokens=10, timeout=30)
                            is_spam = 'spam' in generated_text.lower()
                        except Exception as e:
                            print(f"Error during text generation: {e}")
                            is_spam = False

                        if is_spam:
                            move_to_junk_folder(mail, uid, junk_folder='Junk')
                            print(f"Email '{subject}' moved to Junk folder by classifier.")
                            spam_emails += 1
                        else:
                            print(f"Email '{subject}' is not spam.")
                            clean_emails += 1

                        processed_message_ids.add(message_id)
                        batch_progress_bar.update(1)

            except Exception as e:
                print(f"Error processing email UID {uid}: {e}")
            finally:
                print(f"Finished processing email UID {uid}")
                batch_progress_bar.update(1)

        gc.collect()
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
