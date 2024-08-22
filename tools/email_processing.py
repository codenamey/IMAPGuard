import imaplib
import email
from email.header import decode_header
import pickle
from tqdm import tqdm

from conf.config import IMAP_SERVER, EMAIL_USER, EMAIL_PASS

def connect_and_select_inbox(folder='inbox'):
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    mail.select(folder)
    return mail

def move_to_junk_folder(mail, uid, junk_folder='Junk'):
    result = mail.uid('COPY', uid, junk_folder)
    if result[0] == 'OK':
        mail.uid('STORE', uid, '+FLAGS', r'(\Deleted)')
        mail.expunge()

def ensure_mail_connection(mail):
    try:
        mail.noop()  # Sends a no-op to keep the connection alive
    except imaplib.IMAP4.abort:
        mail = connect_and_select_inbox()
    return mail

def save_replied_senders(replied_senders, file_path="replied_senders.pkl"):
    with open(file_path, "wb") as f:
        pickle.dump(replied_senders, f)
