import imaplib
import email
from email.header import decode_header
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from tqdm import tqdm
import pickle
import os
import hashlib
from dotenv import load_dotenv
from transformers import pipeline, AutoTokenizer

# Lataa ympäristömuuttujat .env-tiedostosta
load_dotenv()

# IMAP-palvelimen asetukset .env-tiedostosta
imap_server = os.getenv('IMAP_SERVER')
email_user = os.getenv('EMAIL_USER')
email_pass = os.getenv('EMAIL_PASS')

# Tallennustiedoston polut
emails_file = "emails.pkl"
model_file = "spam_filter_model.pkl"
progress_file = "progress.pkl"
repeated_emails_file = "repeated_emails.pkl"

# Toistuvien viestien kirjanpito
repeated_emails = {}

# Yhdistä IMAP-palvelimeen ja hae sähköpostit
mail = imaplib.IMAP4_SSL(imap_server)
mail.login(email_user, email_pass)
mail.select('inbox')

# Lataa Hugging Facen valmiiksi koulutettu BERT-malli ja tokenisaattori
bert_classifier = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english', device=0)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# Lataa aiempi edistyminen, jos olemassa
if os.path.exists(progress_file):
    with open(progress_file, "rb") as f:
        processed_message_ids, last_processed_count = pickle.load(f)
    print(f"Loaded {len(processed_message_ids)} processed Message-IDs, last processed count: {last_processed_count}.")
else:
    processed_message_ids = set()
    last_processed_count = 0

# Hae kaikki sähköpostit
status, messages = mail.uid('search', None, 'ALL')
email_uids = messages[0].split()

# Lataa aiemmin kerätyt sähköpostit ja luokittelut, jos olemassa
if os.path.exists(emails_file):
    with open(emails_file, "rb") as f:
        emails, labels = pickle.load(f)
else:
    emails = []
    labels = []

# Lataa aiemmin tallennetut toistuvat viestit, jos olemassa
if os.path.exists(repeated_emails_file):
    with open(repeated_emails_file, "rb") as f:
        repeated_emails = pickle.load(f)

# Lataa tai luo uusi koneoppimismalli
if os.path.exists(model_file):
    with open(model_file, "rb") as f:
        voting_clf, vectorizer = pickle.load(f)
else:
    lr_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100)
    voting_clf = VotingClassifier(estimators=[
        ('lr', lr_model),
        ('rf', rf_model)
    ], voting='hard')  # 'hard' tarkoittaa enemmistöpäätöstä

    vectorizer = CountVectorizer()

# Tarkista, onko `vectorizer` sovitettu
if not hasattr(vectorizer, 'vocabulary_') or not vectorizer.vocabulary_:
    vectorizer.fit(emails)

# Funktio tekstin pilkkomiseen tokenien mukaan
def split_text_into_chunks(text, tokenizer, chunk_size=512):
    tokens = tokenizer(text, return_tensors='pt', truncation=False, add_special_tokens=False)['input_ids'][0]
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        chunks.append(chunk_text)
    return chunks

print("Classifying emails...")
progress_bar = tqdm(email_uids, desc="Processing emails", leave=True)

for i, uid in enumerate(progress_bar):
    if i < last_processed_count:
        continue

    status, msg_data = mail.uid('fetch', uid, '(RFC822)')
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            msg = email.message_from_bytes(response_part[1])

            # Hanki Message-ID
            message_id = msg.get("Message-ID")
            if not message_id or message_id in processed_message_ids:
                continue  # Ohita, jos viestillä ei ole Message-ID:tä tai se on jo käsitelty

            # Tarkista, onko viestillä otsikko
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

            # Viestin tekstin käsittely
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors='replace')
            else:
                body = msg.get_payload(decode=True).decode(errors='replace')

            # Yhdistä otsikko ja viesti analyysia varten
            full_text = f"{subject} {body}"

            # Pilko pitkä viesti tokenien mukaan
            chunks = split_text_into_chunks(full_text, tokenizer)

            # Käytä BERT-mallia jokaisen palan analysoimiseen
            is_spam = False
            for chunk in chunks:
                # Truncate chunk to ensure it's within the model's limit
                bert_result = bert_classifier(chunk[:512])
                bert_label = bert_result[0]['label']
                if bert_label == 'NEGATIVE':  # NEGATIVE tulkitaan mahdollisesti roskapostiksi
                    is_spam = True
                    break

            if is_spam:
                mail.uid('store', uid, '+FLAGS', r'(\Flagged)')
                tqdm.write(f"Email '{subject}' flagged as spam by BERT model.")
                continue

            # Luodaan tiiviste viestin sisällöstä ja otsikosta
            email_hash = hashlib.md5((subject + body).encode('utf-8')).hexdigest()

            # Tarkistetaan, onko viestiä toistettu useita kertoja
            if email_hash in repeated_emails:
                repeated_emails[email_hash]['count'] += 1
                repeated_emails[email_hash]['uids'].append(uid)
            else:
                repeated_emails[email_hash] = {'count': 1, 'uids': [uid]}

            # Jos viesti on toistunut yli 3 kertaa, merkitään se ja kaikki aiemmat samanlaiset viestit roskapostiksi
            if repeated_emails[email_hash]['count'] > 3:
                labels.append(1)
                tqdm.write(f"Repeated email '{subject}' flagged as spam.")

                # Merkitään kaikki aikaisemmat viestit, joilla on sama hash, roskapostiksi
                for spam_uid in repeated_emails[email_hash]['uids']:
                    mail.uid('store', spam_uid, '+FLAGS', r'(\Flagged)')

                continue  # Ohitetaan koneoppimismallin käyttö, koska viesti on jo merkitty roskapostiksi

            # Lisätään viesti normaalisti käsittelyyn, jos se ei ole toistunut liian monta kertaa
            emails.append(body)

            # Muunna uusi sähköposti ja päivitä mallia
            if len(emails) > 1:  # Tarvitaan vähintään 2 datapistettä
                X = vectorizer.transform(emails)
                if X.shape[0] == len(labels):
                    voting_clf.fit(X, labels)  # Päivitetään malli uudella datalla

            # Tee ennustus ja käsittele viesti
            new_data = vectorizer.transform([body])
            prediction = voting_clf.predict(new_data)[0]

            if prediction == 1:
                mail.uid('store', uid, '+FLAGS', r'(\Flagged)')
                tqdm.write(f"Flagged email '{subject}' as spam by combined model.")
            else:
                tqdm.write(f"Email '{subject}' is not spam by combined model.")

            # Lisää käsitelty viesti tallennusjoukkoon
            processed_message_ids.add(message_id)

    last_processed_count = i + 1
    with open(progress_file, "wb") as f:
        pickle.dump((processed_message_ids, last_processed_count), f)

    with open(emails_file, "wb") as f:
        pickle.dump((emails, labels), f)

    with open(model_file, "wb") as f:
        pickle.dump((voting_clf, vectorizer), f)

    with open(repeated_emails_file, "wb") as f:
        pickle.dump(repeated_emails, f)

mail.close()
mail.logout()
