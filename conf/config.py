import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# IMAP server settings
IMAP_SERVER = os.getenv('IMAP_SERVER')
EMAIL_USER = os.getenv('EMAIL_USER')
EMAIL_PASS = os.getenv('EMAIL_PASS')

# White list of senders/domains that should not be marked as spam
WHITE_LIST = ["trusted@example.com", "ficora.fi", "1702.fi", "traficom.fi", "tulli.fi", "posti.fi", "vero.fi", "kela.fi", "poliisi.fi", "vayla.fi", "vrk.fi", "sahko.fi", "prh"]

# Keywords that likely indicate a legitimate email
SAFE_KEYWORDS = ["invoice", "receipt", "order", "payment", "renewal", "alert", "notification"]

# Threshold for the number of repetitions before marking as spam
SUBJECT_SPAM_THRESHOLD = 2
