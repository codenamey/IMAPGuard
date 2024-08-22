import unittest
from unittest.mock import patch, MagicMock
from tools.email_processing import (
    preprocess_email_content,
    connect_and_select_inbox,
    move_to_junk_folder,
    ensure_mail_connection,
    save_replied_senders
)
import imaplib

class TestEmailProcessing(unittest.TestCase):

    def test_preprocess_email_content(self):
        content = "This is a test email content that is very long." * 10
        processed_content = preprocess_email_content(content, max_length=50)
        self.assertEqual(len(processed_content), 50)

    @patch('tools.email_processing.imaplib.IMAP4_SSL')
    def test_connect_and_select_inbox(self, MockIMAP):
        mock_mail = MagicMock()
        MockIMAP.return_value = mock_mail
        mock_mail.login.return_value = ('OK', [b'Logged in'])
        mock_mail.select.return_value = ('OK', [b'INBOX selected'])

        mail = connect_and_select_inbox()

        mock_mail.login.assert_called_once()
        mock_mail.select.assert_called_once_with('inbox')

    @patch('tools.email_processing.imaplib.IMAP4_SSL')
    def test_move_to_junk_folder(self, MockIMAP):
        mock_mail = MagicMock()
        MockIMAP.return_value = mock_mail

        uid = b'1'
        mock_mail.uid.side_effect = [('OK', [b'COPY succeeded']), ('OK', [b'STORE succeeded'])]

        move_to_junk_folder(mock_mail, uid)
        mock_mail.uid.assert_any_call('COPY', uid, 'Junk')
        mock_mail.uid.assert_any_call('STORE', uid, '+FLAGS', r'(\Deleted)')

    @patch('tools.email_processing.imaplib.IMAP4_SSL')
    def test_ensure_mail_connection(self, MockIMAP):
        mock_mail = MagicMock()
        MockIMAP.return_value = mock_mail
        mock_mail.noop.side_effect = imaplib.IMAP4.abort('Test error')

        new_mail = ensure_mail_connection(mock_mail)

        MockIMAP.assert_called_once()
        mock_mail.noop.assert_called_once()

    def test_save_replied_senders(self):
        replied_senders = {"test@example.com"}
        with patch("builtins.open", unittest.mock.mock_open()) as mock_file:
            save_replied_senders(replied_senders, "test_replied_senders.pkl")
            mock_file.assert_called_once_with("test_replied_senders.pkl", "wb")
            mock_file().write.assert_called_once()

if __name__ == '__main__':
    unittest.main()
