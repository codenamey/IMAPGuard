import unittest
from unittest.mock import patch, MagicMock
from tools.email_processing import connect_and_select_inbox, move_to_junk_folder

class TestEmailProcessing(unittest.TestCase):

    @patch('tools.email_processing.imaplib.IMAP4_SSL')
    def test_connect_and_select_inbox(self, MockIMAP):
        # Luo mock-object IMAP:ille
        mock_mail = MagicMock()
        MockIMAP.return_value = mock_mail

        # Mockaa login- ja select-metodit
        mock_mail.login.return_value = ('OK', [b'Logged in'])
        mock_mail.select.return_value = ('OK', [b'INBOX selected'])

        # Suorita testattava funktio
        mail = connect_and_select_inbox()

        # Varmista, että login ja select on kutsuttu
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

if __name__ == '__main__':
    unittest.main()
