# IMAPGuard

(c) Lennart Takanen 

IMAPGuard is a Python-based spam filtering tool that leverages machine learning to classify and manage your emails. It connects to your IMAP email server, processes your inbox, and automatically identifies and moves spam messages to the appropriate folder.

## Installation

Before you begin, make sure you have Python installed on your system.

1. **Clone the Repository** (if needed):
   ```bash
   git clone https://github.com/yourusername/imapguard.git
   cd imapguard
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   ```

3. **Install Dependencies**:
   You can install the required Python libraries by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the root directory of the project with the following contents:

   ```bash
   IMAP_SERVER= 'your-email-server'
   EMAIL_USER = 'user@account.com'
   EMAIL_PASS = 'password'
   ```

   Replace `your-email-server`, `user@account.com`, and `password` with your actual email server, email address, and password, respectively.

2. This `.env` file securely stores your IMAP server credentials, allowing the program to access your email account without exposing sensitive information directly in the code.

## Usage

Once you have installed the necessary libraries and configured the `.env` file, you can run the program by executing the following command in your terminal:

```bash
python imapguard.py
```

## How It Works

- **IMAPGuard** connects to your email server using the credentials provided in the `.env` file.
- The program downloads emails from your inbox and uses a combination of Logistic Regression and Random Forest classifiers to identify potential spam.
- Repeatedly identified emails are automatically flagged as spam and handled accordingly.
- The program saves its progress and learned model data, so it can continue learning from where it left off the next time you run it.

## Contributing

Contributions to improve IMAPGuard are welcome! Feel free to submit pull requests or open issues with suggestions for enhancements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
