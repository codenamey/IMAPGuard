
# IMAPGuard

(c) Lennart Takanen

IMAPGuard is an advanced email filtering tool that leverages a combination of deep learning, natural language processing (NLP), and traditional machine learning techniques to classify and manage emails. It connects to your IMAP email server, processes your inbox, and automatically identifies and moves spam messages to the appropriate folder. The tool is designed to handle multilingual email content, including Finnish and English, by using state-of-the-art models like the Ahma-3B model for Finnish text generation.

## Installation

Before you begin, make sure you have Python installed on your system. You can install the required Python libraries by running the following command:

```bash
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the root directory of the project with the following contents:

```plaintext
IMAP_SERVER= 'your-email-server'
EMAIL_USER = 'user@account.com'
EMAIL_PASS = 'password'
```

Replace `your-email-server`, `user@account.com`, and `password` with your actual email server, email address, and password, respectively.

2. This `.env` file securely stores your IMAP server credentials, allowing the program to access your email account without exposing sensitive information directly in the code.

## Usage

Once you have installed the necessary libraries and configured the `.env` file, you can run the program by executing the following command in your terminal:

```bash
python -m venv myenv
source myenv/bin/activate

python imapguard.py
```

## Key Features and Techniques

1. **Deep Learning:**
   - Utilizes the Ahma-3B model for advanced text generation and analysis in Finnish.
   - Supports English text processing using the GPT-2 model, ensuring effective handling of multilingual emails.

2. **Natural Language Processing (NLP):**
   - Incorporates advanced NLP techniques to parse and understand email content, making spam detection more accurate.
   - Uses spaCy for enhanced grammar-based NLP analysis, improving the model's ability to distinguish between spam and legitimate emails.

3. **Pretrained Classifiers:**
   - Combines deep learning with traditional machine learning classifiers like Logistic Regression and Random Forest to compare and enhance spam detection performance.
   - Employs an ensemble method to aggregate the results from different models, boosting the overall accuracy of spam detection.

4. **Grammar-based NLP:**
   - Uses spaCy's language models to perform detailed grammatical analysis, further improving the system's ability to identify spam based on linguistic patterns.

5. **White List Filtering:**
   - Allows trusted senders or domains to be whitelisted, ensuring that important emails are never mistakenly marked as spam.

6. **Safe Keyword Detection:**
   - Scans emails for specific safe keywords (like "invoice" or "payment") that likely indicate legitimate emails, bypassing the spam check for these messages.

## Contributing

Contributions to improve IMAPGuard are welcome! Feel free to submit pull requests or open issues with suggestions for enhancements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
