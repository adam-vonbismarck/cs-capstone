import os
import sys
import smtplib
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

# -------------------- Configuration -------------------- #

# Path to your Google Service Account credentials JSON file
SERVICE_ACCOUNT_FILE = 'secrets/creds.json'

# Name of the Google Sheet
GOOGLE_SHEET_NAME = 'BMC EXAMPLE TELEM FOR COLIN'

# Email Configuration
SMTP_SERVER = 'smtp.gmail.com'  # e.g., 'smtp.gmail.com' for Gmail
SMTP_PORT = 465  # 465 for SSL, 587 for TLS
SENDER_EMAIL = 'colin_baker@brown.edu'

with open('secrets/pass.txt', mode='r') as file:
    pw = file.readline().replace(' ', '')

SENDER_PASSWORD = pw  # Use App Password if using Gmail
RECEIVER_EMAIL = 'colin_baker@brown.edu'
EMAIL_SUBJECT = 'Daily Google Sheet Data'

# ------------------------------------------------------- #

def get_google_sheet_data(service_account_file, sheet_name):
    """
    Authenticates with Google Sheets and retrieves data.
    
    :param service_account_file: Path to the service account JSON file.
    :param sheet_name: Name of the Google Sheet to access.
    :return: Pandas DataFrame containing the sheet data.
    """
    try:
        # Define the scope
        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        # Authenticate using the service account
        creds = Credentials.from_service_account_file(
            service_account_file, scopes=scope)
        client = gspread.authorize(creds)
        
        # Open the Google Sheet
        sheet = client.open(sheet_name).sheet1  # Access the first sheet
        data = sheet.get_all_values()  # Get all data as a list of dictionaries
    
        # Convert to Pandas DataFrame
        df = pd.DataFrame(data)
        
        df_temp = df.iloc[4:, 1:]
        
        df_temp = df_temp.T.rename(columns={4:"Workout"}).set_index("Workout").T
        
        return df_temp.iloc[:, :np.argmax(df_temp.columns.get_loc(""))].set_index("Name").map(lambda x : float('nan') if x == "" else x).dropna(how="any", axis=0).drop(index="Name")

    except Exception as e:
        print(f"Error accessing Google Sheet: {e}")
        return None

def send_email(smtp_server, smtp_port, sender_email, sender_password,
               receiver_email, subject, html_content, plain_content=None):
    """
    Sends an email with the given content.
    
    :param smtp_server: SMTP server address.
    :param smtp_port: SMTP server port.
    :param sender_email: Sender's email address.
    :param sender_password: Sender's email password.
    :param receiver_email: Receiver's email address.
    :param subject: Subject of the email.
    :param html_content: HTML content of the email.
    :param plain_content: Plain text content of the email (optional).
    """
    try:
        # Create a multipart message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = sender_email
        message["To"] = receiver_email

        # Attach plain text if provided
        if plain_content:
            part1 = MIMEText(plain_content, "plain")
            message.attach(part1)
        
        # Attach HTML content
        part2 = MIMEText(html_content, "html")
        message.attach(part2)

        # Connect to the SMTP server and send the email
        with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
            server.login(sender_email, sender_password)
            server.sendmail(
                sender_email, receiver_email, message.as_string()
            )
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

def main():
    # Fetch data from Google Sheet
    df = get_google_sheet_data(SERVICE_ACCOUNT_FILE, GOOGLE_SHEET_NAME)
    if df is None:
        print("Failed to retrieve data. Exiting.")
        return
    
    df.index.name = None
    df.columns.name = None
    
    # Convert DataFrame to HTML
    html_table = df.to_html(index=True)
    
    # Optional: Convert DataFrame to CSV string
    csv_data = df.to_csv(index=True)
    
    # Create plain text content (optional)
    plain_text = csv_data  # You can customize this as needed

    # Send the email
    send_email(
        smtp_server=SMTP_SERVER,
        smtp_port=SMTP_PORT,
        sender_email=SENDER_EMAIL,
        sender_password=SENDER_PASSWORD,
        receiver_email=RECEIVER_EMAIL,
        subject=EMAIL_SUBJECT,
        html_content=html_table,
        plain_content=plain_text
    )

if __name__ == "__main__":
    main()
