import smtplib
import os
from email.mime.text import MIMEText
from email.header import Header
from dotenv import load_dotenv
import random
import numpy as np
import torch


def send_email_notification(subject, content, receiver):
    load_dotenv()
    # 从环境变量中获取邮箱配置
    mail_host = os.getenv("MAIL_HOST")
    mail_user = os.getenv("MAIL_USER")
    mail_pass = os.getenv("MAIL_PASS")

    if not all([mail_host, mail_user, mail_pass]):
        print("\n[EMAIL ERROR] Email configuration not found in .env file or environment variables.")
        print("Please create a .env file with MAIL_HOST, MAIL_USER, and MAIL_PASS.")
        return

    sender = mail_user
    message = MIMEText(content, "plain", "utf-8")

    # 只对昵称进行编码，邮箱地址保持明文并用尖括号括起来
    message["From"] = f"{Header('RL Training Bot', 'utf-8')} <{sender}>"

    # 收件人和主题也使用 Header 是个好习惯，以防它们包含特殊字符
    message["To"] = Header(receiver, "utf-8")
    message["Subject"] = Header(subject, "utf-8")

    try:
        print("\n[INFO] Training finished. Attempting to send email notification...")
        smtp_port = 465  # QQ邮箱的SSL端口

        # 使用 'with' 语句，可以自动管理连接的关闭，更安全
        with smtplib.SMTP_SSL(mail_host, smtp_port) as smtpObj:
            smtpObj.login(mail_user, mail_pass)
            smtpObj.sendmail(sender, [receiver], message.as_string())

        print(f"[SUCCESS] Email notification sent successfully to {receiver}.")
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email: {e}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
