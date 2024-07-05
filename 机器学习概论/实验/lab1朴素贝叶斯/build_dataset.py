from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from email import message_from_string
from email.message import EmailMessage
# import jieba
import os
import os.path as osp
import json
from tqdm import tqdm
import numpy as np
from  datetime import  datetime
label_base_file_path = "data/trec06p/label/index"

def build_json(label_base_file_path:str,output_path:str):
    output_dict = {}
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))  
    count_bad_encode = 0
    group_number_list = [0,0,0,0,0]
    encoding_attempts = ['utf-8', 'gbk', 'iso-8859-1', 'windows-1252']
    with open(label_base_file_path) as f:
        # import pdb;pdb.set_trace()
        data_list = f.readlines()
        for line in  tqdm(data_list):
            label, email_path = line.strip().split(" ")
            email_path = "data/trec06p" + email_path[2:]
            # print(email_path)
            # try:
            html_text = None
            for encoding in encoding_attempts:
                try:
                    # print(email_path)
                    with open(email_path, "r", encoding=encoding,errors='ignore') as f:
                        rough_html_text = f.read()
                        html_text = rough_html_text.encode(encoding, 'ignore').decode(encoding)

                
                    # with open(email_path,"r",encoding="utf-8") as f:
                    #     rough_html_text = f.read()
                    msg = message_from_string(html_text)
                    if msg['Received']:
                        try:
                            recieved_info = msg['Received'].split(' ')[1][msg['Received'].split(' ')[1].index('.')+1:]
                        except:
                            recieved_info = None
                    else:
                        recieved_info=None
                    # import pdb; pdb.set_trace()
                    if msg['Date']:
                        try:
                            date_info = (msg['Date'].split(' ')[0].strip(','),msg['Date'].split(' ')[-2].split(':')[0])
                        except:
                            date_info = None
                    else:
                        date_info=None
                    xmailer_info = None
                    if msg['X-Mailer']:
                        if "Outlook" in msg['X-Mailer']:
                            xmailer_info = "Outlook"
                        elif "devMail" in msg['X-Mailer']:
                            xmailer_info = "devMail"
                        elif "Frobozz" in msg['X-Mailer']:
                            xmailer_info = "Frobozz"
                        elif "Foxmail" in msg['X-Mailer']:
                            xmailer_info = "Foxmail"
                        elif "Bat" in msg['X-Mailer']:
                            xmailer_info = "Bat"
                    
                    # print(msg)
                    # 初始化用于存储不同类型正文的变量
                    text_body = []
                    html_body = []
                    # import pdb; pdb.set_trace()
                    # 遍历邮件的不同部分
                    if msg.is_multipart():
                        for part in msg.walk():
                            # 获取内容类型
                            content_type = part.get_content_type()
                            content_disposition = part.get("Content-Disposition")
                            
                            # 检查是否是文本或HTML正文部分
                            if content_type == "text/plain" and not content_disposition:
                                text_body.append(part.get_payload(decode=True).decode())
                            elif content_type == "text/html" and not content_disposition:
                                html_body.append(part.get_payload(decode=True).decode())
                    else:
                        # 非多部分邮件，直接获取正文
                        content_type = msg.get_content_type()
                        if content_type == "text/plain":
                            text_body.append(msg.get_payload(decode=True).decode())
                        elif content_type == "text/html":
                            html_body.append(msg.get_payload(decode=True).decode())
                    filtered_text_list = []
                    # print(text_body)
                    # print(html_body)
                    for text_content in text_body:
                        words = word_tokenize(text_content)
                        # 去除停用词
                        filtered_text_list.append([word.lower() for word in words if word.lower() not in stop_words and word.isalpha()])
                    for html_content in html_body:
                        soup = BeautifulSoup(html_content, 'html.parser')
                        text = soup.get_text()

                        words = word_tokenize(text)
                        # 去除停用词
                        filtered_text_list.append([word.lower() for word in words if word.lower() not in stop_words and word.isalpha()])
                    # group_number = np.random.randint(0,5)
                    output_dict[email_path.strip()] = {"content":filtered_text_list,"label":label,"encoding":encoding,"date_info":date_info,"recieved_info":recieved_info,"xmailer_info":xmailer_info}
                        # group_number_list[group_number] += 1
                    # except UnicodeDecodeError:
                    #     # print(f"打开文件失败 {email_path}")
                    #     # import pdb; pdb.set_trace()
                    #     # with open(email_path,"r") as f:
                    #     #     rough_html_text = f.read()
                    #     count_bad_encode += 1
                    break
                except UnicodeDecodeError:
                    continue  
            else:
                count_bad_encode += 1
                continue

        print(f"有效文件：{len(data_list)-count_bad_encode}，无效文件{count_bad_encode}")
        print(f"各组数量:{group_number_list}")
    with open(output_path,"w",encoding='utf-8') as f:
        json.dump(output_dict,f,ensure_ascii=False,indent=4)
if __name__ == "__main__":
    build_json(label_base_file_path,"./output.json")