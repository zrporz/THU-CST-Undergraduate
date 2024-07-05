import os
import re
import xml.etree.ElementTree as ET
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np


def preprocess(folder_path,output_path):

    stop_words = set(stopwords.words('english'))
    porter = PorterStemmer()

    df_list = []


    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
 
            tree = ET.parse(os.path.join(folder_path, filename))
            root = tree.getroot()

            title = root.find(".//title").text if root.find(".//title") is not None else None
            

            publication_day_of_month = root.find(".//meta[@name='publication_day_of_month']").attrib.get('content', None)
            publication_month = root.find(".//meta[@name='publication_month']").attrib.get('content', None)
            publication_year = root.find(".//meta[@name='publication_year']").attrib.get('content', None)
            publication_day_of_week = root.find(".//meta[@name='publication_day_of_week']").attrib.get('content', None)
            

            content_blocks = root.findall(".//block[@class='full_text']")
            content = ''
            for block in content_blocks:
                for p in block.findall(".//p"):
                    if p.text is not None:
                        content += p.text.strip() + ' '
            
            content = content.strip() 


            content = re.sub(r'[^\w\s]', '', content)
            content = re.sub(r'\d+', '', content)
            content = content.lower()
            
            words = word_tokenize(content)

            words = [word for word in words if word not in stop_words]

            words = [porter.stem(word) for word in words]

            processed_content = ' '.join(words)

            categories = set()
            classifier_nodes = root.findall(".//classifier")
            for node in classifier_nodes:
                node_split = node.text.split('/')
                if len(node_split)>2:
                    if node_split[0]=="Top" and node_split[1]=="News"or node_split[1]=="Features":
                        category = node_split[2]
                        categories.add(category)

            df_list.append({
                'title': title,
                'publication_day_of_month': publication_day_of_month,
                'publication_month': publication_month,
                'publication_year': publication_year,
                'publication_day_of_week': publication_day_of_week,
                'content': processed_content,
                'categories': ', '.join(categories) if categories else None
            })

    df = pd.DataFrame(df_list)
    print("Preprocess successfully")
    print(df)
    # 将数据保存为CSV文件
    df.to_csv(output_path, index=False)

    print(f"Data saved successfully! Save to {output_path}")

def transform2bagofwords(input_path,output_path):
    df = pd.read_csv(input_path)
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(df['content'].fillna('')) 

    vocab = vectorizer.get_feature_names_out()

    bow_df = pd.DataFrame(X.toarray(), columns=vocab)

    final_df = pd.concat([df, bow_df], axis=1)

    print("Wordbag build successfully")
    print(final_df)
    final_df.to_csv(output_path)
    print(f"Save to {output_path}")
    return bow_df, final_df

def plot_wordcloud(bow_df,output_path):
    from wordcloud import WordCloud
    

    # 计算每个词在所有新闻中的出现次数
    word_counts = bow_df.sum(axis=0)

    # 选择出现次数最多的100个词
    top_100_words = word_counts.sort_values(ascending=False).head(100)

    # 生成词云图
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_100_words)

    # 显示词云图
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path)

def plot_length_distribution(bow_df,output_path):
    # 计算每个单词的长度
    word_lengths = bow_df.columns.str.len()

    # 统计每个单词长度的数量
    word_length_counts = word_lengths.value_counts().sort_index()

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(word_length_counts.index, word_length_counts.values, color='skyblue')
    plt.title('Distribution of Word Length')
    plt.xlabel('Word Length')
    plt.ylabel('Count')
    plt.savefig(output_path)

def plot_box(bow_df,output_path1,output_path2):

    word_counts = bow_df.sum(axis=1)

    quantiles = np.quantile(word_counts, np.linspace(0, 1, 11))
    
    indices = np.digitize(word_counts, bins=quantiles, right=True)
    
    _, counts = np.unique(indices, return_counts=True)
    
    counts[1] += counts[0]
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(1, 11), counts[1:], tick_label=[f"[{int(quantiles[i])},{int(quantiles[i+1])}]" for i in range(len(quantiles)-1)],color='skyblue')
    plt.title('Depth Binning')
    plt.xlabel('Word Count Ranges')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path1)
    # plt.close()
    
    # 计算等宽的箱子边界
    counts, bin_edges = np.histogram(word_counts, bins=10)
    
    # 绘制直方图
    plt.figure(figsize=(15, 8))
    # import pdb; pdb.set_trace()
    plt.bar(range(1, 11), counts, tick_label=[f"[{int(bin_edges[i])},{int(bin_edges[i+1])}]" for i in range(10)],color='skyblue')
    plt.title('Width Binning')
    plt.xlabel('Word Count Ranges')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path2)

def plot_category(final_df, output_path):
    category_counts = final_df['categories'].str.split(', ').explode().value_counts()

    plt.figure(figsize=(12, 6))
    category_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of News Count by Category')
    plt.xlabel('Category')
    plt.ylabel('Number of News')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)

def plot_month(final_df,output_path):
    final_df['publication_month'] = final_df['publication_month'].astype(int)
    final_df['publication_month'] = final_df['publication_month'].apply(lambda x: str(x).zfill(2))  # 将月份格式化为两位数

    monthly_counts = final_df['publication_month'].astype(int)

    monthly_counts = monthly_counts.value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    monthly_counts.plot(kind='bar', color='skyblue')
    plt.title('Distribution of News Count by Month')
    plt.xlabel('Month')
    plt.ylabel('Number of News')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(output_path)

if __name__ =="__main__":
    preprocess("nyt_corpus/samples_500",'nyt_corpus_processed.csv')
    bow_df,final_df = transform2bagofwords('nyt_corpus_processed.csv','nyt_corpus_bagofwords.csv')
    plot_wordcloud(bow_df,"word_cloud.png")
    plot_length_distribution(bow_df,"word_length_distribution.png")
    plot_box(bow_df,"depth_binning.png","width_binning.png")
    plot_category(final_df,'category_distribution.png')
    plot_month(final_df,'month_distribution.png')
    