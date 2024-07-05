import re
from pypinyin import lazy_pinyin
def is_contain_chinese(check_str):
    for ch in check_str:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

with open('appendix.txt', 'r',encoding='utf-8') as f:
    lines = f.readlines()
    result = []
    for line in lines:
        line = line.strip()
        if line == '':
            continue
        ans = line.split(' ')
        key = " ".join(ans[:-1])
        value = ans[-1]
        result.append((key, value))

chinese_data = []
non_chinese_data = []
for item in result:
    if is_contain_chinese(item[0]):
        chinese_data.append(item)
    else:
        non_chinese_data.append(item)
# 按照字典序排序
chinese_data_sorted_dict = sorted(chinese_data, key=lambda x: x[0])
non_chinese_data_sorted_dict = sorted(non_chinese_data, key=lambda x: x[0].lower())

# 按照拼音序排序
chinese_data_sorted_pinyin = sorted(chinese_data, key=lambda x: lazy_pinyin(x[0]))
non_chinese_data_sorted_pinyin = sorted(non_chinese_data, key=lambda x: lazy_pinyin(x[0]))

# print("中文部分按照字典序排序：", chinese_data_sorted_dict)
print("非中文部分按照字典序排序：", non_chinese_data_sorted_dict)
print("中文部分按照拼音序排序：", chinese_data_sorted_pinyin)
# print("非中文部分按照拼音序排序：", non_chinese_data_sorted_pinyin)

# 输出到文件并添加首字母提示
with open('appendix_sorted_dict.txt', 'w', encoding='utf-8') as f:
    current_initial = ''
    for item in non_chinese_data_sorted_dict:
        initial = item[0].strip()[0].upper()
        if initial != current_initial:
            f.write('\n' + initial + '\n')
            current_initial = initial
        f.write(item[0] + ' ' + item[1] + '\n')

with open('appendix_sorted_pinyin.txt', 'w', encoding='utf-8') as f:
    current_initial = ''
    for item in chinese_data_sorted_pinyin:
        pinyin = lazy_pinyin(item[0].strip())[0]
        initial = pinyin[0].upper()
        if initial != current_initial:
            f.write('\n' + initial + '\n')
            # f.write(initial + '\n')
            current_initial = initial
        f.write(item[0] + ' ' + item[1] + '\n')