import os
md_list = os.listdir('./')
md_list = [i for i in md_list if i.endswith('.md') and i!='all.md']
md_list.sort(key=lambda x: int(x[1] if x[2]=='讲' else x[1:3]))

with open('all.md', 'w', encoding='utf-8') as f:
    for md in md_list:
        print(md)
        with open(md, 'r', encoding='utf-8') as f2:
            f.write(f2.read())
            f.write('\n\n')
            
print('all.md has been created')
print('total', len(md_list), 'files')
print('last file:', md_list[-1])
