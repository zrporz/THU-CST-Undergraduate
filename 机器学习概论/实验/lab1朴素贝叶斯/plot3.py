import json
with open("output.json","r",encoding='utf-8') as f:
    data = json.load(f)
count = {"spam":{},"ham":{}}
for key, value in  data.items():
    k = (value["date_info"][0],value["date_info"][1]) if value["date_info"] else "None"
    if k not in count[value['label']]:
        count[value['label']][k] = 0
    count[value['label']][k] += 1

for key,value in count["spam"].items():
    if value>1000:
        print(key)
for key,value in count["ham"].items():
    if value>1000:
        print(key)
# print(count)
        
# Convert the count dictionary to a format suitable for plotting.
labels = set(count['spam'].keys()).union(count['ham'].keys())
spam_counts = [count['spam'].get(label, 0) for label in labels]
ham_counts = [count['ham'].get(label, 0) for label in labels]

# Plotting
import matplotlib.pyplot as plt

x = range(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x, spam_counts, width, label='Spam')
rects2 = ax.bar([p + width for p in x], ham_counts, width, label='Ham')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Counts')
ax.set_title('Counts by month and time')
# ax.set_xticks([p + width / 2 for p in x])
# ax.set_xticklabels(labels)
ax.legend()

plt.show()