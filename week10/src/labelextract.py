path1 = "http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json"
labels = requests.get(path1)
json_response = labels.content.decode()
dict_json = json.loads(json_response)
print(dict_json.keys())
csvData = dict()
for key, value in dict_json.items():
    print(key, value['label'])
    csvData[key] = len(value['label'])
ll = sorted(csvData.items(), key=lambda x: x[1], reverse=True)
print(ll)