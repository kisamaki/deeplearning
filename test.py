import csv
import numpy as np

def write_csv(file, save_dict):
    with open(file, "w") as f:
        w = csv.writer(f)
        save_row = []
        for key, values in save_dict.items():
            a = []
            a.append(key)
            if values.ndim != 1:
                for value in values:
                    a.append(value)
            else:
                a.append(values)
            save_row.append(a)
        w.writerows(save_row)

def read_dict(file):
    return_dict = {}
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            return_dict[row[0]] = []
            for value in row:
                if value == row[0]:
                    continue
                value = value.strip("[").strip("]").strip(" ").split(" ")
                value = [int(x) for x in value if x != ""]
                return_dict[row[0]].append(value)
        
    return return_dict

data = {}
data["A"] = np.array([[0, 1], [1, 0], [0, 1]])
data["K"] = np.array([[0, 0], [1, 1]])
data["S"] = np.array([0, 0, 0, 0, 0 ,0])
write_csv("test.csv", data)
print(read_dict("test.csv"))