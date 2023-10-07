import random

def main():
    file_path = "C:/Users/Callum/OneDrive/Desktop/VS Code Projects/Python/ARK/csv files/"
    dataset_name = "augmented_rgb_trashnet.csv"

    lines = readFile(file_path+dataset_name)
    
    train_lines, test_lines = train_and_test(lines, dataset_name)

    grey_bool = False
    train_lines = addHeadings(train_lines, grey_bool)
    test_lines = addHeadings(test_lines, grey_bool)

    writeFile(file_path, "train_"+dataset_name, train_lines)
    writeFile(file_path, "test_"+dataset_name, test_lines)

def readFile(file_path):
    with open(file_path, "r") as f:
        lines = []
        for line in f:
            lines.append(line.replace("\n", ""))

    return lines

def train_and_test(lines, dataset_name):
    random.shuffle(lines)

    train_lines = lines[:18500]
    test_lines = lines[18500:]

    return train_lines, test_lines

def addHeadings(lst, grey_bool):
    headers = ["label"]
    if grey_bool:
        for i in range(1, 16385):
            headers.append(f"pixel{i}")
    else:
        for i in range(1, 16385):
            for j in ["a", "b", "c"]:
                headers.append(f"pixel{i}{j}")
    
    lst.insert(0, ",".join(headers))
    return lst

def writeFile(file_path, file_name, lines):
    with open(file_path+file_name, "w") as f:
        for line in lines:
            f.write(line)
    f.close()

main()