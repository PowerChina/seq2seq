import uniout

file = open("./data/train/editorial.txt");
#file = open("./data/train/test.txt");
data = []
line = file.readline()
while line:
    line = line.strip()
    if not len(line) or not line.startswith('Q:'):
        line = file.readline()
        continue
    data.append(line[2:])
    line = file.readline()
    line = line.strip()
    data.append(line)
    line = file.readline()

output_file = open("./data/train/test_out.txt","w")
for v in data:
    output_file.write(v+"\n")

file.close()
output_file.close()
