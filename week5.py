file1 = open("MyFile1.txt", "a")

file2 = open(r"MyFile2.txt", "w+")

file1 = open("MyFile1.txt", "a")
#file1.close()

# - Delete any string I have inserted into txt file
# file_path = 'MyFile1.txt'
# string_delete = 'Hello World'
# with open(file_path, 'r') as f:
#    content = f.read()

# updated_content = content.replace(string_delete, '')

# with open(file_path, 'w') as f:
#    f.write(updated_content)

# file1.write("Hello World, My name is Vinny")

file_path = 'MyFile1.txt'
string_delete = 'Hello World, My name is VinnyHello World, My name is VinnyHello World, My name is Vinny'
with open(file_path, 'r') as f:
    content = f.read()
updated_content = content.replace(string_delete, '')
with open(file_path, 'w') as f:
    f.write(updated_content)