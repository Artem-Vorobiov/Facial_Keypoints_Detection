import os
import shutil
from tqdm import tqdm 


print('\n')
print(os.getcwd())
path     = os.getcwd()
list_dir = os.listdir()
print('\n\tHERE\n')
print(list_dir)
print('\n\n')

for file in tqdm(list_dir):
	if file == '.git':
		# way = path + '/' + file
		# path1 = os.access(file, os.F_OK) 
		# print("Exists the path:", path1)

		# path2 = os.access(file, os.R_OK) 
		# print("Access to read the file:", path2)

		# path3 = os.access(file, os.W_OK) 
		# print("Access to write the file:", path3)

		# path4 = os.access(file, os.X_OK) 
		# print("Check if path can be executed:", path4)

		shutil.rmtree(file, ignore_errors=True)
		print('Removed')
	else:
		print('Nope ')
print('\n\tResult\n')
print(os.listdir())
print('\n\n')


# http://weblomaster.ru/исправление-ошибки-operation-not-permitted-при-удалении/
