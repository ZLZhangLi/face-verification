#! /usr/bin/env python
#-*- coding:utf-8 -*-
import os
# Iterate file system.
def saveFileInfo(file_list, output_path):
	print "Writing file info to", output_path
	with open(output_path, 'w') as f:
		for filenames in file_list:
			for item in filenames:
				line = ''.join(str(item)) + '\n'
				f.write(line)
		f.close()

if __name__ == '__main__':
	file_info = [[1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9, 9.0],[1,2,3]]
	print(file_info)
	output_path = './result2.txt'
	#f = open(output_path,'w')
	#for filenames in file_info:
	#	f.write(str(filenames))
	#	f.write('\n')
	#f.close()
	saveFileInfo(file_info, output_path)