#Implement and demonstrate FIND-S Algorithm for finding the most specific hypothesis based on a given set of training examples.
import random
import csv

attributes=[['Sunny','Rainy'],
			['Warm','Cold'],
			['Normal','High'],
			['Strong','Weak'],
			['Warm','Cool'],
			['Same','Change']]

num_attributes=len(attributes)

print("\nThe most general hypothesis:[?,?,?,?,?,?]\n")
print("\nThe most specific hypothesis:[0,0,0,0,0,0]\n")

a=[]

print("\nThe given Training Data Set\n")
with open('CSVFile.csv','r') as csvFile:
	reader=csv.reader(csvFile)
	for row in reader:
		a.append(row)
		print(row)

print("\nThe initial value of hypothesis:")
hypothesis=['0']*num_attributes
print(hypothesis)

for j in range(0,num_attributes):
	hypothesis[j]=a[0][j] #fill the hypothesis with the a's first row 

print("\nFind-S: Finding a maximally Specific Hypothesis\n")

for i in range(0,len(a)):
	if a[i][num_attributes]=='Yes':
		for j in range(0,num_attributes):
			if a[i][j] != hypothesis[j]:
				hypothesis[j]='?'

print("For training example No:{}".format(i),hypothesis)

print("\nThe final hypothesis is:")
print(hypothesis)
