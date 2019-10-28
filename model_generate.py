from naive_bayes_classifier import NB
import csv
filename = 'IMDB_dataset_sa.csv'
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    reviews = []
    labels = []
    # extract rows
    i= 0
    for row in csvreader:
        if csvreader.line_num == 1:
            continue  #skip first row
        reviews.append(row[0])
        labels.append(row[1].upper())
    # print("Done.")

nb = NB(reviews,labels)
import pickle
p_out = open("model.pkl","wb")
pickle.dump(nb, p_out)
p_out.close()
