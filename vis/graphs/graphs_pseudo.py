import csv

import matplotlib.pyplot as plt
import numpy as np

# the class generator for vis for non-temporal approaches
name = 'D:\Thesis\weights\F1\NA\ps\pseudolabel'
STOP_CRITERIA = 50

pz_train_loss = 0
cz_train_loss = 1
us_train_loss = 2
afs_train_loss = 3
bg_train_loss = 4

pz_train_dice = 5
cz_train_dice = 6
us_train_dice = 7
afs_train_dice = 8
bg_train_dice = 9

pz_val_loss = 10
cz_val_loss = 11
us_val_loss = 12
afs_val_loss = 13
bg_val_loss = 14

pz_val_dice = 15
cz_val_dice = 16
us_val_dice = 17
afs_val_dice = 18
bg_val_dice = 19

train_loss = 20
val_loss = 21

counter = 0
with open(name) as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=';', quoting=csv.QUOTE_NONE)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            print('Column names are {"; ".join(row)}')
        else:
            line_count += 1
    print('Processed {line_count} lines.')
    total_lines = line_count
    graph_arr = np.zeros((22, line_count))
    prev_val_loss = 1000
    with open(name) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=';', quoting=csv.QUOTE_NONE)
        line_count = 0
        for row in csv_reader:

            graph_arr[pz_train_loss][line_count - 1] = row["pz_loss"]
            graph_arr[cz_train_loss][line_count - 1] = row["cz_loss"]
            graph_arr[us_train_loss][line_count - 1] = row["us_loss"]
            graph_arr[afs_train_loss][line_count - 1] = row["afs_loss"]
            graph_arr[bg_train_loss][line_count - 1] = row["bg_loss"]

            graph_arr[pz_train_dice][line_count - 1] = row["pz_dice_coef"]
            graph_arr[cz_train_dice][line_count - 1] = row["cz_dice_coef"]
            graph_arr[us_train_dice][line_count - 1] = row["us_dice_coef"]
            graph_arr[afs_train_dice][line_count - 1] = row["afs_dice_coef"]
            graph_arr[bg_train_dice][line_count - 1] = row["bg_dice_coef"]

            graph_arr[pz_val_loss][line_count - 1] = row["val_pz_loss"]
            graph_arr[cz_val_loss][line_count - 1] = row["val_cz_loss"]
            graph_arr[us_val_loss][line_count - 1] = row["val_us_loss"]
            graph_arr[afs_val_loss][line_count - 1] = row["val_afs_loss"]
            graph_arr[bg_val_loss][line_count - 1] = row["val_bg_loss"]

            graph_arr[pz_val_dice][line_count - 1] = row["val_pz_dice_coef"]
            graph_arr[cz_val_dice][line_count - 1] = row["val_cz_dice_coef"]
            graph_arr[us_val_dice][line_count - 1] = row["val_us_dice_coef"]
            graph_arr[afs_val_dice][line_count - 1] = row["val_afs_dice_coef"]
            graph_arr[bg_val_dice][line_count - 1] = row["val_bg_dice_coef"]

            graph_arr[train_loss][line_count - 1] = float(row["pz_loss"]) + float(row["cz_loss"]) + float(
                row["us_loss"]) + float(row["afs_loss"])
            graph_arr[val_loss][line_count - 1] = float(row["val_pz_loss"]) + float(row["val_cz_loss"]) + float(
                row["val_us_loss"]) + float(row[
                                                "val_afs_loss"])

            cur_val_loss = graph_arr[val_loss][line_count - 1]
            if prev_val_loss > cur_val_loss:
                counter += 1
            prev_val_loss = cur_val_loss
            line_count += 1
            if counter == 50:
                print(line_count)
                break

        print('Processed {line_count} lines.')
        epochs = np.arange(0, line_count - 1)

    # line 1 points
    x1 = epochs
    y1 = graph_arr[train_loss][0:line_count - 1]
    # plotting the line 1 points
    plt.plot(x1, y1, label="train_loss")

    # line 2 points
    x2 = epochs
    y2 = graph_arr[val_loss][0:line_count - 1]
    # plotting the line 2 points
    plt.plot(x2, y2, label="val_loss")

    x3 = epochs
    y3 = graph_arr[cz_train_dice][0:line_count - 1]
    # plotting the line 1 points
    # plt.plot(x3, y3, label="cz_train_dice")

    # line 2 points
    x4 = epochs
    y4 = graph_arr[cz_val_dice][0:line_count - 1]
    # plotting the line 2 points
    # plt.plot(x4, y4, label="cz_val_dice")

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    # giving a title to my graph
    plt.title('Two lines on same graph!')

    # show a legend on the plot
    plt.legend()
    plt.show()
