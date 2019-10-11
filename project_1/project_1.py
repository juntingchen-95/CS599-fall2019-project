import csv
import math

# The class of KNN
class KNN:
    result = []

    # main method of the class, read the data, set the K-value, calculate and write the result in a CSV file
    def __init__(self, k_value):
        self.training_data = self.read_data('dataset/training_set.csv')
        self.test_data = self.read_data('dataset/test_set.csv')
        self.k_value = k_value
        self.calculate()
        self.write_data('result.csv')

    # The method to read the CSV file
    @staticmethod
    def read_data(file_path):
        data = []
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data

    # The method to calculate the result
    def calculate(self):
        for test_point in self.test_data:
            distance_list = []
            # Calculate the distance to all training data point
            for training_point in self.training_data:
                distance_list.append([
                    math.sqrt((float(test_point['x']) - float(training_point['x'])) ** 2
                              + (float(test_point['y']) - float(training_point['y'])) ** 2),
                    int(training_point['type'])
                ])
            # Sort the distance list, and calculate the class of the test point based on the K-value
            distance_list = sorted(distance_list)
            type_stat = [
                0,
                0
            ]
            for i in range(self.k_value):
                if distance_list[i][1] == 0:
                    type_stat[0] += 1
                else:
                    type_stat[1] += 1
            if type_stat[0] > type_stat[1]:
                data_type = 0
            else:
                data_type = 1
            self.result.append([
                test_point['id'],
                test_point['x'],
                test_point['y'],
                data_type
            ])

    # The method to write the calculated result in a CSV file
    def write_data(self, file_name):
        with open(file_name, 'a') as file:
            file.write('id,x,y,type\n')
            for row in self.result:
                file.write(row[0] + ',' + str(row[1]) + ',' + str(row[2]) + ',' + str(row[3]) + '\n')


if __name__ == '__main__':
    KNN = KNN(7)
