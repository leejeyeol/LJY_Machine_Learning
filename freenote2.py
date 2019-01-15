import csv
class Deep_Learning_CSV_Saver():
    '''
    # Usage
    import random
    csv_saver = Deep_Learning_CSV_Saver(rows=['a', 'b', 'c', 'd'], save_path='output.csv')
    for i in range(0, 100):
        for j in range(0, 100):
            iteration_result = [random.randint(0, 10), random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)]
            csv_saver.add_column(iteration_result)
        csv_saver.save()
    '''
    def __init__(self, rows=['1', '2', '3', '4'], save_path='output.csv'):
        self.results = []
        self.rows = rows
        self.len_rows = len(self.rows)
        self.save_path = save_path
        self.rows_write = False


    def add_column(self, data_list):
        self.results.append(data_list)

    def save(self):
        with open(self.save_path, 'a') as outcsv:
            # configure writer to write standard csv file
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            if not self.rows_write:
                writer.writerow(self.rows)
                self.rows_write = True
            for item in self.results:
                # Write item to outcsv
                print(item)
                writer.writerow([item[0], item[1], item[2], item[3]])
            self.results = []