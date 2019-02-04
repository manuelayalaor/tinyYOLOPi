import csv
FRAMES_TO_SAVE = 30


def extract_csv(file, output_dir):
    with open(file) as csv_file:
        with open(output_dir, "w+") as csv_output:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_writer = csv.writer(csv_output, delimiter=',')
            i = 0
            line_count = 0
            for row in csv_reader:
            #Skip first row since it is just the header
                if line_count == 0:
                    line_count += 1
                    continue
                if i % FRAMES_TO_SAVE == 0:
                    csv_writer.writerow(row)
                i += 1


if __name__ == "__main__":
    input_dir = input("What file to read: ")
    output_dir = input("Where to Output: ")
    extract_csv(input_dir, output_dir)
    print("Extraction Successful")