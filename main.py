from headPoseEstimation import headPoseEstimation
from itertools import islice
import concurrent.futures

def process_sublist(sublist, csv_file):
    obj = headPoseEstimation(sublist, csv_file)
    obj.run()

def main():
    dataArray = [f"CHEAT {i}" for i in range(1, 195)]

    for i in range(1, 140):
        dataArray.append(f"NON-CHEAT {i}")

    # Uncomment the next two lines if you want to include NON-CHEAT data
    # for i in range(1, 140):
    #     dataArray.append(f"NON-CHEAT {i}")

    # random.shuffle(dataArray)

    length_to_split = [9] * 37

    dataArray = iter(dataArray)
    dataArray = [list(islice(dataArray, elem)) for elem in length_to_split]

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks to the executor
        futures = [executor.submit(process_sublist, sublist, f"double-data {idx + 1}.csv") for idx, sublist in enumerate(dataArray)]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()