import pickle
import os
import numpy as np
import time


def table_list():
    files = os.listdir('tables')
    return sorted(files)


def load_table(table_name):
    with open(table_name, "rb") as file:
        return pickle.load(file)


if __name__ == '__main__':
    print("\u001b[1mPossible Tables:\u001b[0m")
    tables = table_list()
    for i in range(len(tables)):
        print(str(i + 1).rjust(4) + ") " + tables[i])
    index = input("Table index: ")
    try:
        index = int(index)
    except ValueError:
        print("Index was not an integer!")
        exit(1)
    index -= 1

    if index < 0 or index >= len(tables):
        print("Index out of range!")
        exit(1)

    table = tables[index]
    print()

    start_time = time.time()
    print(f"\u001b[33m[ ] Loading table \"{table}\"\u001b[0m", flush=True, end='')
    table_data = load_table('tables/' + table)
    print(f"\r\u001b[32m[✓] Table \"{table}\" has been loaded!\u001b[0m")

    shape = np.shape(table_data)
    target_shape = shape[0:-1]
    num_actions = shape[-1]

    compressed_table = np.empty(target_shape, dtype=np.byte)
    total_entries = 1
    for i in target_shape:
        total_entries *= i

    print("\u001b[33m[ ] Compressing table;   0%\u001b[0m", flush=True, end='')

    current_index = 0
    current_progress = 0
    for index in np.ndindex(target_shape):
        best_index = table_data[index].argmax()
        compressed_table[index] = best_index

        current_index += 1
        if round((100 * current_index) / total_entries) > current_progress:
            current_progress = round((100 * current_index) / total_entries)
            print(f"\r\u001b[33m[ ] Compressing table; {str(current_progress).rjust(3)}%\u001b[0m", flush=True, end='')
    print(f"\r\u001b[32m[✓] Finished compressing table.\u001b[0m")

    print("\u001b[33m[ ] Saving compressed table.\u001b[0m", flush=True, end='')
    with open("q-table-compressed.pt", "wb") as file:
        pickle.dump(compressed_table, file)
    print(f"\r\u001b[32m[✓] Finished saving table.  \u001b[0m")

    print(f"\u001b[32m[✓] Compressing took {round(time.time() - start_time, 1)} seconds.\u001b[0m")