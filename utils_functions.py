from path import get_path_file

def clone_table(table):
    return [row[::] for row in table]

def print_transportation_table(array):
    printing_table = clone_table(array)
    for i in range(len(array)-1):
        printing_table[i].insert(0, f"P{i+1}")
    printing_table.insert(0,[""])
    for i in range(len(array[0])-1):
        printing_table[0].append(f"C{i+1}")
    printing_table[0].append("provisions")
    printing_table[-1].insert(0, "order")
    printing_table[-1].append("")
    print_table(printing_table)

def print_table(printing_table):
    column_widths = [max(len(str(item)) for item in column) for column in zip(*printing_table)]
    print("+" + "+".join("-" * (width + 2) for width in column_widths) + "+")
    for row in printing_table:
        print("|", end=" ")
        for i, item in enumerate(row):
            if i > len(column_widths)-1:
                break
            print(f"{item: <{column_widths[i]}}", end=" | ")  
        print()  
        print("+" + "+".join("-" * (width + 2) for width in column_widths) + "+")

def read_transportation_table_from_txt(filename):
    with open(get_path_file(filename), 'r') as file:
        table = []
        cpt=0
        for line in file:
            row = line.strip().split()
            if cpt==0:
                cpt+=1
                continue
            table.append(row)
    table[-1].append("")
    return table

def print_allocation_table(array):
    for line in array:
        print(line)

