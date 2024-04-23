from pathlib import Path
from InquirerPy import inquirer
from art import *
from logger import Settings, print
from tools import *
from tabulate import tabulate

styles = ["fancy_grid", "rounded_grid", "mixed_grid"]

path = Path(__file__).parent

if __name__ == '__main__':
    tprint("Graph\n Schedulator\n", font="tarty1",
           chr_ignore=True, decoration="block")
    print("By Paul Mairesse, Axel Loones, Louis Le Meilleur & Joseph Benard")

    tprint("Settings", chr_ignore=True)
    promt = inquirer.checkbox(
        message="Select the settings",
        choices=["Debug mode", "Verbose mode"],
        raise_keyboard_interrupt=False,
        mandatory=False,
        border=True,
        instruction="Use space to select and enter to confirm",
    ).execute()
    if promt:
        if "Debug mode" in promt:
            folder = Path(path / "outputs")
            Settings.path = folder
            # enter path to debug file
            filepath = inquirer.filepath(
                message="Enter the path to the debug file (pass if you want to create a new one)",
                validate=lambda x: Path(
                    x).is_file() and Path(x).suffix == ".txt",
                raise_keyboard_interrupt=False,
                mandatory=False,
                default=str(path / "outputs"),
            ).execute()
            if filepath == None:
                new_file = "debug.txt"
                confirm_promt = inquirer.confirm(message="Do you want to the default file at the location : " + str(
                    folder / new_file), raise_keyboard_interrupt=False, mandatory=False).execute()
                if confirm_promt:
                    open(folder / new_file, "w+")
                    Settings.outfile = new_file
                    Settings.debug = True
                    print("Debug mode enabled")
                else:
                    print("Debug mode disabled")
                    Settings.debug = False
            else:
                Settings.outfile = Path(filepath).relative_to(folder)
                Settings.debug = True
                print("Debug mode enabled")
        if "Verbose mode" in promt:
            Settings.verbose = True

    # Colors
    RED = "\033[1;31m" if not Settings.debug else ""
    GREEN = "\033[1;32m" if not Settings.debug else ""
    YELLOW = "\033[1;33m" if not Settings.debug else ""
    BLUE = "\033[1;34m" if not Settings.debug else ""
    MAGENTA = "\033[1;35m" if not Settings.debug else ""
    CYAN = "\033[1;36m" if not Settings.debug else ""
    WHITE = "\033[1;37m" if not Settings.debug else ""
    BOLD = "\033[1m" if not Settings.debug else ""
    UNDERLINE = "\033[4m" if not Settings.debug else ""
    RESET = "\033[0m" if not Settings.debug else ""

    # Menu
    menu_on = True
    while menu_on:  # Menu loop
        folder = Path(path / "data")
        files = [*map(lambda x: x.relative_to(folder),
                      filter(lambda x: x.is_file(), folder.rglob("*.txt")))]
        # This function lists the files in the folder "FA" which contains all the automaton files
        file_chosen = inquirer.fuzzy(
            message="Which file would you like to import :",  # To chose the file to work on
            choices=files,
            default="",
            raise_keyboard_interrupt=False,
            border=True,
        ).execute()

        print("File chosen : ", file_chosen)
        with open(folder / file_chosen, "r") as file:
            mygraph = Graph.from_file(file)
            print("Graph created")
            mygraph.display(1)
            # tabulate with the matrix header are the states and rows names are the states
            table = mygraph.matrix()
            # convert to list
            table = list(map(list, table))
            # put color on numbers
            for i, row in enumerate(table):
                table[i] = list(map(lambda x: CYAN + str(x) + RESET if x.__class__ == int else x, row))
            print(tabulate(table, headers=[RED + BOLD + state.name + RESET for state in mygraph.states], showindex=[RED + BOLD + state.name + RESET for state in mygraph.states], tablefmt="rounded_grid"))

            # create a calander object
            calander = Calendar(mygraph)

            # make a table with the following columns : rank, state, earliest date, latest date
            ranks = mygraph.ranks()
            earliest_dates = calander.earliest_date()
            latest_dates = calander.latest_date()
            float_dates = calander.float()
            free_float_dates = calander.free_float()
            # order the dictionary by ranks
            ranks = dict(sorted(ranks.items(), key=lambda item: item[1]))
            table = [
                list(ranks.values()),
                [state.name for state in ranks.keys()],
                [state.weight for state in ranks.keys()],
                [earliest_dates[state]
                 for state in ranks.keys()],
                [latest_dates[state]
                 for state in ranks.keys()],
                [float_dates[state] for state in ranks.keys()],
                [free_float_dates[state] for state in ranks.keys()],
            ]
            index = ["rank", "state", "weight", "earliest date", "latest date", "float", "free float"]
            # put headers in first column
            print(tabulate(table, tablefmt="fancy_grid", showindex=index))
            # print the critical path and its weight
            print("Critical path : ", end=" ")
            print(*mygraph.get_critical_path(), sep=" -> ")
            print("Critical path weight : ", sum(
                [state.weight for state in mygraph.get_critical_path()]))
            #! debug for testing
            if sum([state.weight for state in mygraph.get_critical_path()]) != earliest_dates[mygraph.states[-1]]:
                raise Exception("Critical path weight is not equal to the earliest date of the last state")

            # ask if the user wants to compute each possible critical path
            compute = inquirer.confirm(
                message="Do you want to compute each possible critical path ? (this may take some time for large graphs)", raise_keyboard_interrupt=False, mandatory=False).execute()
            if compute:
                for critical in mygraph.get_critial_paths():
                    print(" -", end=" ")
                    print(*critical, sep=" -> ")

            # ask if the user wants to display the graph
            display = inquirer.confirm(
                message="Do you want to display the graph ?", raise_keyboard_interrupt=False, mandatory=False).execute()
            if display:
                calander.display(compute)

        # ask if the user wants to continue
        continue_ = inquirer.confirm(
            message="Do you want to continue ?", raise_keyboard_interrupt=False, mandatory=False).execute()
        if not continue_:
            menu_on = False
            print("Goodbye")
