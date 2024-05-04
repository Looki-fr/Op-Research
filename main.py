from pathlib import Path
from InquirerPy import inquirer
from art import *
from logger import Settings, print, clear_color
from tools import *

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

    if Settings.debug:
        clear_color()

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
            table = TransportationTable.from_file(file)

            # Display the table
            print("Table :")
            table.display()

            # Ask the user to chose the method
            # Nord-West corner method
            # North-West corner method + optimization
            # Balas-Hammer method
            # Balas-Hammer method + optimization

            method = inquirer.select(
                message="Which method would you like to use ?",
                choices=[
                    "Nord-West corner method",
                    "Nord-West corner method + optimization",
                    "Balas-Hammer method",
                    "Balas-Hammer method + optimization",
                ],
                raise_keyboard_interrupt=False,
                border=True,
            ).execute()

            match method:
                case "Nord-West corner method":
                    table.NorthWestCorner()
                case "Nord-West corner method + optimization":
                    table.NordWestOptimized()
                case "Balas-Hammer method":
                    table.BalasHammer()
                case "Balas-Hammer method + optimization":
                    table.BalasHammerOptimized()

            # Display the result
            print("Result :")
            table.show(table.transportation_table)
            # Display the marginal costs
            print("Marginal costs :")
            matrix, (s, t) = table.marginal_costs_force()
            table.show(matrix, s, t, "Potential", "Potential")
            # print the total cost
            print("Total cost : ", table.total_cost())

            # Ask the user if he wants to display the graph
            display_graph = inquirer.confirm(
                message="Do you want to display the graph ?", raise_keyboard_interrupt=False, mandatory=False).execute()
            if display_graph:
                table.graph.display(weights=table.transportation_table)

        # ask if the user wants to continue
        continue_ = inquirer.confirm(
            message="Do you want to continue ?", raise_keyboard_interrupt=False, mandatory=False).execute()
        if not continue_:
            menu_on = False
            print("Goodbye")
