def select_option():
    while True:
        option = input("Select a crop type:\n1. Upper Body\n2. Face\n3. Full Body\nSelect: ")
        if option not in ["1", "2", "3"]:
            print("Invalid option selected. Please try again.")
            print("")
        else:
            break
    return option