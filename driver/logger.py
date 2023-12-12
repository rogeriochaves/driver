from colorama import Fore, Style


def print_action(str: str):
    print(Fore.YELLOW + "\n\n> " + str + "\n" + Style.RESET_ALL)
