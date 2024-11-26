""" A small collection of utilities for displaying things in the terminal and getting user input.
"""

from typing import List

class Display:
    @staticmethod
    def page(title : str, content : str) -> None:
        """ Display a page with a title and content

        Args:
            title (str): The title of the page
            content (str): The content of the page
        """
        # Display the title
        print(f"\n{title}\n")
        
        # Display the content
        print(content)
    
    @staticmethod
    def menu(title : str, options : List[str]) -> int:
        """ Display a menu with a title and options

        Args:
            title (str): The title of the menu
            options (List[str]): The options to display

        Returns:
            int: The index of the selected option
        """
        # Display the title
        print(f"\n{title}\n")
        
        # Display the options
        for idx, option in enumerate(options):
            print(f"{idx + 1}. {option}")
        
        # Get the user's choice
        choice = -1
        while choice < 0 or choice >= len(options):
            try:
                choice = int(input(f"\nChoose an option (1-{len(options)}): ")) - 1
            except ValueError:
                pass
        
        return choice
    
    @staticmethod
    def get_int(title : str) -> int:
        """ Get a number from the user

        Args:
            title (str): The title of the number to get

        Returns:
            int: The number entered by the user
        """
        # Get the user's number
        number = -1
        while number < 0:
            try:
                number = int(input(f"\n{title}: "))
            except ValueError:
                pass
        
        return number
    
    @staticmethod
    def get_float(title : str) -> float:
        """ Get a decimal number from the user

        Args:
            title (str): The title of the number to get

        Returns:
            float: The number entered by the user
        """
        # Get the user's number
        number = -1.0
        while number < 0.0:
            try:
                number = float(input(f"\n{title}: "))
            except ValueError:
                pass
        
        return number