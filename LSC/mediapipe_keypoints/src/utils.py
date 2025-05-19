import os
import shutil

def create_folder(folder, reset=True, auto=False):
    # Create a folder if it does not exist. If it exists, it can be deleted when reset is True. 
    # This deletion can be done automatically or by asking the user depending on the value of the auto parameter.
    if os.path.exists(folder) and reset==True:
        if auto == False:
            response = input("The folder {} already exists. Do you want to delete it? (y/n): ".format(folder))
            if response.lower() == 'y':
                shutil.rmtree(folder)
        else:
            shutil.rmtree(folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
