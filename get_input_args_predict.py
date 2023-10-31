import argparse

def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these 3 command line arguments. If 
    the user fails to provide some or all of the 3 arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as 'image_path'
      2. Chekpoint positional argument as 'checkpoint'
      3. Top Probabilities as '--num_top_prob' with default=5
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Creating Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type=str)
    parser.add_argument('checkpoint')
    parser.add_argument('--gpu', type=str, default='cuda:0', help = 'choose to run model on GPU or CPU')
    parser.add_argument('--json_file', type=str, default='cat_to_name.json', help='a JSON file that maps the class values to other category names')
    parser.add_argument('--num_top_prob', type=int, default = 5, help = 'number of top probabilities to predict')
     
    return parser.parse_args()