import argparse

from ArbitryonMasking.Florence.FlorenceMasking import FlorenceMasking

def main():
    # Create an instance of FlorenceMasking
    florence = FlorenceMasking()
    florence.load_model()

    parser = argparse.ArgumentParser(description="Speak Insight. Get insights from your chats and never make a mistake again.")
    parser.add_argument("-f", "--filepath",type=str, help="The path to the file to read.", default=None)
    parser.add_argument("-fl", "--folderpath",type=str, help="The path to the folder to read files from.", default=None)

    args = parser.parse_args()

    if args.filepath:
        florence.get_mask(args.filepath)
    elif args.folderpath:
        florence.get_mask_from_folder(args.folderpath)
    else:
        print("Usage Error: Please provide either a file path or a folder path.")

if __name__ == "__main__":
    main()
