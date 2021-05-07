from google_images_download import google_images_download
from pprint import PrettyPrinter

def read_labels(path):
    label_file = open(path, 'r')
    return [line.rstrip('\n') for line in label_file.readlines()]

def main():
    pp = PrettyPrinter()
    keywords = read_labels('labels.txt')
    chromedriver_path = 'C:/Users/frank/Documents/my-projects/anime-character-recognition/dataset/chromedriver_win.exe'
    keywords = ','.join(keywords)
    #keywords = keywords.rstrip('\n')
    #print("keywords: ", keywords)
    response = google_images_download.googleimagesdownload()   #class instantiation
    arguments = {"keywords":keywords,"limit":500,"print_urls":True, "chromedriver": chromedriver_path}   #creating list of arguments
    pp.pprint(arguments)
    paths = response.download(arguments)   #passing the arguments to the function
    #print(paths)   #printing absolute paths of the downloaded 
    return paths

if __name__ == "__main__":
    main()