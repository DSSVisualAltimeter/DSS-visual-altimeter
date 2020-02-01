import os 
  
# Function to rename multiple files 
def main(): 
    i = 0
    
    for filename in os.listdir("C:/Users/Alec Otterson/AlecDocs/AI/DroneAlt/data/train/two_feet"): 
        dst ="2." + str(i) + ".png"
        src ='C:/Users/Alec Otterson/AlecDocs/AI/DroneAlt/data/train/two_feet/'+ filename 
        dst ='C:/Users/Alec Otterson/AlecDocs/AI/DroneAlt/data/train/two_feet/'+ dst 
          
        # rename() function will 
        # rename all the files 
        os.rename(src, dst) 
        i += 1
  
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 