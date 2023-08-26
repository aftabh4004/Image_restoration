import os
import csv
import random
root = '/home/mt0/22CS60R54/image_regen/Dataset'
root_list = '/home/mt0/22CS60R54/image_regen/dataset_list'

def create_dataset_list():
    with open(os.path.join(root_list, "photo-reconstruction.txt"), "w") as fp:
    
        for category in  ["Cat", "Dog", "Elephant", "Tiger"]:
            masked_path = os.path.join(root, "Training_Data", category, "Masked_Train")
            unmasked_path = os.path.join(root, "Training_Data", category, "Unmasked_Train")

            with open(os.path.join(masked_path, "masked_info.csv"), 'r') as fin:
                lines = csv.reader(fin)
                for line in lines:
                    try:
                        _, img, b1r, b1c, b2r, b2c = line

                        masked_image_path = os.path.join(masked_path, img)
                        unmasked_image_path = os.path.join(unmasked_path, img)
                        if(os.path.exists(masked_image_path) and os.path.exists(unmasked_image_path)):
                            print(",".join([masked_image_path, unmasked_image_path, b1r, b1c, b2r, b2c]), file=fp)
                    except Exception as e:
                        print(f'skiped {line}')
                        print(e)



def create_train_val_test_list():
    with open(os.path.join(root_list, "photo-reconstruction.txt"), "r") as fp, \
        open(os.path.join(root_list, "train.txt"), "w") as ftrain, \
        open(os.path.join(root_list, "test.txt"), "w") as ftest, \
        open(os.path.join(root_list, "val.txt"), "w") as fval:
        
        lines = fp.readlines()
        random.seed(42)
        random.shuffle(lines)

        train_split_index = int(len(lines) * 0.8)
        train_val, test = lines[:train_split_index], lines[train_split_index:]
        
        val_split_index = int(len(train_val) * 0.9)
        train, val = train_val[:val_split_index], train_val[val_split_index:]

        ftrain.writelines(train)
        ftest.writelines(test)
        fval.writelines(val)

def main():
    # create_dataset_list()
    create_train_val_test_list()



    

if __name__ == "__main__":
    main()