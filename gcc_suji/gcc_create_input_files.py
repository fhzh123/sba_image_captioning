from utils import create_input_files


if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='gcc',
                       karpathy_json_path='/home/jangsj/dataset_gcc_split.json',
                       image_folder='/HDD/kyohoon/gcc/train/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/HDD/jangsj/',
                       max_len=50)