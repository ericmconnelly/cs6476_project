# Final Project - CS6476 - Classification and Detection using Convolutional Neural Networks:

### Google Drive to pretrained weights
https://drive.google.com/drive/folders/10IEdbCWo42hHU6b9k93xgZgqd8GNdooO?usp=sharing

Custom Model : custom_model.json, weights.h5
VGG16 scratch: vgg_16.json, vgg16.h5
VGG16 Pre-trained: vgg_16_pretrained.json, vgg_16_pretrained.h5

All model weight should be in the ./src directory

Both correctly labeled and incorrectly labelled are
in ./data/correct-images and ./src/data/wrong-images

Dataset for training: http://ufldl.stanford.edu/housenumbers/
Format 1 train/test should go to ./src/data/format1/train and ./src/data/format1/test
Format 2 ./src/data/format2/

To run:

cd src/
conda create --file cv_proj.yml
conda activate cs6476_fina
python run.py <path_to_image_file>
