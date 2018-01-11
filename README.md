# Comics Generation
conditional GAN for comics generation

## Requirements
python 3

Keras==2.0.7

Tensorflow==1.3

## Data is not provided in this repository !!!

## Test
python3 generate.py sample_testing_text.txt

Images will be generated in the samples/ directory based on conditions in sample_testing_text.txt. Given one condition, the model will generate 5 images

## Available tags
[color hair]:

'orange hair', 'white hair', 'aqua hair', 'gray hair', 'green hair', 'red hair', 'purple hair', 'pink hair', 'blue hair', 'black hair', 'brown hair', 'blonde hair'.

[color eyes]: 
'gray eyes', 'black eyes', 'orange eyes', 'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes', 'green eyes', 'brown eyes', 'red eyes', 'blue eyes'.

You can change the tags based on the above to see the generated images.

## Demo
Condition1: blue hair red eyes

Condition2: black hair blue eyes

Condition3: red hair green eyes



(Each condition generates 5 images.)

## Training Progress





# Comics Generation without conditions
GAN for comics generation is in z2img/ directory.

