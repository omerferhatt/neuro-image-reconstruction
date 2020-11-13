### What is `dataset_organizer.py` doing?

It maps original .csv files with real ILSVRC2013 Training images

    e.g. 
    
    File name          : [--data-dir-path]/MindBigData_Imagenet_Insight_n00007846_6247_1_785.csv
    Imagenet file path : [--local-inet-path]/n00007846/n00007846_6247.JPEG
        
### What is the output of this script?

It saves all mapped and sliced information to a .csv file

|       |path	    |dataset    |device     |wn_id      |im_id      |eeg_session    |global_session |inet_path  |
|---    |---        |---        |---        |---        |---        |---            |---            |---        |
|0      |data/mind-big-data/MindBigData_Imagenet_Insight_n06874185_7720_1_1752.csv|Imagenet|Insight|n06874185|7720|1|1752|data/imagenet/n06874185/n06874185_7720.JPEG
|1      |data/mind-big-data/MindBigData_Imagenet_Insight_n03599486_3112_1_2091.csv|Imagenet|Insight|n03599486|3112|1|2091|data/imagenet/n03599486/n03599486_3112.JPEG
|2      |data/mind-big-data/MindBigData_Imagenet_Insight_n02085620_477_1_995.csv|Imagenet|Insight|n02085620|477|1|995|data/imagenet/n02085620/n02085620_477.JPEG

### If Imagenet images will be copied?

In dataset.csv file, `inet_path` column will be updated with new copied paths
    
    e.g. 
    
    (Old)   [--local-inet-path]/n06874185/n06874185_7720.JPEG
    (New)   [--copy-imagenet]/n06874185/n06874185_7720.JPEG

### How to use and what is these parameters?

Just run script with parameters
   
    e.g.
    
    $ python3 dataset_organizer.py \
            --data-dir-path data/mind-big-data \
            --local-inet-path data/imagenet \
            --csv-save-path data/dataset.csv \
            --copy-imagenet data/mind-big-data-imagenet


    --data-dir-path     : Path to directory which includes all .csv files in it     (default: data/mind-big-data)
    --local-inet-path   : Path to local ILSVRC 2013 Training folder                 (default: None)
    --copy-imagenet     : Path to new directory to copy related images,             (default: None)
                          don't use this parameter if you don't want to copy.

    --csv-save-path     : Output save path                                          (default: data/dataset.csv)
