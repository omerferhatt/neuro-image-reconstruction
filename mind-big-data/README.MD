### What is `dataset_organizer.py` doing?

It maps original .csv files with real ILSVRC2013 Training images

    e.g. 
        File name          : ... MindBigData_Imagenet_Insight_n00007846_6247_1_785
        Imagenet file path : ... n00007846/n00007846_6247.JPEG
        
### What is the output of this script?

It saves all mapped and sliced information to a .csv file

|       |path	    |dataset    |device     |wn_id      |im_id      |eeg_session    |global_session |inet_path  |
|---    |---        |---        |---        |---        |---        |---            |---            |---        |
|0      |mind-big-data/data/MindBigData_Imagenet_Insight_n06874185_7720_1_1752.csv|Imagenet|Insight|n06874185|7720|1|1752|~/imagenet/n06874185/n06874185_7720.JPEG
|1      |mind-big-data/data/MindBigData_Imagenet_Insight_n03599486_3112_1_2091.csv|Imagenet|Insight|n03599486|3112|1|2091|~/imagenet/n03599486/n03599486_3112.JPEG
|2      |mind-big-data/data/MindBigData_Imagenet_Insight_n02085620_477_1_995.csv|Imagenet|Insight|n02085620|477|1|995|~/imagenet/n02085620/n02085620_477.JPEG

### How to use and what is these parameters?

Just run script with parameters
   
    e.g.
    $ python3 dataset_organizer.py --data-dir-path mind-big-data/data --local-inet-path ~/imagenet --csv-save-path mind-big-data/dataset.csv

    --data-dir-path:    Path to directory which includes all .csv files in it   (default: mind-big-data/data)
    --local-inet-path:  Path to local ILSVRC 2013 Training folder               (default: None)
    --csv-save-path:    Output save path                                        (default: mind-big-data/dataset.csv)
