This software is written in Python™ 3.8.

MitoTrack_v4_0 was developed on the operative system Linux Ubuntu 20.04 64-bit
    and tested on Linux Ubuntu 18.04 and 20.04 64-bit, Windows 8.1 Professional, Windows 10 Professional and
    McOs High Sierra 10.13.6.

 
- Run MitoTrack_v4_0:

	Unzip the file MitoTrack_v4_0.zip and put all the files in a folder, then open a terminal (or the cmd command 
	prompt for windows users), go into the folder and type:
	
	python3 MitoTrack_v4_0.py
    
    and press enter. The graphical user interface will pop up and let you work.

- MitoTrack_v4_0 is a software package that allows to follows transcriptional activation during early drosophila embryo across
    2 consecutive cell cycle keeping track of the cell lineage. It is a development of MitoTrack v1, already published 
    (https://doi.org/10.1093/bioinformatics/btz717).
    With respect to the publish version, MitoTrack_v4_0:
    1) has a better tool for nuclei segmentation in the 'during mitosis' part
    2) detects transcriptional sites in 3D
    3) can track and remove mitotical transcriptional sites
    4) adds transctpional sites intensity to the table of the final results
    5) is better optimized and reduces the size of the analysis folder 
    
- Possible issues:

    Depending on the size of your data and the specifications of your computer,
    you can have a MemoryError. In this case try to shut down all the other tasks 
    your pc is running and eventually crop your data.	
    
           
For any question or issue send an email at:
    antonio.trullo@igmm.cnrs.fr