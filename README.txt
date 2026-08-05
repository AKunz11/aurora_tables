Aurora Tables
Alexander Kunz

Some helpful functions to accomodate my periapse_quicklooks repo.
This takes data from periapse_quicklooks files and combines the outputted csv's into a "master" csv.
Additionally, it creates an ML set that was used for something at some point in time (not really important)

This master csv is used in just about all my other aurora repos, especially in:
- aurora_tools
- cosmic_ray_CNN

So its not a bad idea to just utilize this. It'll save you some time.

File descriptions:

combine_ql_tables.py
Combines the tables outputted by periapse_quicklooks code

merge_class_tables.py
Adds the "by-eye" labels for aurora that were reviewed

drop_false.py
Drops the false detections seen via the "by-eye" inspection

make_master_table.py
Does all three above at once
This was added later, which is why I have the other files

all_detections.py
This is a special one.
Pretty much makes a periapse_quicklooks style table for EVERY SINGLE SCAN I have data for.
This takes hours to run, do not recommend 


Running order
1. Combine
2. Merge
3. Drop
OR alternatively, just run the make_master_table.py


Note:
I added example csvs to show how the table structure works
For orbits 8000-8100:

example_ql_table.csv
    A table the periapse_quicklooks code produces

example_class.csv
    An example of the by-eye inspection.
    Pearson R header is actually not necessary, no need to include it

example_false.csv
    A table with false positive detections
    Labeled as orbit.scan (zero padded)
