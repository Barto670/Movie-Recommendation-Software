To compile this project I used PyCharm IDE.

Operating System : Windows 10 Proffesional (64-bit)

Version of Python : python3.8.3
Version of Spark : 3.1.2
Version of Hadoop : 2.7
Version of Java : 
		java version "1.8.0_301"
		Java(TM) SE Runtime Environment (build 1.8.0_301-b09)
		Java HotSpot(TM) 64-Bit Server VM (build 25.301-b09, mixed mode)

Aditional Environment variables : "PYTHONUNBUFFERED=1"


Packages needed for it to compile:

pip install pandas
pip install pyspark
pip install pyspark[sql]
pip install matplotlib

**The file to run the algorithm is run.py**

If the program is ran correctly this will show in the console:

	"Looks like you've already rated the movies. Overwrite ratings (y/N)?"

If personalRatings.txt does not exist in the folder then you will be asked to fill in personal ratings



