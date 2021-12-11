#!/usr/bin/env python

""" Part 1
    This part of the code runs rateMovies.py to check for personal ratings exists or not,
    then loads the data from the correct destination. After these are done it passes the
    configured data into (ratings, movies, sc, myRatings, myRatingsRDD) into ALSmachinelearning.py
    """

# general imports
import sys
from math import sqrt
from operator import add
from os.path import isfile
from pyspark.sql import SparkSession

# import for other .py files
import rateMovies as rMovies
import ALSmachinelearning as als


def parseRating(line):
    """
    Parses a rating record in MovieLens format userId::movieId::rating::timestamp .
    """
    fields = line.strip().split("::")
    return int(fields[3]) % 10, (int(fields[0]),
                                 int(fields[1]), float(fields[2]))


def parseMovie(line):
    """
    Parses a movie record in MovieLens format movieId::movieTitle .
    """
    fields = line.strip().split("::")
    return int(fields[0]), fields[1]


def loadRatings(ratingsFile):
    """
    Load ratings from file.
    """
    if not isfile(ratingsFile):
        print("File %s does not exist." % ratingsFile)
        sys.exit(1)
    f = open(ratingsFile, 'r')
    ratings = [parseRating(line)[1] for line in f]
    f.close()
    if not ratings:
        print("No ratings provided.")
        sys.exit(1)
    else:
        return ratings


def computeRmse(model, data, n):
    """
    Compute RMSE (Root Mean Squared Error).
    """
    predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
    predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
        .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
        .values()
    return sqrt(
        predictionsAndRatings.map(
            lambda x: (
                x[0] -
                x[1]) ** 2).reduce(add) /
        float(n))


if __name__ == "__main__":
    """
    imported rateMovies at the top of the file, we run a function to either input new movies
    rated by the user or we use the current settings which are located in './personalRatings.txt'
    """
    rMovies.runMovies()

    # set up environment
    spark = SparkSession.builder \
        .getOrCreate()

    sc = spark.sparkContext

    # load personal ratings
    myRatings = loadRatings('./personalRatings.txt')
    # create an rdd for myRatings
    myRatingsRDD = sc.parallelize(myRatings, 1)

    # rating and movie data

    # ratings is an RDD of (last digit of timestamp, (userId, movieId, rating))
    ratings = sc.textFile("./movielens/medium/ratings.dat").map(parseRating)

    # movies is an RDD of (movieId, movieTitle, genres)
    movies = dict(
        sc.textFile("./movielens/medium/movies.dat").map(parseMovie).collect())

    # passing required arguments into runALS function located in
    # ALSmachinelearning.py
    als.runALS(ratings, movies, myRatings, sc, myRatingsRDD)