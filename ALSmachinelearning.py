""" Part 2
    we set up the actual RDDs that we will use for the machine learning algorithm
    at this point spark can be accessed at http://localhost:4040/ , we can
    use this to see how python executes commands and loads data
    """

# general imports
import itertools
from math import sqrt
from operator import add
import matplotlib.pyplot as plt

# import ALS from MLlib.
from pyspark.mllib.recommendation import ALS

# import for other .py files
import run as run


# using the following parameters that were passed from run.py to setup the
# values for machine learning
def runALS(ratings, movies, myRatings, sc, myRatingsRDD):
    # count the number of values inside ratings for ratings,users and movies
    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()

    # print the above values in the console
    print(
        "Ratings= " +
        str(numRatings) +
        " Users= " +
        str(numUsers) +
        " Movies= " +
        str(numMovies) +
        "\n")

    # excercise 2) Create Train and Test datasets for your ML Model.

    numPartitions = 4
    # ratings currently has 1 million lines
    # we split this data into training and validation and test
    # all of these data subsets are not overlapping
    # training having around 6/10 of the results
    training = ratings.filter(lambda x: x[0] < 6).values().union(
        myRatingsRDD).repartition(numPartitions).cache()

    # training having around 2/10 of the results
    validation = ratings.filter(
        lambda x: 6 <= x[0] < 8).values().repartition(numPartitions).cache()

    # test will contain the remaining values that training and validation
    # doesn't have
    test = ratings.filter(lambda x: x[0] >= 8).values().cache()

    # we count how many training/validation/test entries we have
    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    # print the above values in the console
    print(
        "Number of Trainings= " +
        str(numTraining) +
        " Number of Validations=  " +
        str(numValidation) +
        " Number of Tests= " +
        str(numTest) +
        "\n")

    # excercise 3) Train the model using ALS from MLlib.

    # we set up the required data needed for the for loop, ALS.train and
    # validation Rmse
    ranks = [8, 12]
    lambdas = [0.1, 10.0]
    numIters = [5, 15]
    bestModel = None
    bestValidationRmse = float("inf")
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = -1

    print("\n")

    # variables to for matplotlib graph
    x = []
    y = []
    i = 0

    # setting up matplotlib
    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('Training No.')
    # naming the y axis
    plt.ylabel('Improvement %')
    # giving a title to my graph
    plt.title('Improvement Graph')

    for rank, lmbda, numIter in itertools.product(ranks, lambdas, numIters):
        i = i + 1
        model = ALS.train(training, rank, numIter, lmbda)

        validationRmse = run.computeRmse(model, validation, numValidation)

        # print the above values in the console
        print(
            "RMSE = (" +
            str(validationRmse) +
            ") for the model trained with Rank= " +
            str(rank) +
            " Lambda= " +
            str(lmbda) +
            " Number of Iters= " +
            str(numIter))

        # if the current model we tested is better than the last we replace the
        # values with the new best model
        if (validationRmse < bestValidationRmse):
            bestModel = model
            bestValidationRmse = validationRmse
            bestRank = rank
            bestLambda = lmbda
            bestNumIter = numIter

        # calculating testRMSE to use in the improvement equation
        testRmse = run.computeRmse(bestModel, test, numTest)

        # calculating meanRating to use in the baselineRmse
        meanRating = training.union(validation).map(lambda x: x[2]).mean()

        # calculating baselineRmse to use in the improvement equation
        baselineRmse = sqrt(
            test.map(
                lambda x: (
                    meanRating -
                    x[2]) ** 2).reduce(add) /
            numTest)

        # this is the calculated improvement, this value is used to be
        # displayed in the console and also plotted onto the graph
        improvement = (baselineRmse - testRmse) / baselineRmse * 100

        # we append the current i and improvement % to the array so we can display it after
        # the machine learning part is finished
        x.append(i)
        y.append(improvement)
        plt.plot(x, y)

    print("\n")

    # print the above value in the console
    print("The best model improved by= " + str(improvement) + "\n")

    # myRatings contains all the data and we only need the ids, so we create a
    # variable to store these ids in a set
    myRatedMovieIds = set([x[1] for x in myRatings])

    # making sure that the movies in our list will not get recommended to us
    # again
    candidates = sc.parallelize(
        [m for m in movies if m not in myRatedMovieIds])

    # we take our bestModel generated by the machine learning and
    # we apply our bestModel to the movies stored in candidates
    predictions = bestModel.predictAll(
        candidates.map(lambda x: (0, x))).collect()

    # Sort the data in reverse order and only show top 5 movies
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:5]

    # excercise 4/5) Return recommendation in the following format.

    print("Movies recommended for you= ")
    # iterate through recommendations and print the value in the correct format
    for i in range(len(recommendations)):
        print(str(i + 1) + ": " +
              str((movies[recommendations[i][1]])))

    # excercise 6) Report on the Accuracy of the model.
    print(
        "Best Rank= " +
        str(bestRank) +
        " Best Lambda= " +
        str(bestLambda) +
        " Best Iteration= " +
        str(bestNumIter) +
        " Best RMSE= " +
        str(testRmse))

    plt.show()

    sc.stop()