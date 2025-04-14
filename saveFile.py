import pickle


def saveResults(fileName, *args, **kwargs):
    f = open(fileName, 'w')
    for i in args:
        f.writelines(str(i) + '\n')
    f.close()
    return


def saveLog(fileName, log):
    f = open(fileName, 'wb')
    pickle.dump(log, f)
    f.close()
    return


def bestInd(toolbox, population, number):
    bestInd = []
    best = toolbox.selectElitism(population, k=number)
    for i in best:
        bestInd.append(i)
    return bestInd


def saveAllResults(randomSeeds, dataSetName, best_ind_va, hof, trainTime, testTime, testResults, classifier, features):
    fileName = str(randomSeeds) + 'FinalResult' + dataSetName + '.txt'
    saveResults(fileName, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                'trainResults', hof[0].fitness,
                'testTime', testTime, 'testResults', testResults, 'bestInd in training',
                hof[0], 'Best individual',
                *best_ind_va[:], 'classifier: 1-lsvm, 2-rf, 3-lr, 4-erf', classifier,
                'the size of the best individual', len(best_ind_va[0]) - 1
                )

    return


def saveAllResults_new(randomSeeds, dataSetName, best_ind_va, log, hof,
                       trainTime, testTime, testResults, classifier, features):
    save_path = "E:/pycode/i_c/results/"  #保存结果的路径
    fileName1 = str(randomSeeds) + 'Results_on' + dataSetName + '.txt'
    saveLog(fileName1, log)
    fileName2 = str(randomSeeds) + 'Final_Result_son' + dataSetName + '.txt'
    saveResults(fileName2, 'randomSeed', randomSeeds, 'trainTime', trainTime,
                'trainResults', hof[0].fitness,
                'testTime', testTime, 'testResults', testResults, 'bestInd in training',
                hof[0], 'Best individual in each run',
                *best_ind_va[:], 'final best fitness', best_ind_va[-1].fitness,
                'initial fitness', best_ind_va[0].fitness, 'classifier: 1-lsvm, 2-rf, 3-lr, 4-erf', classifier,
                'the size of the best individual in training', len(best_ind_va[0]) - 1,
                'the number of features for classification:', len(features))
    return
